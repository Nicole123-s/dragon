import os
import re
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from dragon_baseline import DragonBaseline
from dragon_baseline.nlp_algorithm import ProblemType

import llm_fusion


class DragonSubmission(DragonBaseline):
    """混合提交方案：

    - 编码器微调（RoBERTa Large 领域特定）作为全部 28 个任务的主力
    - 对弱任务（T06/T08/T12/T13/T14/T24）额外做零样本 LLM 提取，并在验证集上
      搜索融合权重 λ 后与编码器预测融合
    - LLM 不可用/失败时自动回退纯编码器结果
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 编码器：官方最强的领域特定预训练 Large 模型（论文总分 0.819）
        # 本地调试可用 ENCODER_MODEL 环境变量临时换回小模型（如 dragon-roberta-base-mixed-domain）
        self.model_name = os.environ.get(
            "ENCODER_MODEL", "joeranbosma/dragon-roberta-large-domain-specific"
        )
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 2
        self.gradient_checkpointing = False
        self.max_seq_length = 512
        self.learning_rate = 1e-05
        self.num_train_epochs = 5

        # LLM 融合设置（可用环境变量覆盖，便于本地调试）
        self.llm_enabled = os.environ.get("DISABLE_LLM", "0") != "1"
        self.llm_model = os.environ.get("LLM_MODEL", "gemma2:9b")
        self.llm_num_predict = int(os.environ.get("LLM_NUM_PREDICT", "512"))
        # 少样本示例数：仅对标量标签任务（二分类/多分类）启用
        self.llm_num_examples = int(os.environ.get("LLM_NUM_EXAMPLES", "3"))
        self._llm_lambda = None  # 验证集上选出的融合权重（编码器权重）

    def custom_text_cleaning(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Perform custom text cleaning on the input text.

        Args:
            text (Union[str, List[str]]): The input text to be cleaned. It can be a string or a list of strings.

        Returns:
            Union[str, List[str]]: The cleaned text. If the input is a string, the cleaned string is returned.
            If the input is a list of strings, a list of cleaned strings is returned.

        """
        if isinstance(text, str):
            # Remove HTML tags and URLs:
            text = re.sub(r"<.*?>", "", text)
            text = re.sub(r"http\S+", "", text)

            return text
        else:
            # If text is a list, apply the function to each element
            return [self.custom_text_cleaning(t) for t in text]

    def preprocess(self):
        # Example of how to adapt the DRAGON baseline to use a different preprocessing function
        super().preprocess()

        # Uncomment the following lines to use the custom_text_cleaning function
        # for df in [self.df_train, self.df_val, self.df_test]:
        #     df[self.task.input_name] = df[self.task.input_name].map(self.custom_text_cleaning)

    # ---------------- LLM 融合 ----------------

    def _llm_applicable(self) -> bool:
        """当前任务是否启用 LLM 融合。"""
        if not self.llm_enabled:
            return False
        try:
            task_id = llm_fusion.task_id_from_name(self.task.task_name)
        except ValueError:
            return False
        if task_id not in llm_fusion.WEAK_TASK_IDS:
            return False
        return self.task.target.problem_type in (
            ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION,
            ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION,
            ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION,
            ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION,
            ProblemType.MULTI_LABEL_REGRESSION,
        )

    def _llm_workdir(self, task_id: str, split: str) -> Path:
        return Path(self.model_save_dir).parent / "llm" / f"task{task_id}_{split}"

    def _extract_llm(self, task_id: str, df: pd.DataFrame) -> dict:
        # 少样本仅对标量标签任务启用（结构化输出任务保持零样本）
        problem_type_str = self.task.target.problem_type.value
        num_examples = (
            self.llm_num_examples
            if problem_type_str in llm_fusion.FEWSHOT_PROBLEM_TYPES
            else 0
        )
        return llm_fusion.run_llm_extraction(
            task_id=task_id,
            problem_type=self.task.target.problem_type,
            df=df,
            input_name=self.task.input_name,
            workdir=self._llm_workdir(task_id, "val" if df is self.df_val else "test"),
            model_name=self.llm_model,
            num_predict=self.llm_num_predict,
            train_df=self.df_train,
            num_examples=num_examples,
            label_name=self.task.target.label_name,
        )

    def _predict_encoder_distribution(self, df: pd.DataFrame):
        """单标签多分类任务：返回 (uids, classes, softmax 概率矩阵)。"""
        import torch
        from scipy.special import softmax
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_save_dir, truncation_side=self.task.recommended_truncation_side
        )
        tokenizer.model_max_length = self.max_seq_length
        model = AutoModelForSequenceClassification.from_pretrained(self.model_save_dir).to(self.device)
        model.eval()

        classes = [model.config.id2label[i] for i in sorted(model.config.id2label)]
        probs = []
        with torch.no_grad():
            for _, row in df.iterrows():
                inputs = tokenizer(
                    row[self.task.input_name], return_tensors="pt", truncation=True
                ).to(self.device)
                logits = model(**inputs).logits.detach().cpu().numpy()[0]
                probs.append(softmax(logits))
        return df["uid"].astype(str).tolist(), classes, np.asarray(probs)

    def _tune_lambda(self, task_id: str) -> float:
        """在验证集上搜索融合权重 λ（编码器权重）。"""
        pred_name = self.task.target.prediction_name
        label_name = self.task.target.label_name
        uids = self.df_val["uid"].astype(str).tolist()
        y_true = self.df_val.set_index(self.df_val["uid"].astype(str)).loc[uids, label_name]

        llm_val = self._extract_llm(task_id, self.df_val)

        if self.task.target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
            enc_val = super().predict(df=self.df_val)
            p_enc = enc_val.set_index(enc_val["uid"].astype(str)).loc[uids, pred_name]
            q_llm = [llm_val.get(u) for u in uids]
            lam = llm_fusion.tune_lambda_binary(y_true.astype(int), p_enc, q_llm)
        elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
            _, classes, enc_dist = self._predict_encoder_distribution(self.df_val)
            llm_labels = [llm_val.get(u) for u in uids]
            lam = llm_fusion.tune_lambda_multiclass(
                y_true.astype(str), enc_dist, classes, llm_labels, task_id
            )
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
            enc_val = super().predict(df=self.df_val)
            enc_vals = enc_val.set_index(enc_val["uid"].astype(str)).loc[uids, pred_name]
            llm_vals = [llm_val.get(u) for u in uids]
            lam = llm_fusion.tune_lambda_mlbinary(y_true, enc_vals, llm_vals)
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
            enc_val = super().predict(df=self.df_val)
            enc_labels = enc_val.set_index(enc_val["uid"].astype(str)).loc[uids, pred_name]
            llm_labels = [llm_val.get(u) for u in uids]
            lam = llm_fusion.tune_lambda_mlmc(y_true, enc_labels, llm_labels, task_id)
        else:  # MULTI_LABEL_REGRESSION
            enc_val = super().predict(df=self.df_val)
            enc_vals = enc_val.set_index(enc_val["uid"].astype(str)).loc[uids, pred_name]
            llm_vals = [llm_val.get(u) for u in uids]
            epsilon = llm_fusion.EPSILON_BY_TASK.get(task_id, 1.0)
            lam = llm_fusion.tune_lambda_mlreg(y_true, enc_vals, llm_vals, epsilon)

        print(f"[llm_fusion] task={task_id} 验证集选出 λ(编码器权重)={lam}")
        return lam

    def _blend(self, task_id: str, df: pd.DataFrame, encoder_pred: pd.DataFrame, llm_pred: dict, lam: float) -> pd.DataFrame:
        """按任务类型融合编码器与 LLM 预测。"""
        pred_name = self.task.target.prediction_name
        uids = df["uid"].astype(str).tolist()
        out = encoder_pred.copy()
        out["uid"] = out["uid"].astype(str)
        out = out.set_index("uid").loc[uids].reset_index()

        if self.task.target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
            p_enc = out[pred_name].astype(float).values
            q_llm = [llm_pred.get(u) for u in uids]
            out[pred_name] = llm_fusion.blend_binary(p_enc, q_llm, lam)
        elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
            _, classes, enc_dist = self._predict_encoder_distribution(df)
            llm_labels = [llm_pred.get(u) for u in uids]
            out[pred_name] = llm_fusion.blend_multiclass(enc_dist, classes, llm_labels, lam)
        elif self.task.target.problem_type in (
            ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION,
            ProblemType.MULTI_LABEL_REGRESSION,
        ):
            enc_vals = out[pred_name].values
            llm_vals = [llm_pred.get(u) for u in uids]
            out[pred_name] = llm_fusion.blend_mlregression(enc_vals, llm_vals, lam)
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
            enc_labels = out[pred_name].values
            llm_labels = [llm_pred.get(u) for u in uids]
            out[pred_name] = llm_fusion.blend_mlmc(enc_labels, llm_labels, lam)
        return out

    def predict(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """编码器预测为主；弱任务叠加 LLM 提取并按验证集权重融合。"""
        if not self._llm_applicable():
            return super().predict(df=df)

        task_id = llm_fusion.task_id_from_name(self.task.task_name)
        encoder_pred = super().predict(df=df)
        try:
            if df is self.df_test and self._llm_lambda is None:
                self._llm_lambda = self._tune_lambda(task_id)
            lam = self._llm_lambda if self._llm_lambda is not None else 1.0
            llm_pred = self._extract_llm(task_id, df)
            n_valid = sum(v is not None for v in llm_pred.values())
            print(f"[llm_fusion] task={task_id} model={self.llm_model} λ={lam} LLM 有效 {n_valid}/{len(df)}")
            return self._blend(task_id, df, encoder_pred, llm_pred, lam)
        except Exception as e:
            print(f"[llm_fusion] LLM 提取失败，回退纯编码器结果: {e}")
            return encoder_pred


if __name__ == "__main__":
    DragonSubmission().process()
