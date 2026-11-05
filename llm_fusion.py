"""LLM 融合模块：弱任务的零样本 LLM 提取 + 与编码器预测的融合。

设计要点：
- 仅对 WEAK_TASK_IDS 内的任务启用；LLM 任何环节失败由调用方回退编码器结果
- 融合权重 λ（编码器权重）在验证集上网格搜索，按任务官方指标选型
- 逐行校验 LLM 输出（status/取值合法），无效行保留编码器预测
"""
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# tiktoken 离线缓存（镜像自官方 LLM 模板仓库），须在 import llm_extractinator 之前设置
os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", str(Path(__file__).parent / "tiktoken_cache")
)

WEAK_TASK_IDS = {
    "003", "004", "006", "007", "008", "009",
    "012", "013", "014", "016", "017", "024",
}
# 本地调试用：LLM_EXTRA_TASKS="102,106" 可把合成任务临时纳入弱任务（须 llm_tasks/ 有对应提示词）
WEAK_TASK_IDS |= set(
    x for x in os.environ.get("LLM_EXTRA_TASKS", "").split(",") if x.strip()
)
# 少样本（few-shot）只对标量标签任务启用；结构化输出（多标签）任务保持零样本
FEWSHOT_PROBLEM_TYPES = ("single_label_binary_classification", "single_label_multi_class_classification")
BINARY_TASK_IDS = {"006", "008"}
MULTICLASS_TASK_IDS = {"012", "013", "014"}
ML_REGRESSION_TASK_IDS = {"024"}
# T14 类别有序（线性加权 Kappa）；T12/T13 无序（普通 Kappa）
LINEAR_KAPPA_TASK_IDS = {"014"}
# T24 RSMAPE 的 epsilon（来自任务页，单位 mm）
EPSILON_BY_TASK = {"024": 4.0}

LAMBDA_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)

LLM_TASK_DIR = Path(__file__).parent / "llm_tasks"


def task_id_from_name(task_name: str) -> str:
    """从任务名提取三位任务编号，如 Task014_textual_entailment_clf -> '014'。"""
    match = re.search(r"(\d{3})", task_name)
    if not match:
        raise ValueError(f"无法从任务名 '{task_name}' 提取任务编号")
    return match.group(1)


def prepare_llm_text(df: pd.DataFrame, input_name: str, task_id: str) -> pd.Series:
    """把任务输入整理成提示用文本（与官方 LLM 模板的 preprocess 对齐）。

    text_parts 任务：T14 为 NLI，两个句子分开标注；其余拼接为整段。
    """
    if input_name == "text":
        return df["text"].astype(str)
    if task_id == "014":
        return df["text_parts"].apply(
            lambda parts: "Sentence 1: " + parts[0] + "\n\nSentence 2: " + parts[1]
        )
    return df["text_parts"].apply(lambda parts: " ".join(str(p) for p in parts))


def parse_row(problem_type, row: dict):
    """按问题类型解析 extractinate 的一行输出；无效（失败/非法）返回 None。

    按 problem_type 而非任务编号判断类型，任意任务（含合成任务）均可工作。
    """
    from dragon_baseline.nlp_algorithm import ProblemType as PT

    if row.get("status") != "success":
        return None
    try:
        if problem_type == PT.SINGLE_LABEL_BINARY_CLASSIFICATION:
            label = row.get("label")
            if label in (True, "True", "true", "1", 1):
                return 1.0
            if label in (False, "False", "false", "0", 0):
                return 0.0
            return None
        if problem_type == PT.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
            label = row.get("label")
            return str(label) if label not in (None, "") else None
        if problem_type == PT.MULTI_LABEL_BINARY_CLASSIFICATION:
            keys = ["lesion_1", "lesion_2", "lesion_3", "lesion_4", "lesion_5"]
            return [1.0 if row[key] in (True, "True", "true", 1) else 0.0 for key in keys]
        if problem_type == PT.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
            if "attenuation" in row and "location" in row:
                return [str(row["attenuation"]), str(row["location"])]
            label = row.get("label")
            return [str(v) for v in label] if isinstance(label, list) else None
        if problem_type == PT.MULTI_LABEL_REGRESSION:
            keys = ["lesion_1", "lesion_2", "lesion_3", "lesion_4", "lesion_5"]
            return [float(row[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return None
    return None


def example_output_json(problem_type, label):
    """把训练标签序列化为 few-shot 示例的期望输出 JSON 字符串。"""
    import json as _json

    from dragon_baseline.nlp_algorithm import ProblemType as PT

    if problem_type == PT.SINGLE_LABEL_BINARY_CLASSIFICATION:
        return _json.dumps({"reasoning": "", "label": bool(label)})
    if problem_type == PT.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
        return _json.dumps({"reasoning": "", "label": str(label)})
    if problem_type == PT.MULTI_LABEL_BINARY_CLASSIFICATION:
        keys = ["lesion_1", "lesion_2", "lesion_3", "lesion_4", "lesion_5"]
        return _json.dumps({"reasoning": "", **{k: bool(v) for k, v in zip(keys, label)}})
    if problem_type == PT.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
        return _json.dumps({"reasoning": "", "attenuation": str(label[0]), "location": str(label[1])})
    if problem_type == PT.MULTI_LABEL_REGRESSION:
        keys = ["lesion_1", "lesion_2", "lesion_3", "lesion_4", "lesion_5"]
        return _json.dumps({"reasoning": "", **{k: float(v) for k, v in zip(keys, label)}})
    return None


def run_llm_extraction(
    *,
    task_id: str,
    problem_type,
    df: pd.DataFrame,
    input_name: str,
    workdir: Path,
    model_name: str,
    num_predict: int = 512,
    train_df: pd.DataFrame = None,
    num_examples: int = 0,
    label_name: str = None,
) -> dict:
    """对给定数据运行 LLM 提取（零样本或少样本），返回 {uid: 预测值或 None}。

    num_examples>0 时从训练集选取相似示例注入提示词（few-shot），
    复用 llm_extractinator 的提示词、结构化输出与失败重试机制。
    """
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / "data").mkdir(parents=True)
    (workdir / "tasks").mkdir(parents=True)
    (workdir / "output").mkdir(parents=True)

    # 本地调试：LLM_SUBSET=30 时仅对前 30 条做 LLM 提取（生产环境勿设）
    subset = os.environ.get("LLM_SUBSET")
    if subset:
        n = int(subset)
        print(f"[llm_fusion] 本地测试模式：仅对前 {n}/{len(df)} 条做 LLM 提取")
        df = df.head(n).copy()

    data = pd.DataFrame({"uid": df["uid"].astype(str).values})
    data["text"] = prepare_llm_text(df, input_name, task_id).values
    data.to_json(workdir / "data" / "test.json", orient="records")

    task_jsons = list(LLM_TASK_DIR.glob(f"Task{task_id}_*.json"))
    if not task_jsons:
        raise FileNotFoundError(f"llm_tasks/ 下找不到 Task{task_id} 的提示词 JSON")
    task_json_dest = workdir / "tasks" / task_jsons[0].name
    shutil.copy(task_jsons[0], task_json_dest)

    example_dir = workdir / "examples"
    if num_examples > 0 and train_df is not None and len(train_df) > 0 and label_name:
        # 构造 few-shot 示例库：input=报告文本, output=期望 JSON 输出
        examples = pd.DataFrame({
            "text": prepare_llm_text(train_df, input_name, task_id).values,
            "input": prepare_llm_text(train_df, input_name, task_id).values,
        })
        examples["output"] = [
            example_output_json(problem_type, row[label_name])
            for _, row in train_df.iterrows()
        ]
        examples = examples[examples["output"].notna()]
        example_dir.mkdir(parents=True, exist_ok=True)
        examples.to_json(example_dir / "train.json", orient="records")
        # 在任务配置里声明示例路径（TaskRunner 据此加载）
        task_cfg = json.load(open(task_json_dest))
        task_cfg["Example_Path"] = "train.json"
        json.dump(task_cfg, open(task_json_dest, "w"), indent=4)
        print(f"[llm_fusion] few-shot 模式：{num_examples} 个示例（示例库 {len(examples)} 条）")
    else:
        num_examples = 0
        example_dir.mkdir(parents=True, exist_ok=True)

    from llm_extractinator import extractinate

    extractinate(
        task_id=int(task_id),
        model_name=model_name,
        num_examples=num_examples,
        temperature=0.0,
        max_context_len="split",
        num_predict=num_predict,
        translate=False,
        data_dir=workdir / "data",
        output_dir=workdir / "output",
        task_dir=workdir / "tasks",
        example_dir=example_dir,
        n_runs=1,
        verbose=False,
        run_name="run",
        reasoning_model=False,
        seed=42,
        overwrite=True,
    )

    # extractinate 内部吞异常，这里以产物文件为准做硬校验
    pred_files = sorted(workdir.glob("output/run/*/nlp-predictions-dataset.json"))
    if not pred_files:
        raise RuntimeError("LLM 提取未产生任何输出文件")
    with open(pred_files[-1]) as f:
        rows = json.load(f)
    if len(rows) != len(df):
        print(f"[llm_fusion] 警告: LLM 输出 {len(rows)} 条 ≠ 输入 {len(df)} 条")
    return {str(r["uid"]): parse_row(problem_type, r) for r in rows}


# ---------------- 融合函数 ----------------

def blend_binary(p_enc, q_llm, lam: float) -> np.ndarray:
    """二分类：score = λ·编码器概率 + (1-λ)·LLM 投票(0/1)；LLM 无效行保留编码器。"""
    out = np.asarray(p_enc, dtype=float).copy()
    mask = np.array([q is not None for q in q_llm], dtype=bool)
    q_vals = np.array([q if q is not None else 0.0 for q in q_llm], dtype=float)
    out[mask] = lam * out[mask] + (1.0 - lam) * q_vals[mask]
    return out


def tune_lambda_binary(y_true, p_enc, q_llm) -> float:
    from sklearn.metrics import roc_auc_score

    best_lam, best_score = 1.0, -np.inf
    enc_score = None
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器（λ 大）
        try:
            score = roc_auc_score(y_true, blend_binary(p_enc, q_llm, lam))
        except ValueError:
            continue
        if lam == 1.0:
            enc_score = score
        if score > best_score + 1e-9:
            best_score, best_lam = score, lam
    # 收缩保护：LLM 混合必须明显优于纯编码器，否则回退 λ=1
    if best_lam != 1.0 and enc_score is not None and best_score <= enc_score + 0.002:
        return 1.0
    return best_lam


def blend_multiclass(enc_dist: np.ndarray, classes: list, llm_labels, lam: float) -> list:
    """多分类：分布混合 λ·softmax + (1-λ)·LLM one-hot，取 argmax。"""
    dist = np.asarray(enc_dist, dtype=float).copy()
    class_index = {c: i for i, c in enumerate(classes)}
    for i, label in enumerate(llm_labels):
        if label is not None and label in class_index:
            onehot = np.zeros(len(classes))
            onehot[class_index[label]] = 1.0
            dist[i] = lam * dist[i] + (1.0 - lam) * onehot
    return [classes[i] for i in dist.argmax(axis=1)]


def tune_lambda_multiclass(y_true, enc_dist, classes, llm_labels, task_id: str) -> float:
    from sklearn.metrics import cohen_kappa_score

    weights = "linear" if task_id in LINEAR_KAPPA_TASK_IDS else None
    best_lam, best_score = 1.0, -np.inf
    enc_score = None
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器
        preds = blend_multiclass(enc_dist, classes, llm_labels, lam)
        score = cohen_kappa_score(y_true, preds, weights=weights)
        if lam == 1.0:
            enc_score = score
        if score > best_score + 1e-9:
            best_score, best_lam = score, lam
    # 收缩保护
    if best_lam != 1.0 and enc_score is not None and best_score <= enc_score + 0.002:
        return 1.0
    return best_lam


def blend_mlregression(enc_vals, llm_vals, lam: float) -> list:
    """多标签回归：逐元素 λ·编码器 + (1-λ)·LLM；LLM 无效行保留编码器。"""
    out = []
    for enc, llm in zip(enc_vals, llm_vals):
        if llm is None:
            out.append(list(enc))
        else:
            out.append([lam * e + (1.0 - lam) * l for e, l in zip(enc, llm)])
    return out


def tune_lambda_mlreg(y_true, enc_vals, llm_vals, epsilon: float) -> float:
    from dragon_eval.evaluation import score_rsmape

    y_true = np.asarray([list(v) for v in y_true], dtype=float)
    best_lam, best_score = 1.0, -np.inf
    enc_score = None
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器
        preds = blend_mlregression(enc_vals, llm_vals, lam)
        score = score_rsmape(
            y_true=y_true, y_pred=np.asarray(preds, dtype=float),
            epsilon=epsilon, ignore_missing_targets=True,
        )
        if lam == 1.0:
            enc_score = score
        if score > best_score + 1e-9:
            best_score, best_lam = score, lam
    # 收缩保护
    if best_lam != 1.0 and enc_score is not None and best_score <= enc_score + 0.002:
        return 1.0
    return best_lam


def blend_mlmc(enc_labels, llm_labels, lam: float) -> list:
    """多标签多分类：lambda 作为来源开关（>=0.5 用编码器标签，否则用 LLM 标签）。

    编码器不输出概率时这是无分布信息的退化融合；LLM 无效位置回退编码器。
    """
    out = []
    for enc, llm in zip(enc_labels, llm_labels):
        merged = []
        for e, l in zip(enc, llm if llm is not None else [None] * len(enc)):
            if l is None:
                merged.append(e)
            else:
                merged.append(e if lam >= 0.5 else l)
        out.append(merged)
    return out


def tune_lambda_mlmc(y_true, enc_labels, llm_labels, task_id: str) -> float:
    """按官方 kappa（各标签列取均值）在验证集选择来源开关。"""
    from sklearn.metrics import cohen_kappa_score

    weights = "linear" if task_id in LINEAR_KAPPA_TASK_IDS else None
    n_elems = len(enc_labels.iloc[0] if hasattr(enc_labels, "iloc") else enc_labels[0])
    best_lam, best_score = 1.0, -np.inf
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器
        preds = blend_mlmc(enc_labels, llm_labels, lam)
        scores = []
        for j in range(n_elems):
            y_t = [str(v[j]) for v in y_true]
            y_p = [str(v[j]) for v in preds]
            scores.append(cohen_kappa_score(y_t, y_p, weights=weights))
        score = float(np.mean(scores))
        if score > best_score + 1e-9:
            best_score, best_lam = score, lam
    return best_lam


def tune_lambda_mlbinary(y_true, enc_vals, llm_vals) -> float:
    """多标签二分类（T16）：各 lesion 列 AUC 取均值。"""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray([list(v) for v in y_true], dtype=float)
    n_elems = y_true.shape[1]
    best_lam, best_score = 1.0, -np.inf
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器
        preds = np.asarray(blend_mlregression(enc_vals, llm_vals, lam), dtype=float)
        scores = []
        for j in range(n_elems):
            try:
                scores.append(roc_auc_score(y_true[:, j], preds[:, j]))
            except ValueError:
                scores.append(0.5)
        score = float(np.mean(scores))
        if score > best_score + 1e-9:
            best_score, best_lam = score, lam
    return best_lam
