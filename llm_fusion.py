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

WEAK_TASK_IDS = {"006", "008", "012", "013", "014", "024"}
# 本地调试用：LLM_EXTRA_TASKS="102,106" 可把合成任务临时纳入弱任务（须 llm_tasks/ 有对应提示词）
WEAK_TASK_IDS |= set(
    x for x in os.environ.get("LLM_EXTRA_TASKS", "").split(",") if x.strip()
)
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


def parse_row(task_id: str, row: dict):
    """解析 extractinate 的一行输出为预测值；无效（失败/非法）返回 None。"""
    if row.get("status") != "success":
        return None
    try:
        if task_id in BINARY_TASK_IDS:
            label = row.get("label")
            if label in (True, "True", "true", "1", 1):
                return 1.0
            if label in (False, "False", "false", "0", 0):
                return 0.0
            return None
        if task_id in MULTICLASS_TASK_IDS:
            label = row.get("label")
            return str(label) if label not in (None, "") else None
        if task_id in ML_REGRESSION_TASK_IDS:
            keys = ["lesion_1", "lesion_2", "lesion_3", "lesion_4", "lesion_5"]
            return [float(row[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return None
    return None


def run_llm_extraction(
    *,
    task_id: str,
    df: pd.DataFrame,
    input_name: str,
    workdir: Path,
    model_name: str,
    num_predict: int = 512,
) -> dict:
    """对给定数据运行零样本 LLM 提取，返回 {uid: 预测值或 None}。

    复用 llm_extractinator 的提示词、结构化输出与失败重试机制。
    """
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / "data").mkdir(parents=True)
    (workdir / "tasks").mkdir(parents=True)
    (workdir / "output").mkdir(parents=True)

    data = pd.DataFrame({"uid": df["uid"].astype(str).values})
    data["text"] = prepare_llm_text(df, input_name, task_id).values
    data.to_json(workdir / "data" / "test.json", orient="records")

    task_jsons = list(LLM_TASK_DIR.glob(f"Task{task_id}_*.json"))
    if not task_jsons:
        raise FileNotFoundError(f"llm_tasks/ 下找不到 Task{task_id} 的提示词 JSON")
    shutil.copy(task_jsons[0], workdir / "tasks" / task_jsons[0].name)

    from llm_extractinator import extractinate

    extractinate(
        task_id=int(task_id),
        model_name=model_name,
        num_examples=0,
        temperature=0.0,
        max_context_len="split",
        num_predict=num_predict,
        translate=False,
        data_dir=workdir / "data",
        output_dir=workdir / "output",
        task_dir=workdir / "tasks",
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
    return {str(r["uid"]): parse_row(task_id, r) for r in rows}


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
    for lam in LAMBDA_GRID:
        try:
            score = roc_auc_score(y_true, blend_binary(p_enc, q_llm, lam))
        except ValueError:
            continue
        if score > best_score:
            best_score, best_lam = score, lam
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
    for lam in LAMBDA_GRID:
        preds = blend_multiclass(enc_dist, classes, llm_labels, lam)
        score = cohen_kappa_score(y_true, preds, weights=weights)
        if score > best_score:
            best_score, best_lam = score, lam
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
    for lam in LAMBDA_GRID:
        preds = blend_mlregression(enc_vals, llm_vals, lam)
        score = score_rsmape(
            y_true=y_true, y_pred=np.asarray(preds, dtype=float),
            epsilon=epsilon, ignore_missing_targets=True,
        )
        if score > best_score:
            best_score, best_lam = score, lam
    return best_lam
