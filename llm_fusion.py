"""LLM 融合模块：弱任务的零样本 LLM 提取 + 与编码器预测的融合。

设计要点：
- 仅对 WEAK_TASK_IDS 内的任务启用；LLM 任何环节失败由调用方回退编码器结果
- 融合权重 λ（编码器权重）在验证集上网格搜索，按任务官方指标选型
- 逐行校验 LLM 输出（status/取值合法），无效行保留编码器预测
"""
import json
import logging
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

WEAK_TASK_IDS = {"003", "006", "008", "012", "013", "014", "024", "103"}
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
    if task_id in ("014", "103"):  # NLI 句对任务：两个句子分开标注
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
        if problem_type == PT.MULTI_LABEL_REGRESSION:
            keys = ["lesion_1", "lesion_2", "lesion_3", "lesion_4", "lesion_5"]
            return [float(row[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _port_free(host="127.0.0.1", port=11434) -> bool:
    import socket

    s = socket.socket()
    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        return False


def _ollama_selfcheck(*, workdir: Path, model_name: str) -> None:
    """环境自检：ollama 可用性 + 一次真实生成计时 + 服务端日志（GPU/CPU 判别）。

    自检结束后关闭自起的服务并等待端口释放，让 extractinate 干净地启动自己的服务。
    """
    import shutil as _shutil
    import subprocess as _sp
    import time as _time

    print(f"[llm_fusion] SELFCHECK ollama binary: {_shutil.which('ollama')}")
    srv_log_path = workdir / "ollama_selfcheck.log"
    srv_log = open(srv_log_path, "w")
    srv = _sp.Popen(["ollama", "serve"], stdout=srv_log, stderr=_sp.STDOUT)
    _time.sleep(10)
    try:
        t0 = _time.time()
        r = _sp.run(
            ["ollama", "run", model_name, "Reply with exactly: OK"],
            capture_output=True, text=True, timeout=420,
            env={**os.environ, "OLLAMA_NO_UPDATE": "1"},
        )
        dt = _time.time() - t0
        print(f"[llm_fusion] SELFCHECK generate took {dt:.1f}s  (<30s≈GPU, >60s≈CPU)")
        print(f"[llm_fusion] SELFCHECK output: {r.stdout[:80]!r}")
        if r.returncode != 0:
            print(f"[llm_fusion] SELFCHECK stderr: {r.stderr[-500:]!r}")
    except Exception as e:
        print(f"[llm_fusion] SELFCHECK FAILED: {e}")
    try:
        srv_log.close()
        tail = open(srv_log_path, errors="replace").read()[-1500:]
        print("----- ollama server log tail -----")
        print(tail)
        print("----- end -----")
    except Exception:
        pass
    # 关闭自检服务，等待端口释放（最多 60s）
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    for _ in range(60):
        if _port_free():
            break
        _time.sleep(1)
    print(f"[llm_fusion] SELFCHECK 服务已关闭，端口释放: {_port_free()}")


def _dump_server_logs(workdir: Path, *, srv_log_path: Path) -> None:
    """打印提取全过程的日志尾部：自检服务、子进程 stdout/stderr、extractinate 日志、其管理的 ollama 服务。"""
    for name, path in [
        ("selfcheck", srv_log_path),
        ("child-stdout", Path(workdir) / "child_stdout.log"),
        ("child-stderr", Path(workdir) / "child_stderr.log"),
        ("task_runner", Path(workdir) / "output" / "logs" / "task_runner.log"),
        ("extraction", Path(workdir) / "output" / "logs" / "ollama_server.log"),
    ]:
        try:
            tail = open(path, errors="replace").read()[-2000:]
            print(f"----- {name} log tail ({path}) -----")
            print(tail)
            print("----- end -----")
        except Exception:
            pass


def _extractinate_child(kwargs: dict, result_queue, log_path: str) -> None:
    """extractinate 的子进程入口（spawn 需要模块级函数）。

    - stdout/stderr 重定向到文件：避免 GC 日志管道阻塞导致整个子进程卡死，
      同时完整留证，结束后由父进程打印尾部
    - faulthandler 每 3 分钟把所有线程栈打到 stderr：若再挂，日志里直接看到挂在哪一行
    - 跳过运行时 ollama pull：模型已烤入镜像，pull 是纯网络依赖（GC 出口受限时会挂死）
    - count_tokens 替换为纯 Python 估算：tiktoken>=0.13 换新缓存键，镜像里旧键缓存不命中
      会联网下载，GC 出口代理对非白名单域名挂起不放 → 提取阶段死锁（v1.2.8/1.2.9/1.3.1
      三次 40 分钟超时的共同根因）。token 计数只用于长短切分与 num_ctx，精度要求低。
    - 全局 socket 超时兜底：任何其他隐藏的网络依赖最多卡 30 秒就报错，不会永久挂死
    """
    import faulthandler
    import socket
    import sys
    from pathlib import Path

    socket.setdefaulttimeout(30)

    # 若容器/系统带代理环境变量，确保 ollama 的 localhost 请求不被代理拦截
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

    sys.stdout = open(log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stderr = open(
        Path(log_path).with_name("child_stderr.log"),
        "w",
        encoding="utf-8",
        errors="replace",
        buffering=1,
    )
    faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)
    print("[llm_fusion.child] 子进程启动", flush=True)
    try:
        import logging as _logging

        from llm_extractinator.data_loader import DataLoader
        from llm_extractinator.ollama_server import OllamaServerManager

        def _fast_count_tokens(self, text, model_name="cl100k_base"):
            try:
                return int(len(str(text).split()) * 1.3)
            except Exception:
                return 0

        DataLoader.count_tokens = _fast_count_tokens

        def _skip_pull(self, model_name):
            _logging.getLogger(__name__).info(
                f"[llm_fusion] 跳过运行时 ollama pull（模型已内置镜像）: {model_name}"
            )

        OllamaServerManager.pull_model = _skip_pull

        _patch_predictor_concurrency()  # spawn 子进程是全新解释器，父进程的补丁不会带过来

        from llm_extractinator import extractinate

        extractinate(**kwargs)
        print("[llm_fusion.child] extractinate 返回", flush=True)
        result_queue.put(("ok", None))
    except Exception as e:  # noqa: BLE001
        result_queue.put(("err", repr(e)))


def _patch_predictor_concurrency() -> None:
    """替换 extractinate 的 Predictor.predict：并发 + 健壮解析（本函数是关键修复）。

    原版 chain = prompt | model | OutputFixingParser，有两个实测问题：
    1. 模型会照抄提示词里的 JSON schema 外壳（输出 {"properties": {...}, "required": [...]}），
       pydantic 解析必失败；OutputFixingParser 再用 LLM 重试 3 次，同样失败，
       且每次失败多花 3 次 LLM 调用（40 条样本就是 160 次）。
    2. 无并发参数时顺序推理跑不完。
    新版：prompt | model 后自行解析——标准解析 → 剥 "properties" 外壳 → 失败记 failure。
    """
    from llm_extractinator.callbacks import BatchCallBack
    from llm_extractinator.predictor import Predictor
    from llm_extractinator.validator import handle_prediction_failure

    max_concurrency = int(os.environ.get("LLM_MAX_CONCURRENCY", "4"))

    def predict(self, test_data):
        import json as _json

        logger_src = logging.getLogger(__name__)
        logger_src.info(
            "Starting prediction on test data with %d samples (max_concurrency=%d, robust-parser).",
            len(test_data), max_concurrency,
        )
        chain = self.prompt | self.model
        test_data_processed = [
            {"input": row[self.input_field]} for _, row in test_data.iterrows()
        ]
        callbacks = BatchCallBack(len(test_data_processed))
        raw_results = chain.batch(
            test_data_processed,
            config={"callbacks": [callbacks], "max_concurrency": max_concurrency},
            return_exceptions=True,
        )
        callbacks.progress_bar.close()

        def robust_parse(text: str):
            # 1) 标准 pydantic JSON 解析
            try:
                return self.base_parser.parse(text)
            except Exception:
                pass
            # 2) 模型照抄 schema 外壳：{"properties": {"reasoning": ..., "label": ...}, "required": [...]}
            try:
                obj = _json.loads(text)
                inner = obj.get("properties") if isinstance(obj, dict) else None
                if isinstance(inner, dict):
                    return self.parser_model.model_validate(inner)
            except Exception:
                pass
            # 3) 逐字段正则提取（外壳套得更深时，原始值通常仍原样出现在文本里）
            try:
                extracted = {}
                for fname, fdef in self.parser_model.model_fields.items():
                    ann = fdef.annotation
                    is_num = ann in (int, float) or getattr(ann, "__args__", None) and any(
                        a in (int, float) for a in ann.__args__
                    )
                    if is_num:
                        m = re.search(rf'"{fname}"\s*:\s*(-?[\d.]+)', text)
                        if m:
                            extracted[fname] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
                    else:
                        m = re.search(rf'"{fname}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
                        if m:
                            extracted[fname] = m.group(1)
                if extracted:
                    return self.parser_model.model_validate(extracted)
            except Exception:
                pass
            return None

        final_results = []
        failure_counter = 0
        for input_data, result in zip(test_data_processed, raw_results):
            if isinstance(result, Exception):
                failure_counter += 1
                if failure_counter <= 5:
                    logger_src.warning(
                        "LLM call failed (sample %.60r): %r",
                        input_data.get("input", ""), result,
                    )
                final_results.append(
                    handle_prediction_failure(result, input_data, self.parser_model)
                )
                continue
            text = result.content if hasattr(result, "content") else str(result)
            obj = robust_parse(text)
            if obj is None:
                failure_counter += 1
                if failure_counter <= 5:
                    logger_src.warning(
                        "LLM output unparseable (sample %.60r): %.1200r",
                        input_data.get("input", ""), text,
                    )
                final_results.append(
                    handle_prediction_failure(
                        ValueError("unparseable LLM output"), input_data, self.parser_model
                    )
                )
            else:
                result_dict = obj.model_dump() if hasattr(obj, "model_dump") else obj
                result_dict["status"] = "success"
                final_results.append(result_dict)

        logger_src.info("Prediction completed. Failed predictions: %d", failure_counter)
        return final_results

    if getattr(Predictor.predict, "_patched_concurrency", False):
        return
    predict._patched_concurrency = True
    Predictor.predict = predict


def run_llm_extraction(
    *,
    task_id: str,
    problem_type,
    df: pd.DataFrame,
    input_name: str,
    workdir: Path,
    model_name: str,
    num_predict: int = 224,
    max_rows: int = None,
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

    # 本地调试：LLM_SUBSET=30 时仅对前 30 条做 LLM 提取（生产环境勿设）
    subset = os.environ.get("LLM_SUBSET")
    if subset:
        n = int(subset)
        print(f"[llm_fusion] 本地测试模式：仅对前 {n}/{len(df)} 条做 LLM 提取")
        df = df.head(n).copy()
    if max_rows and len(df) > max_rows:
        print(f"[llm_fusion] 调参模式：验证集取前 {max_rows}/{len(df)} 条")
        df = df.head(max_rows).copy()

    data = pd.DataFrame({"uid": df["uid"].astype(str).values})
    data["text"] = prepare_llm_text(df, input_name, task_id).values
    data.to_json(workdir / "data" / "test.json", orient="records")

    task_jsons = list(LLM_TASK_DIR.glob(f"Task{task_id}_*.json"))
    if not task_jsons:
        raise FileNotFoundError(f"llm_tasks/ 下找不到 Task{task_id} 的提示词 JSON")
    shutil.copy(task_jsons[0], workdir / "tasks" / task_jsons[0].name)

    _ollama_selfcheck(workdir=workdir, model_name=model_name)
    _patch_predictor_concurrency()
    print(f"[llm_fusion] 提取开始：task={task_id} model={model_name} num_predict={num_predict} 条数={len(df)}")

    kwargs = dict(
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

    # 限时运行：超时直接终止并清理，让调用方回退编码器（避免容器被平台整体杀掉、零产出）
    import multiprocessing as mp
    import subprocess as sp
    import time as _time

    budget_min = float(os.environ.get("LLM_TIME_BUDGET_MIN", "40"))
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    child_log = workdir / "child_stdout.log"
    proc = ctx.Process(
        target=_extractinate_child, args=(kwargs, result_queue, str(child_log))
    )
    proc.start()
    deadline = _time.time() + budget_min * 60
    next_beat = _time.time() + 300
    while proc.is_alive() and _time.time() < deadline:
        proc.join(timeout=30)
        if _time.time() >= next_beat and proc.is_alive():
            print(
                f"[llm_fusion] 提取进行中... 已等待 {(_time.time() - (deadline - budget_min * 60)) / 60:.0f} 分钟",
                flush=True,
            )
            next_beat = _time.time() + 300
    status, err = ("err", "child exited without result"), None
    if proc.is_alive():
        print(f"[llm_fusion] 提取超时（>{budget_min:.0f} 分钟），终止 LLM 子进程并回退编码器")
        proc.terminate()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
        sp.run(["pkill", "-f", "ollama"], capture_output=True)
        _dump_server_logs(workdir, srv_log_path=workdir / "ollama_selfcheck.log")
        raise RuntimeError(f"LLM extraction timed out after {budget_min:.0f} min")
    try:
        status, err = result_queue.get_nowait()
    except Exception:
        pass
    if status != "ok":
        _dump_server_logs(workdir, srv_log_path=workdir / "ollama_selfcheck.log")
        raise RuntimeError(f"extractinate failed: {err}")

    _dump_server_logs(workdir, srv_log_path=workdir / "ollama_selfcheck.log")

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
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器（λ 大）
        try:
            score = roc_auc_score(y_true, blend_binary(p_enc, q_llm, lam))
        except ValueError:
            continue
        if score > best_score + 1e-9:
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
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器
        preds = blend_multiclass(enc_dist, classes, llm_labels, lam)
        score = cohen_kappa_score(y_true, preds, weights=weights)
        if score > best_score + 1e-9:
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
    for lam in sorted(LAMBDA_GRID, reverse=True):  # 并列时偏好编码器
        preds = blend_mlregression(enc_vals, llm_vals, lam)
        score = score_rsmape(
            y_true=y_true, y_pred=np.asarray(preds, dtype=float),
            epsilon=epsilon, ignore_missing_targets=True,
        )
        if score > best_score + 1e-9:
            best_score, best_lam = score, lam
    return best_lam
