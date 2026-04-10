#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
归因稳定性（Robustness / Stability）验证实验

对事故文本做「语义等价」非关键词同义替换（人机管环领域词保持不变），
分别计算原始与扰动文本的词级 SHAP（与 `interpretability_verification_experiment` +
`shap_fidelity_perturbation_test.compute_shap_spans_via_interpretability_analyzer` 一致），
比较相同关键词上 SHAP 的 Pearson 相关与 Top-K 特征的 Jaccard 重合度，并绘制散点图。

用法示例：
  python shap_robustness_stability.py --task accident_type
  python shap_robustness_stability.py --top-richness 50 --shap-window-sizes 1 --output-dir ./out
  python shap_robustness_stability.py --demo-shap  # 不加载模型，伪 SHAP 仅验证管线

进度条：默认开启（外层按样本、内层按句词级 SHAP）；加 --no-progress 可关闭。需安装 tqdm。
"""

from __future__ import annotations

import argparse
import heapq
import logging
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import jieba
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "5model_compare":
    MODEL_DIR = os.path.dirname(SCRIPT_DIR)
else:
    MODEL_DIR = SCRIPT_DIR
sys.path.append(SCRIPT_DIR)

# 复用项目内 SHAP 管线
from shap_fidelity_perturbation_test import (  # noqa: E402
    DEFAULT_EXCEL_PATH,
    DEFAULT_QWEN_BASE_PATH,
    DEFAULT_LORA_ADAPTER_PATH,
    DEFAULT_SHAP_CACHE_PATH,
    EXCEL_FALLBACK_PATHS,
    build_shap_spans_jieba,
    compute_shap_spans_via_interpretability_analyzer,
    get_deterministic_analyzer_class,
    parse_shap_window_sizes,
    resolve_adapter_path,
    resolve_base_model_path,
    resolve_excel_path,
    samples_top_richness_from_excel,
)


# 与 data/print_richness_rank2_and_rank5.RISK_KEYWORDS 一致，并扩展人机管环常见实体/维度词
DEFAULT_DOMAIN_KEYWORDS: Set[str] = {
    "操作",
    "设备",
    "环境",
    "管理",
    "故障",
    "错误",
    "失效",
    "风险",
    "事故",
    "人员",
    "无人机",
    "系统",
    "软件",
    "硬件",
    "传感器",
    "电池",
    "导航",
    "操作员",
    "飞手",
    "飞行员",
    "维护",
    "培训",
    "监督",
    "程序",
    "制度",
    "人为",
    "复合型",
    "干扰",
}


# 非关键词同义替换（单向：仅当整词 jieba 分词命中时替换；领域词见 DEFAULT_DOMAIN_KEYWORDS 保护）
DEFAULT_SYNONYM_MAP: Dict[str, str] = {
    "执行": "进行",
    "实施": "实行",
    "坠落": "掉落",
    "跌落": "落下",
    "导致": "造成",
    "由于": "因为",
    "因此": "所以",
    "随后": "之后",
    "立即": "马上",
    "最终": "最后",
    "严重": "重大",
    "迅速": "快速",
    "检查": "查验",
    "发现": "察觉",
    "继续": "持续",
    "停止": "中止",
    "启动": "开启",
    "关闭": "关上",
    "升高": "上升",
    "降低": "下降",
    "增大": "增加",
    "减小": "减少",
    "可能": "也许",
    "应当": "应该",
    "必须": "务必",
    "如果": "假如",
    "但是": "然而",
    "而且": "并且",
    "以及": "还有",
    "通过": "经由",
    "根据": "依据",
    "完成": "结束",
    "尚未": "还未",
    "已经": "已然",
    "正在": "正",
    "发生": "出现",
    "出现": "产生",
    "影响": "作用",
    "控制": "操控",
    "调整": "调节",
    "确认": "核实",
    "记录": "记载",
    "报告": "上报",
    "分析": "剖析",
    "评估": "评价",
    "处理": "处置",
    "排除": "消除",
    "避免": "防止",
    "保证": "确保",
    "满足": "符合",
    "达到": "实现",
    "超过": "超出",
    "低于": "少于",
    "高于": "多于",
    "附近": "周围",
    "内部": "里面",
    "外部": "外面",
}


def default_protected_vocabulary() -> Set[str]:
    """人机管环及风险领域词：不参与同义替换。"""
    return set(DEFAULT_DOMAIN_KEYWORDS)


def semantic_equivalent_perturbation(
    text: str,
    *,
    protected: Optional[Set[str]] = None,
    synonym_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    保持领域关键词不变，对 jieba 分词单元中非保护词按同义词表替换。
    """
    prot = protected if protected is not None else default_protected_vocabulary()
    smap = synonym_map if synonym_map is not None else DEFAULT_SYNONYM_MAP

    out: List[str] = []
    pos = 0
    for word, start, end in jieba.tokenize(text):
        out.append(text[pos:start])
        if word in prot:
            out.append(word)
        else:
            out.append(smap.get(word, word))
        pos = end
    out.append(text[pos:])
    return "".join(out)


def spans_to_word_shap_dict(text: str, spans: Sequence[Tuple[int, int, float]]) -> Dict[str, float]:
    """由 (start, end, shap) 列表得到 词 -> SHAP；同一词多次出现取首次（值相同）。"""
    d: Dict[str, float] = {}
    for a, b, s in spans:
        w = text[int(a) : int(b)]
        if not w.strip():
            continue
        if w not in d:
            d[w] = float(s)
    return d


def _pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 2:
        return float("nan")
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return float("nan")
    try:
        from scipy import stats

        r, _ = stats.pearsonr(xa, ya)
        return float(r)
    except ImportError:
        c = np.corrcoef(xa, ya)[0, 1]
        return float(c)


def topk_feature_jaccard(
    dict_a: Dict[str, float],
    dict_b: Dict[str, float],
    k: int,
) -> float:
    """按 |SHAP| 取 Top-K 词集合，计算 Jaccard。"""
    if k <= 0 or not dict_a or not dict_b:
        return float("nan")

    def top_set(d: Dict[str, float]) -> Set[str]:
        # [优化] 使用 heapq.nlargest 在 O(n log k) 内取 Top-K，避免对全量词项排序
        # [优化] key 含词面 tie-breaker，消除哈希随机化与平台差异带来的非确定性
        key_fn = lambda item: (abs(item[1]), item[0])
        top_items = heapq.nlargest(k, d.items(), key=key_fn)
        return {w for w, _ in top_items}

    sa, sb = top_set(dict_a), top_set(dict_b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return float("nan")
    return inter / union


def shared_domain_keyword_terms(
    dict_o: Dict[str, float],
    dict_p: Dict[str, float],
    domain_keywords: Iterable[str],
) -> List[str]:
    """
    在原始与扰动两条归因的「词级 SHAP 字典」中均作为 key 出现的领域词。
    用字典 key 判定，避免子串匹配（如「机」命中「无人机」）导致的 Token 错位。
    顺序固定为 domain_keywords 遍历顺序。
    """
    out: List[str] = []
    for w in domain_keywords:
        if w and (w in dict_o) and (w in dict_p):
            out.append(w)
    return out


def build_synthetic_samples(n: int, task: str) -> List[Dict[str, Any]]:
    """
    无 Excel 时的烟测数据：若干条含人机管环要素的叙述，循环拼满 n 条。
    标签为占位，仅用于 `--demo-shap` 跑通管线。
    """
    label_map = {
        "accident_type": "人为因素主导型",
        "damage_level": "1 无损害",
        "casualty_count": "0 死亡0人，受伤0人",
    }
    base_texts = [
        "人机管环：操作员在强风环境下执行起飞前检查不够充分，对电池状态评估不足。",
        "事故链描述：随后无人机在爬升过程中发生失控，最终坠落至地面，造成设备损坏。",
        "管理方面未严格执行维护制度；环境侧阵风超过当日作业标准。",
        "人机管环：飞手培训充分但当日疲劳作业，传感器读数异常未能立即察觉。",
        "事故链描述：导航系统出现短暂失效，操作员通过手动干预恢复，未造成伤亡。",
        "设备电池循环次数已高；监督人员未现场核实风险。",
        "人机管环：维护人员完成固件升级后未执行完整自检程序。",
        "事故链描述：软件错误导致电机响应延迟，无人机在降落阶段与障碍物碰撞。",
        "管理缺失体现在记录不完整；环境为夜间低能见度。",
    ]
    out: List[Dict[str, Any]] = []
    for i in range(n):
        t = base_texts[i % len(base_texts)]
        if i >= len(base_texts):
            t = t + f" (样本序号{i+1})"
        out.append(
            {
                "text": t,
                "target_label": label_map.get(task, label_map["accident_type"]),
                "target_type": task,
            }
        )
    return out


def _wrap_samples_progress(
    samples: List[Dict[str, Any]],
    *,
    enabled: bool,
    desc: str = "归因稳定性",
    unit: str = "条",
) -> Iterable[Dict[str, Any]]:
    """将样本列表包装为 tqdm 可迭代对象；未安装 tqdm 时退回原列表并打日志。"""
    if not enabled or not samples:
        return samples
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("未安装 tqdm，无法显示进度条。可执行: pip install tqdm")
        return samples
    return tqdm(
        samples,
        desc=desc,
        unit=unit,
        total=len(samples),
        dynamic_ncols=True,
    )


def robustness_check(
    analyzer: Any,
    samples: List[Dict[str, Any]],
    *,
    window_sizes: Sequence[int] = (1,),
    top_k_jaccard: int = 10,
    domain_keywords: Optional[Sequence[str]] = None,
    protected_for_perturbation: Optional[Set[str]] = None,
    synonym_map: Optional[Dict[str, str]] = None,
    shap_cache_path: Optional[str] = None,
    use_shap_cache: bool = True,
    show_progress: bool = True,
    demo_shap: bool = False,
) -> Dict[str, Any]:
    """
    对样本列表逐条：生成扰动文本 -> 计算原始/扰动 SHAP -> Pearson（相同领域关键词）与 Top-K Jaccard。

    每条 sample 需含: text, target_label, target_type（与 fidelity 脚本一致）。
    """
    dom = list(domain_keywords) if domain_keywords is not None else list(DEFAULT_DOMAIN_KEYWORDS)
    prot = protected_for_perturbation if protected_for_perturbation is not None else default_protected_vocabulary()
    smap = synonym_map if synonym_map is not None else DEFAULT_SYNONYM_MAP

    pearsons: List[float] = []
    jaccards: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    per_sample: List[Dict[str, Any]] = []

    # 外层：按样本进度；内层：compute_shap_spans 内按句 tqdm（与 show_progress 联动）
    loop: Iterable[Dict[str, Any]] = _wrap_samples_progress(samples, enabled=show_progress)

    for s in loop:
        text_o = str(s["text"])
        label = str(s["target_label"])
        ttype = str(s["target_type"])
        text_p = semantic_equivalent_perturbation(text_o, protected=prot, synonym_map=smap)

        if hasattr(loop, "set_postfix_str"):
            short = (label[:30] + "…") if len(label) > 30 else label
            loop.set_postfix_str(f"{ttype}|{short}", refresh=False)

        if demo_shap:
            spans_o = build_shap_spans_jieba(text_o, True)
            spans_p = build_shap_spans_jieba(text_p, True)
        else:
            spans_o = compute_shap_spans_via_interpretability_analyzer(
                analyzer,
                text_o,
                label,
                ttype,
                window_sizes=window_sizes,
                show_progress=show_progress,
                cache_path=shap_cache_path,
                use_cache=use_shap_cache,
            )
            spans_p = compute_shap_spans_via_interpretability_analyzer(
                analyzer,
                text_p,
                label,
                ttype,
                window_sizes=window_sizes,
                show_progress=show_progress,
                cache_path=shap_cache_path,
                use_cache=use_shap_cache,
            )

        dict_o = spans_to_word_shap_dict(text_o, spans_o)
        dict_p = spans_to_word_shap_dict(text_p, spans_p)

        terms = shared_domain_keyword_terms(dict_o, dict_p, dom)
        vo = [dict_o[w] for w in terms]
        vp = [dict_p[w] for w in terms]
        for a, b in zip(vo, vp):
            xs.append(a)
            ys.append(b)

        r = _pearson_r(vo, vp)
        j = topk_feature_jaccard(dict_o, dict_p, top_k_jaccard)
        pearsons.append(r)
        jaccards.append(j)
        per_sample.append(
            {
                "n_shared_keywords": len(terms),
                "shared_keywords": terms,
                "pearson_r": r,
                "jaccard_topk": j,
                "perturbed_preview": text_p[:200] + ("…" if len(text_p) > 200 else ""),
            }
        )

    valid_r = [x for x in pearsons if not np.isnan(x)]
    valid_j = [x for x in jaccards if not np.isnan(x)]
    mean_r = float(np.mean(valid_r)) if valid_r else float("nan")
    mean_j = float(np.mean(valid_j)) if valid_j else float("nan")

    return {
        "mean_pearson_r": mean_r,
        "mean_jaccard_topk": mean_j,
        "pearson_per_sample": pearsons,
        "jaccard_per_sample": jaccards,
        "scatter_x_original_shap": xs,
        "scatter_y_perturbed_shap": ys,
        "per_sample": per_sample,
        "n_samples": len(samples),
        "top_k_jaccard": top_k_jaccard,
    }


def save_dataframe_csv_with_fallback(df: pd.DataFrame, path: str) -> str:
    """
    将 DataFrame 写入 CSV；若 path 被占用则改为「原名_时间戳.csv」再写。
    返回实际写入的文件路径。
    """
    kwargs: Dict[str, Any] = {"index": False, "encoding": "utf-8-sig"}
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = f"{root}_{ts}{ext}"
        df.to_csv(alt_path, **kwargs)
        return alt_path


def save_figure_with_fallback(out_path: str, dpi: int = 150) -> str:
    """
    保存当前 matplotlib 图像；若 out_path 被占用则改为「原名_时间戳」+ 原扩展名再保存。
    与 save_dataframe_csv_with_fallback 的 PermissionError 回退策略一致（DRY 语义）。
    """
    import matplotlib.pyplot as plt

    try:
        plt.savefig(out_path, dpi=dpi)
        return out_path
    except PermissionError:
        root, ext = os.path.splitext(out_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = f"{root}_{ts}{ext}"
        plt.savefig(alt_path, dpi=dpi)
        return alt_path


def _plot_scatter(result: Dict[str, Any], out_path: str, title: str) -> str:
    import matplotlib.pyplot as plt

    # [优化] 使用 rc_context 隔离全局状态污染，避免修改 plt.rcParams 默认值
    custom_rc = {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }

    with plt.rc_context(custom_rc):
        x = result["scatter_x_original_shap"]
        y = result["scatter_y_perturbed_shap"]
        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, alpha=0.45, s=22, edgecolors="none")
        lims: List[float] = []
        for v in x + y:
            if np.isfinite(v):
                lims.append(float(v))
        if lims:
            lo, hi = min(lims), max(lims)
            pad = (hi - lo) * 0.05 + 1e-6
            plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.35, linewidth=1)
            plt.xlim(lo - pad, hi + pad)
            plt.ylim(lo - pad, hi + pad)
        plt.xlabel("原始文本 SHAP（相同关键词）")
        plt.ylabel("扰动文本 SHAP（相同关键词）")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        saved_path = save_figure_with_fallback(out_path, dpi=150)
        plt.close()
    return saved_path


@dataclass
class RobustnessRunContext:
    """CLI 一次运行所需的环境快照（输出目录、数据路径、SHAP 缓存与可选分析器）。"""

    out_dir: str
    resolved_excel: Optional[str]
    excel_paths_tried: List[str]
    shap_cache_path: Optional[str]
    use_shap_cache: bool
    analyzer: Any


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(description="归因稳定性：语义等价扰动 vs SHAP 一致性")
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--excel-path", type=str, default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--top-richness", type=int, default=50, help="丰富度 Top-N 样本（与 fidelity 一致）")
    parser.add_argument(
        "--task",
        type=str,
        default="accident_type",
        choices=["accident_type", "damage_level", "casualty_count"],
    )
    parser.add_argument("--shap-window-sizes", type=str, default="1")
    parser.add_argument("--top-k-jaccard", type=int, default=10, metavar="K", help="Top-K 重要特征 Jaccard 的 K")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--shap-cache-path", type=str, default=DEFAULT_SHAP_CACHE_PATH)
    parser.add_argument("--no-shap-cache", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条（默认开启：样本级 tqdm + 非 demo 时词级 SHAP 按句 tqdm）",
    )
    parser.add_argument(
        "--demo-shap",
        action="store_true",
        help="不加载 Qwen，词级 SHAP 用伪随机（仅验证流程）",
    )
    parser.add_argument(
        "--use-synthetic-samples",
        action="store_true",
        help="不读取 Excel，使用内置合成事故叙述（与 --top-richness 组合指定条数；用于无数据表时的烟测）",
    )
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> RobustnessRunContext:
    """
    创建输出目录、解析 Excel 路径；在非 demo 模式下按需配置 torch 设备并加载 Qwen 分析器。
    """
    out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "robustness_stability_output")
    os.makedirs(out_dir, exist_ok=True)

    resolved_excel, tried = resolve_excel_path(args.excel_path)
    if not resolved_excel and not args.use_synthetic_samples:
        logger.warning(
            "未找到 Excel，请检查路径或使用 --use-synthetic-samples。尝试过的路径: %s",
            tried,
        )

    use_cache = not args.no_shap_cache
    cache_p = args.shap_cache_path if use_cache else None

    analyzer: Any = None
    if not args.demo_shap:
        try:
            import torch as _torch

            if not _torch.cuda.is_available():
                os.environ.setdefault("BERT_FIDELITY_DEVICE_MAP", "cpu")
        except ImportError:
            pass
        base = resolve_base_model_path(args.base_model_path)
        adapter = resolve_adapter_path(args.adapter_path)
        if not base:
            raise SystemExit(
                "未找到 Qwen 基座。请放置模型到 "
                f"{DEFAULT_QWEN_BASE_PATH} 或使用 --base-model-path / QWEN_BASE_MODEL_PATH"
            )
        logger.info("基座: %s\nLoRA: %s", base, adapter)
        Cls = get_deterministic_analyzer_class()
        analyzer = Cls(base, adapter)

    return RobustnessRunContext(
        out_dir=out_dir,
        resolved_excel=resolved_excel,
        excel_paths_tried=list(tried),
        shap_cache_path=cache_p,
        use_shap_cache=use_cache,
        analyzer=analyzer,
    )


def main() -> None:
    args = parse_args()
    try:
        ws = parse_shap_window_sizes(args.shap_window_sizes)
    except ValueError as e:
        raise SystemExit(str(e))

    ctx = setup_environment(args)

    if args.use_synthetic_samples:
        samples = build_synthetic_samples(args.top_richness, args.task)
        list_csv = os.path.join(ctx.out_dir, f"robustness_synthetic_n{len(samples)}_{args.task}_samples.csv")
        # [优化] 直接由样本 dict 建表并限定列，避免 list comprehension 逐条复制小 dict
        target_columns = ["text", "target_label", "target_type"]
        sample_df = pd.DataFrame(samples, columns=target_columns)
        list_csv = save_dataframe_csv_with_fallback(sample_df, list_csv)
        logger.info("使用合成样本: %s 条，清单: %s", len(samples), list_csv)
    else:
        if not ctx.resolved_excel:
            raise SystemExit("无法读取数据表，请指定存在的 --excel-path，或加 --use-synthetic-samples")
        samples, top_df = samples_top_richness_from_excel(
            ctx.resolved_excel,
            args.top_richness,
            args.task,
            demo_shap=False,
        )
        list_csv = os.path.join(ctx.out_dir, f"robustness_richness_top{args.top_richness}_{args.task}_samples.csv")
        list_csv = save_dataframe_csv_with_fallback(top_df, list_csv)
        logger.info("样本数: %s，清单: %s", len(samples), list_csv)

    result = robustness_check(
        ctx.analyzer,
        samples,
        window_sizes=ws,
        top_k_jaccard=args.top_k_jaccard,
        shap_cache_path=ctx.shap_cache_path,
        use_shap_cache=ctx.use_shap_cache,
        show_progress=not args.no_progress,
        demo_shap=args.demo_shap,
    )

    logger.info("")
    logger.info("========== 归因稳定性结果 ==========")
    logger.info(
        "平均 Pearson 相关系数（相同领域关键词上的 SHAP 向量）: %.4f",
        result["mean_pearson_r"],
    )
    logger.info(
        "平均 Top-%s Jaccard 重合度: %.4f",
        args.top_k_jaccard,
        result["mean_jaccard_topk"],
    )
    logger.info(
        "散点图点数（关键词级配对）: %s",
        len(result["scatter_x_original_shap"]),
    )

    png = os.path.join(ctx.out_dir, f"robustness_shap_scatter_{args.task}_top{args.top_richness}.png")
    saved_png = _plot_scatter(
        result,
        png,
        title=f"归因稳定性 | {args.task} | n={result['n_samples']} | "
        f"mean_r={result['mean_pearson_r']:.3f}",
    )
    logger.info("散点图已保存: %s", saved_png)


if __name__ == "__main__":
    main()
