"""
Generates clean, standalone SVG visual figures from evaluation results.
Requires zero external dependencies (pure Python SVG generator).
"""

import os
import json
import math

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")
FIGURES_DIR = os.path.dirname(__file__)


def create_bar_chart_svg(title: str, categories: list, series: dict, filename: str, y_label: str = "Rate (%)"):
    """
    series is dict: { "Series Name": [val1, val2, ...] }
    values are in percentages (0-100).
    """
    width = 750
    height = 420
    margin = {"top": 60, "right": 40, "bottom": 80, "left": 70}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    num_cats = len(categories)
    num_series = len(series)

    group_w = chart_w / num_cats
    bar_w = (group_w * 0.7) / max(1, num_series)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#ffffff; font-family:system-ui, -apple-system, sans-serif;">',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">{title}</text>',
        f'<g transform="translate({margin["left"]}, {margin["top"]})">'
    ]

    # Gridlines and Y-axis
    y_steps = [0, 25, 50, 75, 100]
    for y_val in y_steps:
        y_pos = chart_h - (y_val / 100.0) * chart_h
        svg.append(f'<line x1="0" y1="{y_pos}" x2="{chart_w}" y2="{y_pos}" stroke="#e2e8f0" stroke-dasharray="4,4" />')
        svg.append(f'<text x="-12" y="{y_pos + 4}" text-anchor="end" font-size="12" fill="#64748b">{y_val}%</text>')

    svg.append(f'<line x1="0" y1="{chart_h}" x2="{chart_w}" y2="{chart_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg.append(f'<line x1="0" y1="0" x2="0" y2="{chart_h}" stroke="#94a3b8" stroke-width="1.5" />')

    # Bars
    for c_idx, cat in enumerate(categories):
        cx = c_idx * group_w + group_w * 0.15
        for s_idx, (s_name, s_vals) in enumerate(series.items()):
            val = s_vals[c_idx] if c_idx < len(s_vals) else 0.0
            bh = (val / 100.0) * chart_h
            bx = cx + s_idx * bar_w
            by = chart_h - bh
            color = colors[s_idx % len(colors)]
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.9:.1f}" height="{bh:.1f}" fill="{color}" rx="3" />')
            if bh > 18:
                svg.append(f'<text x="{bx + bar_w*0.45:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">{val:.1f}%</text>')

        # X label
        svg.append(f'<text x="{c_idx * group_w + group_w/2:.1f}" y="{chart_h + 24}" text-anchor="middle" font-size="12" font-weight="500" fill="#334155">{cat}</text>')

    # Legend
    legend_y = height - 25
    total_legend_w = sum(len(name) * 8 + 35 for name in series.keys())
    start_lx = (width - total_legend_w) / 2
    cur_lx = start_lx
    for s_idx, (s_name, _) in enumerate(series.items()):
        color = colors[s_idx % len(colors)]
        svg.append(f'<rect x="{cur_lx}" y="{legend_y - 10}" width="14" height="14" fill="{color}" rx="2" />')
        svg.append(f'<text x="{cur_lx + 20}" y="{legend_y + 2}" font-size="12" fill="#475569">{s_name}</text>')
        cur_lx += len(s_name) * 8 + 35

    svg.append('</g>')
    svg.append('</svg>')

    out_path = os.path.join(FIGURES_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated figure: {out_path}")


def generate_all_figures():
    # 1. Figure 1: Retrieval Recall @ 4, 8, 12 by Category
    e2_path = os.path.join(BASELINE_DIR, "e2_retrieval_summary.json")
    if os.path.exists(e2_path):
        with open(e2_path, "r", encoding="utf-8") as f:
            e2 = json.load(f)
        cats = ["Single Fact", "Multi-Chunk", "Multi-Doc", "Paraphrase", "Conflict"]
        rec4 = [
            e2["by_category"].get("single_chunk_fact", {}).get("recall_at_4", 0) * 100,
            e2["by_category"].get("multi_chunk_single_doc", {}).get("recall_at_4", 0) * 100,
            e2["by_category"].get("multi_document", {}).get("recall_at_4", 0) * 100,
            e2["by_category"].get("paraphrase", {}).get("recall_at_4", 0) * 100,
            e2["by_category"].get("distractor_conflict", {}).get("recall_at_4", 0) * 100,
        ]
        rec12 = [
            e2["by_category"].get("single_chunk_fact", {}).get("recall_at_12", 0) * 100,
            e2["by_category"].get("multi_chunk_single_doc", {}).get("recall_at_12", 0) * 100,
            e2["by_category"].get("multi_document", {}).get("recall_at_12", 0) * 100,
            e2["by_category"].get("paraphrase", {}).get("recall_at_12", 0) * 100,
            e2["by_category"].get("distractor_conflict", {}).get("recall_at_12", 0) * 100,
        ]
        create_bar_chart_svg(
            "Retrieval Recall by Query Category (k=4 vs k=12)",
            cats,
            {"Recall@4": rec4, "Recall@12": rec12},
            "figure1_retrieval_recall.svg"
        )

    # 2. Figure 2: Position-Sensitivity Gap (H1/H2)
    pos_cats = ["Chunk Start (Tokens 0-50)", "Chunk End (Tokens >256)"]
    if os.path.exists(e2_path):
        pos_h1 = e2.get("position_sensitive_h1_analysis", {})
        start_r4 = pos_h1.get("start_of_chunk", {}).get("recall_at_4", 1.0) * 100
        end_r4 = pos_h1.get("end_of_chunk", {}).get("recall_at_4", 0.0) * 100
        start_r12 = pos_h1.get("start_of_chunk", {}).get("recall_at_12", 1.0) * 100
        end_r12 = pos_h1.get("end_of_chunk", {}).get("recall_at_12", 0.0) * 100
        create_bar_chart_svg(
            "Position-Sensitive Recall Gap (Hypothesis H1 Truncation)",
            pos_cats,
            {"Recall@4": [start_r4, end_r4], "Recall@12": [start_r12, end_r12]},
            "figure2_position_sensitivity.svg"
        )

    # 3. Figure 3: End-to-End Ablation Comparison
    e4_path = os.path.join(BASELINE_DIR, "e4_ablation_summary.json")
    if os.path.exists(e4_path):
        with open(e4_path, "r", encoding="utf-8") as f:
            e4 = json.load(f)
        arms = ["Standard RAG", "Similarity Only", "No Retry", "Full System"]
        keys = ["Arm_B_Standard_RAG", "Arm_D_Similarity_Only", "Arm_C_No_Retry", "Arm_A_Full_System"]
        
        corr_vals = [e4.get(k, {}).get("answer_correctness", {}).get("rate", 0.0) * 100 for k in keys]
        unsupp_vals = [e4.get(k, {}).get("unsupported_answer_rate", {}).get("rate", 0.0) * 100 for k in keys]
        over_abs_vals = [e4.get(k, {}).get("over_abstention_rate", {}).get("rate", 0.0) * 100 for k in keys]

        create_bar_chart_svg(
            "End-to-End Ablation: Correctness vs Hallucination vs Over-Abstention",
            arms,
            {
                "Answer Correctness": corr_vals,
                "Unsupported Answer Rate": unsupp_vals,
                "Over-Abstention Rate": over_abs_vals
            },
            "figure3_ablation_comparison.svg"
        )


if __name__ == "__main__":
    generate_all_figures()
