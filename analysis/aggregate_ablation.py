#!/usr/bin/env python3
"""Aggregate TCN-family compression ablations into paper-ready summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = RESEARCH_ROOT / "experiments" / "compression" / "runs" / "tcn_family_ablation"
PHASES = ("LR", "LS", "PSw", "Sw")


def load_metrics(path: Path) -> dict[str, Any] | None:
    for candidate in (path / "metrics.json", path):
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return None


def scan_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for metrics_path in sorted(root.glob("**/metrics.json")):
        data = json.loads(metrics_path.read_text())
        rel = metrics_path.parent.relative_to(root)
        parts = rel.parts
        model = parts[0] if parts else "unknown"
        study = parts[1] if len(parts) > 1 else "main"
        tag = "/".join(parts[2:]) if len(parts) > 2 else parts[-1]
        rows.append(
            {
                "model": model,
                "study": study,
                "tag": tag,
                "path": str(metrics_path.parent.relative_to(RESEARCH_ROOT)),
                **data,
            }
        )
    return rows


def pareto_frontier(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(points, key=lambda p: p["size_kb"])
    frontier: list[dict[str, Any]] = []
    dominated: list[dict[str, Any]] = []
    best_f1 = float("-inf")
    for pt in ordered:
        if pt["macro_f1"] > best_f1:
            frontier.append(pt)
            best_f1 = pt["macro_f1"]
        else:
            dominated.append(pt)
    return frontier, dominated


def summarize_finetune(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    out = []
    for row in sorted(
        [r for r in rows if r["model"] == model and r["study"] == "finetune"],
        key=lambda r: (r.get("name", ""), r.get("finetune_epochs", 0)),
    ):
        out.append(
            {
                "config": row.get("name"),
                "finetune_epochs": row.get("finetune_epochs"),
                "macro_f1": row.get("macro_f1"),
                "phase_f1": row.get("phase_f1"),
                "params": row.get("params"),
                "size_kb": row.get("size_kb"),
            }
        )
    return out


def summarize_pareto_grid(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    grid_rows = [r for r in rows if r["model"] == model and r["study"] == "pareto_grid"]
    points = [
        {
            "name": r["name"],
            "size_kb": r["size_kb"],
            "macro_f1": r["macro_f1"],
            "phase_f1": r.get("phase_f1", {}),
            "params": r.get("params"),
            "keep_ratio": r.get("keep_ratio"),
            "quant_bits": r.get("quant_bits"),
        }
        for r in grid_rows
    ]
    frontier, dominated = pareto_frontier(points)
    psw_drops = []
    fp32 = next((p for p in points if p["name"] == "FP32"), None)
    if fp32:
        base_psw = fp32["phase_f1"].get("PSw", fp32["phase_f1"].get("PSw"))
        for p in points:
            if p["name"] == "FP32":
                continue
            psw = p["phase_f1"].get("PSw")
            if psw is not None and base_psw is not None:
                psw_drops.append({"name": p["name"], "delta_psw_pp": round((psw - base_psw) * 100, 2)})
    psw_drops.sort(key=lambda x: x["delta_psw_pp"])
    return {
        "points": points,
        "frontier": [p["name"] for p in frontier],
        "dominated": [p["name"] for p in dominated],
        "psw_drops_sorted": psw_drops,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ABLATION_ROOT)
    p.add_argument("--out", type=Path, default=RESEARCH_ROOT / "analysis" / "ablation_summary.json")
    p.add_argument("--markdown", type=Path, default=RESEARCH_ROOT / "analysis" / "ABLATION_TABLES.md")
    args = p.parse_args()

    rows = scan_runs(args.root)
    summary: dict[str, Any] = {"root": str(args.root.relative_to(RESEARCH_ROOT)), "models": {}}
    md_sections: list[str] = ["# TCN-Family Ablation Tables", ""]

    for model in ("tcn",):
        summary["models"][model] = {
            "pareto_grid": summarize_pareto_grid(rows, model),
            "finetune_ablation": summarize_finetune(rows, model),
        }
        grid = summary["models"][model]["pareto_grid"]
        md_sections.append(f"## {model.upper()} — Pareto grid ({len(grid['points'])} points)")
        md_sections.append(
            markdown_table(
                ["Config", "Params", "Size (KB)", "Macro F1 (%)", "PSw F1 (%)"],
                [
                    [
                        pt["name"],
                        f"{pt['params']:,}",
                        f"{pt['size_kb']:.1f}",
                        f"{pt['macro_f1'] * 100:.2f}",
                        f"{pt['phase_f1'].get('PSw', 0) * 100:.2f}",
                    ]
                    for pt in sorted(grid["points"], key=lambda x: x["size_kb"])
                ],
            )
        )
        md_sections.append("")
        md_sections.append(f"Pareto frontier: {', '.join(grid['frontier'])}")
        md_sections.append("")

        ft = summary["models"][model]["finetune_ablation"]
        if ft:
            md_sections.append(f"## {model.upper()} — Prune50 fine-tune schedule")
            md_sections.append(
                markdown_table(
                    ["Config", "Fine-tune epochs", "Macro F1 (%)", "PSw F1 (%)"],
                    [
                        [
                            r["config"],
                            str(r.get("finetune_epochs", "")),
                            f"{r['macro_f1'] * 100:.2f}",
                            f"{r.get('phase_f1', {}).get('PSw', 0) * 100:.2f}",
                        ]
                        for r in ft
                    ],
                )
            )
            md_sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    args.markdown.write_text("\n".join(md_sections))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
