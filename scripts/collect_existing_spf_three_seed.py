#!/usr/bin/env python3
"""Collect existing SPF ablation results without running experiments."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VARIANT_SPECS = {
    "local_only": (0.00, 0.00),
    "anchor_only_original": (0.00, 0.10),
    "weak_fusion_only": (0.01, 0.00),
}


@dataclass(frozen=True)
class Run:
    variant: str
    gamma: float
    shared_lambda: float
    seed: int
    acc_csv_path: Path
    log_path: Path
    detection_evidence: str


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize already-existing SPF DTD three-seed results."
    )
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--variants", required=True, nargs="+")
    parser.add_argument("--expected-rounds", required=True, type=int)
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        fail(f"failed to read {path}: {exc}")


def first_unique(
    label: str, values: Iterable[tuple[str, str]], acc_path: Path, problems: list[str]
) -> tuple[str | None, str]:
    cleaned = [(value.strip(), source) for value, source in values if value.strip()]
    unique = sorted({value for value, _ in cleaned})
    evidence = ", ".join(f"{source}={value}" for value, source in cleaned)
    if len(unique) > 1:
        problems.append(
            f"{acc_path}: conflicting {label} values {unique}; evidence: {evidence}"
        )
        return None, evidence
    if not unique:
        problems.append(f"{acc_path}: missing unique {label}; evidence: {evidence}")
        return None, evidence
    return unique[0], evidence


def regex_values(pattern: str, text: str, source: str) -> list[tuple[str, str]]:
    return [(match.group(1), source) for match in re.finditer(pattern, text, re.M)]


def detect_run(acc_path: Path, dataset: str, variants: set[str]) -> tuple[Run | None, list[str]]:
    problems: list[str] = []
    log_candidates = sorted(
        path
        for path in acc_path.parent.iterdir()
        if path.is_file()
        and (
            path.name == "log.txt"
            or path.suffix == ".log"
            or "log" in path.name.lower()
        )
    )
    if len(log_candidates) != 1:
        problems.append(
            f"{acc_path}: expected exactly one related log in same directory, "
            f"found {[str(path) for path in log_candidates]}"
        )
        return None, problems

    log_path = log_candidates[0]
    log_text = read_text(log_path)
    path_text = str(acc_path)

    dataset_values = regex_values(r"^dataset:\s*([^\s]+)\s*$", log_text, "log:dataset")
    dataset_values += regex_values(r"--dataset\s+([^\s]+)", log_text, "log:--dataset")
    dataset_values += regex_values(r"(?:^|/)(" + re.escape(dataset) + r")(?:/|$)", path_text, "path")
    dataset_value, dataset_evidence = first_unique(
        "dataset", dataset_values, acc_path, problems
    )

    seed_values = regex_values(r"^seed:\s*(\d+)\s*$", log_text, "log:seed")
    seed_values += regex_values(r"--seed\s+(\d+)", log_text, "log:--seed")
    seed_values += regex_values(r"(?:^|/)seed_(\d+)(?:/|$)", path_text, "path:seed")
    seed_value, seed_evidence = first_unique("seed", seed_values, acc_path, problems)

    gamma_values = regex_values(
        r"^spf_gamma_init:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
        log_text,
        "log:spf_gamma_init",
    )
    gamma_values += regex_values(
        r"--spf_gamma_init\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        log_text,
        "log:--spf_gamma_init",
    )
    gamma_values += regex_values(
        r"(?:^|/)spf_g([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:_|/|$)",
        path_text,
        "path:spf_g",
    )
    gamma_value, gamma_evidence = first_unique("gamma", gamma_values, acc_path, problems)

    lambda_values = regex_values(
        r"^spf_shared_lambda:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
        log_text,
        "log:spf_shared_lambda",
    )
    lambda_values += regex_values(
        r"--spf_shared_lambda\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        log_text,
        "log:--spf_shared_lambda",
    )
    lambda_value, lambda_evidence = first_unique(
        "shared_lambda", lambda_values, acc_path, problems
    )

    if problems:
        return None, problems

    try:
        seed = int(seed_value or "")
        gamma = float(gamma_value or "")
        shared_lambda = float(lambda_value or "")
    except ValueError:
        problems.append(f"{acc_path}: non-numeric seed/gamma/shared_lambda")
        return None, problems

    if (dataset_value or "").lower() != dataset.lower():
        return None, []

    matched = [
        name
        for name in variants
        if name in VARIANT_SPECS
        and math.isclose(gamma, VARIANT_SPECS[name][0], rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            shared_lambda, VARIANT_SPECS[name][1], rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if len(matched) > 1:
        problems.append(
            f"{acc_path}: matches multiple variants {matched}; "
            f"gamma={gamma}, shared_lambda={shared_lambda}"
        )
        return None, problems
    if not matched:
        return None, []

    evidence = "; ".join(
        item
        for item in [
            dataset_evidence,
            seed_evidence,
            gamma_evidence,
            lambda_evidence,
            f"log_path={log_path}",
        ]
        if item
    )
    return (
        Run(
            variant=matched[0],
            gamma=gamma,
            shared_lambda=shared_lambda,
            seed=seed,
            acc_csv_path=acc_path,
            log_path=log_path,
            detection_evidence=evidence,
        ),
        [],
    )


def discover_runs(
    results_root: Path, dataset: str, seeds: list[int], variants: list[str]
) -> list[Run]:
    if not results_root.exists():
        fail(f"results root does not exist: {results_root}")
    unknown_variants = [variant for variant in variants if variant not in VARIANT_SPECS]
    if unknown_variants:
        fail(f"unknown variants requested: {unknown_variants}")

    acc_paths = sorted(results_root.rglob("acc.csv"))
    if not acc_paths:
        fail(f"no acc.csv files found under {results_root}")

    detected: list[Run] = []
    problems: list[str] = []
    for acc_path in acc_paths:
        run, run_problems = detect_run(acc_path, dataset, set(variants))
        problems.extend(run_problems)
        if run is not None and run.seed in seeds:
            detected.append(run)

    if problems:
        fail("run detection failed:\n" + "\n".join(problems))

    duplicate_keys: dict[tuple[str, int], list[Run]] = {}
    for run in detected:
        duplicate_keys.setdefault((run.variant, run.seed), []).append(run)
    duplicates = {key: value for key, value in duplicate_keys.items() if len(value) > 1}
    if duplicates:
        lines = []
        for (variant, seed), runs in sorted(duplicates.items()):
            lines.append(f"duplicate target for variant={variant}, seed={seed}:")
            lines.extend(f"  {run.acc_csv_path}" for run in runs)
        fail("\n".join(lines))

    expected_keys = {(variant, seed) for variant in variants for seed in seeds}
    found_keys = {(run.variant, run.seed) for run in detected}
    missing = sorted(expected_keys - found_keys)
    extra = sorted(found_keys - expected_keys)
    if missing or extra or len(detected) != len(expected_keys):
        lines = [
            f"expected exactly {len(expected_keys)} target runs, found {len(detected)}",
        ]
        if missing:
            lines.append(f"missing variant/seed pairs: {missing}")
        if extra:
            lines.append(f"unexpected variant/seed pairs: {extra}")
        lines.append("detected candidate target paths:")
        lines.extend(
            f"  variant={run.variant}, seed={run.seed}, gamma={run.gamma}, "
            f"shared_lambda={run.shared_lambda}, acc={run.acc_csv_path}"
            for run in sorted(detected, key=lambda item: (item.variant, item.seed))
        )
        fail("\n".join(lines))

    return sorted(detected, key=lambda item: (variants.index(item.variant), item.seed))


def read_round_accuracies(acc_path: Path, expected_rounds: int) -> np.ndarray:
    try:
        with acc_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                fail(f"{acc_path}: missing CSV header")
            fieldnames = reader.fieldnames
            epoch_cols = []
            for col in fieldnames:
                match = re.fullmatch(r"epoch_(\d+)", col)
                if match:
                    epoch_cols.append((int(match.group(1)), col))
            epoch_cols.sort(key=lambda item: item[0])
            if len(epoch_cols) != expected_rounds:
                fail(
                    f"{acc_path}: expected {expected_rounds} epoch columns, "
                    f"found {len(epoch_cols)}"
                )
            expected_indices = list(range(expected_rounds))
            actual_indices = [idx for idx, _ in epoch_cols]
            if actual_indices != expected_indices:
                fail(f"{acc_path}: epoch columns are not epoch_0..epoch_{expected_rounds - 1}")

            rows: list[list[float]] = []
            for row_number, row in enumerate(reader, start=2):
                values = []
                for _, col in epoch_cols:
                    raw = (row.get(col) or "").strip()
                    try:
                        value = float(raw)
                    except ValueError:
                        fail(f"{acc_path}:{row_number}: non-numeric value in {col}: {raw!r}")
                    if math.isnan(value):
                        fail(f"{acc_path}:{row_number}: NaN value in {col}")
                    if not math.isfinite(value):
                        fail(f"{acc_path}:{row_number}: non-finite value in {col}: {value}")
                    values.append(value)
                rows.append(values)
    except OSError as exc:
        fail(f"failed to read {acc_path}: {exc}")

    if not rows:
        fail(f"{acc_path}: expected at least 1 client row")
    values = np.asarray(rows, dtype=float)
    if values.shape[0] < 1:
        fail(f"{acc_path}: expected at least 1 client row")
    return values.mean(axis=0)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_mean_std(mean: float, std: float) -> str:
    return f"{format_float(mean)}±{format_float(std)}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(
    runs: list[Run], expected_rounds: int
) -> tuple[list[dict[str, object]], dict[tuple[str, int], dict[str, object]]]:
    rows: list[dict[str, object]] = []
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for run in runs:
        round_acc = read_round_accuracies(run.acc_csv_path, expected_rounds)
        last10 = round_acc[-10:]
        best_acc = float(round_acc.max())
        best_round_0 = int(np.argmax(round_acc))
        row: dict[str, object] = {
            "variant": run.variant,
            "gamma": run.gamma,
            "shared_lambda": run.shared_lambda,
            "seed": run.seed,
            "last10_avg": float(last10.mean()),
            "last10_std": float(np.std(last10, ddof=0)),
            "best_acc": best_acc,
            "best_round_0_based": best_round_0,
            "best_round_1_based": best_round_0 + 1,
            "final_acc": float(round_acc[-1]),
            "acc_csv_path": str(run.acc_csv_path),
            "log_path": str(run.log_path),
        }
        rows.append(row)
        by_key[(run.variant, run.seed)] = row
    return rows, by_key


def summarize_by_variant(
    variants: list[str], seeds: list[int], metrics: dict[tuple[str, int], dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        per_seed = [metrics[(variant, seed)] for seed in seeds]
        gamma, shared_lambda = VARIANT_SPECS[variant]
        row: dict[str, object] = {
            "variant": variant,
            "gamma": gamma,
            "shared_lambda": shared_lambda,
            "last10_avg_mean": float(np.mean([item["last10_avg"] for item in per_seed])),
            "last10_avg_sample_std": float(
                np.std([item["last10_avg"] for item in per_seed], ddof=1)
            ),
            "last10_temporal_std_mean": float(
                np.mean([item["last10_std"] for item in per_seed])
            ),
            "last10_temporal_std_sample_std": float(
                np.std([item["last10_std"] for item in per_seed], ddof=1)
            ),
            "best_acc_mean": float(np.mean([item["best_acc"] for item in per_seed])),
            "best_acc_sample_std": float(
                np.std([item["best_acc"] for item in per_seed], ddof=1)
            ),
            "best_round_0_based_per_seed": "; ".join(
                f"{seed}:{metrics[(variant, seed)]['best_round_0_based']}" for seed in seeds
            ),
            "final_acc_mean": float(np.mean([item["final_acc"] for item in per_seed])),
            "final_acc_sample_std": float(
                np.std([item["final_acc"] for item in per_seed], ddof=1)
            ),
        }
        rows.append(row)
    return rows


def paired_deltas(
    variants: list[str], seeds: list[int], metrics: dict[tuple[str, int], dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        if variant == "local_only":
            continue
        delta_last10 = [
            float(metrics[(variant, seed)]["last10_avg"])
            - float(metrics[("local_only", seed)]["last10_avg"])
            for seed in seeds
        ]
        delta_final = [
            float(metrics[(variant, seed)]["final_acc"])
            - float(metrics[("local_only", seed)]["final_acc"])
            for seed in seeds
        ]
        row: dict[str, object] = {
            "variant": variant,
            "baseline_variant": "local_only",
            "delta_last10_by_seed": "; ".join(
                f"{seed}:{format_float(delta)}" for seed, delta in zip(seeds, delta_last10)
            ),
            "delta_final_by_seed": "; ".join(
                f"{seed}:{format_float(delta)}" for seed, delta in zip(seeds, delta_final)
            ),
            "delta_last10_mean": float(np.mean(delta_last10)),
            "delta_last10_sample_std": float(np.std(delta_last10, ddof=1)),
            "delta_final_mean": float(np.mean(delta_final)),
            "delta_final_sample_std": float(np.std(delta_final, ddof=1)),
            "last10_positive_seeds": f"{sum(delta > 0 for delta in delta_last10)}/{len(seeds)}",
            "final_positive_seeds": f"{sum(delta > 0 for delta in delta_final)}/{len(seeds)}",
        }
        rows.append(row)
    return rows


def write_markdown(
    path: Path,
    variants: list[str],
    summary_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
) -> None:
    summary = {row["variant"]: row for row in summary_rows}
    deltas = {row["variant"]: row for row in delta_rows}
    lines = [
        "| Variant | Gamma | Shared Lambda | Last10 Avg (mean±std) | Last10 Temporal Std (mean±std) | Best Acc (mean±std) | Best Round 0-based per seed | Final Acc (mean±std) | Delta Last10 vs Local | Last10 Wins | Delta Final vs Local | Final Wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for variant in variants:
        row = summary[variant]
        delta = deltas.get(variant)
        lines.append(
            "| {variant} | {gamma} | {shared_lambda} | {last10_avg} | "
            "{last10_temporal_std} | {best_acc} | {best_rounds} | {final_acc} | "
            "{delta_last10} | {last10_wins} | {delta_final} | {final_wins} |".format(
                variant=variant,
                gamma=format_float(float(row["gamma"])),
                shared_lambda=format_float(float(row["shared_lambda"])),
                last10_avg=format_mean_std(
                    float(row["last10_avg_mean"]),
                    float(row["last10_avg_sample_std"]),
                ),
                last10_temporal_std=format_mean_std(
                    float(row["last10_temporal_std_mean"]),
                    float(row["last10_temporal_std_sample_std"]),
                ),
                best_acc=format_mean_std(
                    float(row["best_acc_mean"]),
                    float(row["best_acc_sample_std"]),
                ),
                best_rounds=row["best_round_0_based_per_seed"],
                final_acc=format_mean_std(
                    float(row["final_acc_mean"]),
                    float(row["final_acc_sample_std"]),
                ),
                delta_last10=(
                    format_mean_std(
                        float(delta["delta_last10_mean"]),
                        float(delta["delta_last10_sample_std"]),
                    )
                    if delta
                    else "baseline"
                ),
                last10_wins=delta["last10_positive_seeds"] if delta else "baseline",
                delta_final=(
                    format_mean_std(
                        float(delta["delta_final_mean"]),
                        float(delta["delta_final_sample_std"]),
                    )
                    if delta
                    else "baseline"
                ),
                final_wins=delta["final_positive_seeds"] if delta else "baseline",
            )
        )

    lines.extend(["", "## Decision Aid", ""])
    label_map = {
        "anchor_only_original": "Anchor-only",
        "weak_fusion_only": "Weak-fusion-only",
    }
    for variant in ["anchor_only_original", "weak_fusion_only"]:
        delta = deltas[variant]
        lines.extend(
            [
                f"- {label_map[variant]} relative to Local-only:",
                "  paired Delta Last10 = "
                + format_mean_std(
                    float(delta["delta_last10_mean"]),
                    float(delta["delta_last10_sample_std"]),
                ),
                f"  positive seeds = {delta['last10_positive_seeds']}",
                "",
            ]
        )
    lines.extend(
        [
            "- Warning:",
            "  n=3 is insufficient to claim statistical significance.",
            "  Last10 Avg is the primary metric.",
            "  Best Acc cannot be used alone to select a method.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset = args.dataset.lower()
    variants = args.variants
    seeds = args.seeds

    runs = discover_runs(args.results_root, dataset, seeds, variants)
    per_seed_rows, metrics = build_metrics(runs, args.expected_rounds)
    summary_rows = summarize_by_variant(variants, seeds, metrics)
    delta_rows = paired_deltas(variants, seeds, metrics)

    output_dir = args.results_root / f"spf_existing_three_seed_summary_{dataset}"
    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    write_csv(
        tmp_dir / "discovered_runs.csv",
        [
            "variant",
            "gamma",
            "shared_lambda",
            "seed",
            "acc_csv_path",
            "log_path",
            "detection_evidence",
        ],
        [
            {
                "variant": run.variant,
                "gamma": run.gamma,
                "shared_lambda": run.shared_lambda,
                "seed": run.seed,
                "acc_csv_path": str(run.acc_csv_path),
                "log_path": str(run.log_path),
                "detection_evidence": run.detection_evidence,
            }
            for run in runs
        ],
    )
    write_csv(
        tmp_dir / "per_seed_metrics.csv",
        [
            "variant",
            "gamma",
            "shared_lambda",
            "seed",
            "last10_avg",
            "last10_std",
            "best_acc",
            "best_round_0_based",
            "best_round_1_based",
            "final_acc",
            "acc_csv_path",
            "log_path",
        ],
        per_seed_rows,
    )
    write_csv(
        tmp_dir / "summary_by_variant.csv",
        [
            "variant",
            "gamma",
            "shared_lambda",
            "last10_avg_mean",
            "last10_avg_sample_std",
            "last10_temporal_std_mean",
            "last10_temporal_std_sample_std",
            "best_acc_mean",
            "best_acc_sample_std",
            "best_round_0_based_per_seed",
            "final_acc_mean",
            "final_acc_sample_std",
        ],
        summary_rows,
    )
    write_csv(
        tmp_dir / "paired_deltas_vs_local_only.csv",
        [
            "variant",
            "baseline_variant",
            "delta_last10_by_seed",
            "delta_final_by_seed",
            "delta_last10_mean",
            "delta_last10_sample_std",
            "delta_final_mean",
            "delta_final_sample_std",
            "last10_positive_seeds",
            "final_positive_seeds",
        ],
        delta_rows,
    )
    write_markdown(tmp_dir / "summary_by_variant.md", variants, summary_rows, delta_rows)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.replace(output_dir)
    print(f"Wrote summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
