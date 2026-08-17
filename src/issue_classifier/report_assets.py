"""Generate the tables and figures used by the final report."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import ALLOWED_REPOSITORIES
from .dataset_inspection import inspect_dataset
from .stores import CsvIssueStore


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latex(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def _number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _table(headers: list[str], rows: list[list[object]], alignment: str) -> str:
    line_break = " " + chr(92) * 2
    lines = [
        "\\begin{tabular}{" + alignment + "}",
        "\\toprule",
        " & ".join(_latex(header) for header in headers) + line_break,
        "\\midrule",
    ]
    lines.extend(" & ".join(_latex(value) for value in row) + line_break for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _official_setfit_f1(path: Path) -> float:
    pattern = re.compile(r"\|\s*overall\s*\|\s*average\s*\|[^|]+\|[^|]+\|\s*\*\*([0-9.]+)\*\*")
    match = pattern.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Could not find the official SetFit score in {path}")
    return float(match.group(1))


def _candidate_display_name(parameters: dict[str, Any]) -> str:
    if parameters["estimator"] == "DummyClassifier":
        return "Dummy"
    if parameters["estimator"] == "LogisticRegression":
        return f"Word LR C={parameters['C']}"
    return f"Word+char SVC C={parameters['C']}"


def _zoomed_axis_limits(means: list[float], stds: list[float]) -> tuple[float, float]:
    padding = 0.02
    return (
        max(0.0, min(mean - std for mean, std in zip(means, stds, strict=True)) - padding),
        min(1.0, max(mean + std for mean, std in zip(means, stds, strict=True)) + padding),
    )


def _write_figures(output: Path, cv: dict[str, Any], metrics: dict[str, Any]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Report figures require matplotlib") from exc

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            _candidate_display_name(cv["candidate_parameters"][item["candidate"]]["parameters"]),
            item["global_mean"],
            item["global_std"],
        )
        for item in cv["summaries"]
        if cv["candidate_parameters"][item["candidate"]]["parameters"]["estimator"]
        != "DummyClassifier"
    ]
    labels, means, stds = zip(*rows, strict=True)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.errorbar(range(len(labels)), means, yerr=stds, fmt="o", color="#174a7e", capsize=3)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_ylabel("Cross-validation macro-F1")
    axis.set_ylim(*_zoomed_axis_limits(list(means), list(stds)))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    cv_path = figures / "cross-validation-candidates.png"
    figure.savefig(cv_path, dpi=140, metadata={"Software": "issue_classifier report generator"})
    plt.close(figure)

    matrix = metrics["policies"]["primary"]["global"]["aggregate_confusion_matrix"]
    labels = metrics["labels"]
    figure, axis = plt.subplots(figsize=(4.2, 3.8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, value, ha="center", va="center")
    figure.tight_layout()
    confusion_path = figures / "chosen-model-confusion-matrix.png"
    figure.savefig(confusion_path, dpi=140, metadata={"Software": "issue_classifier report generator"})
    plt.close(figure)
    return [cv_path, confusion_path]


def generate_report_assets(project_root: str | Path = ".", output_dir: str | Path = "report/generated") -> list[Path]:
    """Generate stable report inputs from the saved chosen-model results."""

    root = Path(project_root)
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    cv = _load_json(root / "results/cross_validation.json")
    metrics = _load_json(root / "results/chosen-model-seed42/metrics.json")
    manifest = _load_json(root / "results/chosen-model-seed42/manifest.json")
    baseline_paths = {
        name: root / f"official_nlbse24/published_results/{name}/results.json"
        for name in ("setfit", "roberta", "fasttext")
    }
    baselines = {name: _load_json(path) for name, path in baseline_paths.items()}
    official_readme = root / "official_nlbse24/ORIGINAL_REPOSITORY_README.md"
    official_setfit_f1 = _official_setfit_f1(official_readme)
    store = CsvIssueStore(root / "official_nlbse24/data")
    inspection = inspect_dataset(store.load("train"), store.load("test"))
    assets: list[Path] = []

    assets.append(_write(output / "cv_comparison.tex", _table(
        ["Candidate", "Mean macro-F1", "Population std."],
        [[item["candidate"], _number(item["global_mean"]), _number(item["global_std"])] for item in cv["summaries"]],
        "lrr",
    )))
    summary_by_candidate = {item["candidate"]: item for item in cv["summaries"]}
    candidate_rows = []
    for name, spec in cv["candidate_parameters"].items():
        parameters = spec["parameters"]
        if parameters["estimator"] == "DummyClassifier":
            feature_name = "None / Dummy"
        elif parameters["estimator"] == "LogisticRegression":
            feature_name = "Word TF-IDF / LR"
        else:
            feature_name = "Word+char TF-IDF / SVC"
        summary = summary_by_candidate[name]
        candidate_rows.append([_candidate_display_name(parameters), feature_name, _number(summary["global_mean"]), _number(summary["global_std"])])
    assets.append(_write(output / "candidate_grid.tex", _table(
        ["Model", "Features / estimator", "Mean macro-F1", "Population std."], candidate_rows, "llrr"
    )))

    summary_rows = []
    for split in ("train", "test"):
        data = inspection[split]
        summary_rows.append([split, data["row_count"], min(data["repo_counts"].values()), 100, data["missing_values"]["title"], data["missing_values"]["body"], len(data["exact_duplicate_groups"])])
    assets.append(_write(output / "dataset_summary.tex", _table(
        ["Split", "Rows", "Rows/repo", "Rows/repo-label", "Null title", "Null body", "Dup. groups"], summary_rows, "lrrrrrr"
    )))

    primary = metrics["policies"]["primary"]
    leakage = metrics["policies"]["leakage_sensitive"]
    repo_rows = []
    for repository in sorted(ALLOWED_REPOSITORIES):
        for policy_name, values in (("Primary", primary["repositories"][repository]["macro"]), ("Leakage-sensitive", leakage["repositories"][repository]["macro"])):
            repo_rows.append([policy_name, repository, _number(values["precision"]), _number(values["recall"]), _number(values["f1"])])
    assets.append(_write(output / "chosen_repositories.tex", _table(["Policy", "Repository", "Macro precision", "Macro recall", "Macro F1"], repo_rows, "llrrr")))
    class_rows = []
    for policy_name, policy in (("Primary", primary), ("Leakage-sensitive", leakage)):
        for label in metrics["labels"]:
            values = policy["global"]["per_class"][label]
            class_rows.append([policy_name, label, _number(values["precision"]), _number(values["recall"]), _number(values["f1"]), values["support"]])
    assets.append(_write(output / "chosen_classes.tex", _table(["Policy", "Label", "Precision", "Recall", "F1", "Support"], class_rows, "llrrrr")))
    assets.append(_write(output / "policy_summary.tex", _table(
        ["Policy", "Included rows", "Excluded rows", "Cross-repository macro-F1", "Delta vs. primary"],
        [["Primary", primary["included_count"], primary["excluded_count"], _number(primary["global"]["macro_f1"], 6), _number(0, 6)], ["Leakage-sensitive", leakage["included_count"], leakage["excluded_count"], _number(leakage["global"]["macro_f1"], 6), _number(leakage["global"]["macro_f1"] - primary["global"]["macro_f1"], 6)]],
        "lrrrr",
    )))

    baseline_rows = [
        ["Our chosen model", _number(primary["global"]["macro_f1"])],
        ["NLBSE'24 detailed SetFit score", _number(baselines["setfit"]["overall"]["average"]["f1-score"])],
        ["NLBSE'24 RoBERTa score", _number(baselines["roberta"]["overall"]["average"]["f1-score"])],
        ["NLBSE'24 fastText score", _number(baselines["fasttext"]["overall"]["average"]["f1-score"])],
        ["NLBSE'24 headline SetFit score", _number(official_setfit_f1)],
    ]
    assets.append(_write(output / "baseline_comparison.tex", _table(["System", "Overall macro-F1"], baseline_rows, "lr")))

    counts: Counter[tuple[str, str, str]] = Counter()
    with (root / "results/chosen-model-seed42/predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "primary" and row["true_label"] != row["predicted_label"]:
                counts[(row["repo"], row["true_label"], row["predicted_label"])] += 1
    error_rows = [[repo, true, predicted, count] for (repo, true, predicted), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]]
    assets.append(_write(output / "error_analysis.tex", _table(["Repository", "True", "Predicted", "Errors"], error_rows, r"lll@{\hspace{0.8em}}r")))
    assets.extend(_write_figures(output, cv, metrics))
    if manifest.get("run_id") != "chosen-model-seed42":
        raise ValueError("Unexpected chosen-model result manifest run_id")
    return sorted(assets)


def main() -> int:
    for path in generate_report_assets():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_report_assets", "main"]
