import hashlib
from pathlib import Path

import pytest

from issue_classifier import report_assets
from issue_classifier.report_assets import _zoomed_axis_limits, generate_report_assets


def _hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_report_assets_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generated = generate_report_assets(output_dir=first)
    generate_report_assets(output_dir=second)
    names = {path.name for path in generated}
    assert {"cv_comparison.tex", "chosen_repositories.tex", "chosen_classes.tex", "dataset_summary.tex", "error_analysis.tex"} <= names
    assert (first / "figures/cross-validation-candidates.png").is_file()
    assert (first / "figures/chosen-model-confusion-matrix.png").is_file()
    grid = (first / "candidate_grid.tex").read_text(encoding="utf-8")
    assert "Word+char SVC C=1" in grid
    assert '"' not in grid and "parameters" not in grid
    baselines = (first / "baseline_comparison.tex").read_text(encoding="utf-8")
    assert "Our chosen model" in baselines
    assert "Challenge" not in baselines
    assert _hashes(first) == _hashes(second)


def test_report_assets_module_main(monkeypatch, capsys) -> None:
    monkeypatch.setattr(report_assets, "generate_report_assets", lambda: [Path("report/generated/a.tex")])
    assert report_assets.main() == 0
    assert capsys.readouterr().out.strip().endswith("a.tex")


def test_cross_validation_plot_limits_zoom_around_real_candidates() -> None:
    lower, upper = _zoomed_axis_limits([0.73, 0.75], [0.07, 0.06])
    assert lower == pytest.approx(0.64)
    assert upper == pytest.approx(0.83)
