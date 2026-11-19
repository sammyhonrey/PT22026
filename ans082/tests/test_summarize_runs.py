import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from summarize_runs import collect, plot_comparison, to_markdown  # noqa: E402


def _write_summary(folder, name, trainable, best_f1):
    summary = {
        "experiment": name, "model": "vit", "total_parameters": 85_806_346,
        "trainable_parameters": trainable, "best_epoch": 7, "best_val_acc": 0.9858,
        "best_val_macro_f1": best_f1, "total_train_time_s": 1251.2, "mean_epoch_time_s": 156.4,
    }
    (folder / f"{name}_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_collect_builds_one_row_per_experiment(tmp_path):
    _write_summary(tmp_path, "E3_vit_full", 85_806_346, 0.9825)
    _write_summary(tmp_path, "E2_vit_head_only", 7_690, 0.93)
    table = collect(tmp_path)
    assert list(table["experiment"]) == ["E2_vit_head_only", "E3_vit_full"]
    assert table.loc[1, "total_train_min"] == 20.9
    assert table.loc[0, "trainable_params_M"] == 0.008


def test_markdown_has_a_header_and_one_line_per_row(tmp_path):
    _write_summary(tmp_path, "E3_vit_full", 85_806_346, 0.9825)
    text = to_markdown(collect(tmp_path))
    lines = text.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("| experiment")


def test_comparison_plot_is_written(tmp_path):
    _write_summary(tmp_path, "E3_vit_full", 85_806_346, 0.9825)
    _write_summary(tmp_path, "E2_vit_head_only", 7_690, 0.93)
    out_png = tmp_path / "figures" / "experiments_val.png"
    plot_comparison(collect(tmp_path), out_png)
    assert out_png.exists() and out_png.stat().st_size > 0
