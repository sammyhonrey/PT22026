"""Known contrast values and every declared text surface, without a browser claim."""

from pathlib import Path
import pytest

from scripts.check_ui_contrast import check_palettes, contrast


def test_reference_contrast_and_current_palette_surfaces():
    assert contrast("#ffffff", "#000000") == pytest.approx(21)
    assert contrast("#ffffff", "#ffffff") == pytest.approx(1)
    assert contrast("#777777", "#ffffff") < 4.5  # Do not round a near failure into a pass.
    result = check_palettes(Path(__file__).parents[1] / "config/palettes.json")
    assert result["passed"], [row for row in result["checks"] if not row["passed"]]
