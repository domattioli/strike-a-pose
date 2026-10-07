"""Smoke tests for report/plots.py: the coverage and width PNG files, built from a results.csv."""

import csv
from pathlib import Path

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from strike_a_pose.report.plots import (
    COVERAGE_PLOT_NAME,
    WIDTH_PLOT_NAME,
    build_coverage_figure,
    build_width_figure,
    read_result_rows,
    write_plots,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MEASUREMENTS = ("height", "chest", "waist", "hip", "thigh")
PANEL_TITLES = ["Height", "Chest", "Waist", "Hip", "Thigh"]
RESULT_COLUMNS = (
    "cell_id",
    "views",
    "noise_deg",
    "measurement",
    "nominal_level",
    "n_cal",
    "n_test",
    "coverage",
    "median_width_cm",
    "mae_cm",
    "mean_signed_error_cm",
    "clipped_count",
    "in_band",
    "q_hat",
    "seed",
    "config_hash",
    "code_version",
    "hardware_class",
)
# The series labels and colours of the Okabe-Ito palette for the 0, 2, and 5 degree series.
SERIES_COLOURS = {
    "0° placement noise": "#0072B2",
    "2° placement noise": "#E69F00",
    "5° placement noise": "#009E73",
}


def synthetic_coverage(views: int, index: int) -> float:
    """Return a coverage between 0.89 and 0.91 that varies with the view count and measurement."""
    return 0.90 + 0.01 * ((views + index) % 3 - 1)


def synthetic_width(views: int, index: int, noise: float) -> float:
    """Return a width in centimetres that shrinks with more views and grows with noise."""
    return (20.0 + 8.0 * index) / views + 0.2 * noise


def result_row(views: int, noise: float, index: int) -> dict[str, object]:
    """Return one results.csv row (contracts/artifacts.md columns) for one cell and measurement."""
    return {
        "cell_id": f"v{views}_n{noise:g}",
        "views": views,
        "noise_deg": noise,
        "measurement": MEASUREMENTS[index],
        "nominal_level": 0.9,
        "n_cal": 2500,
        "n_test": 2500,
        "coverage": synthetic_coverage(views, index),
        "median_width_cm": synthetic_width(views, index, noise),
        "mae_cm": 1.2,
        "mean_signed_error_cm": 0.1,
        "clipped_count": 0,
        "in_band": "true",
        "q_hat": 1.6,
        "seed": 20261007,
        "config_hash": "3c" * 32,
        "code_version": "0.1.0",
        "hardware_class": "cpu",
    }


def design_rows(views_levels: tuple[int, ...], noise_levels: tuple[float, ...]) -> list[dict]:
    """Return one row per cell and measurement, in the cell order of results.csv."""
    return [
        result_row(views, noise, index)
        for views in views_levels
        for noise in noise_levels
        for index in range(len(MEASUREMENTS))
    ]


def write_rows(path: Path, rows: list[dict]) -> Path:
    """Write the rows as a results.csv with the header of contracts/artifacts.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def series_lines(panel) -> list:
    """Return the lines of one panel that are placement-noise series, in drawing order."""
    return [line for line in panel.get_lines() if line.get_label().endswith("placement noise")]


def legend_labels(figure: Figure) -> list[str]:
    """Return the labels of the figure's one legend."""
    legends = [axis.get_legend() for axis in figure.axes if axis.get_legend() is not None]
    assert len(legends) == 1
    return [text.get_text() for text in legends[0].get_texts()]


@pytest.fixture
def results_csv(tmp_path: Path) -> Path:
    """A results.csv with the full design: 3 view counts, 3 noise levels, 5 measurements."""
    return write_rows(
        tmp_path / "evaluate" / "results.csv", design_rows((1, 2, 4), (0.0, 2.0, 5.0))
    )


def test_writes_the_coverage_and_width_png_files(results_csv: Path, tmp_path: Path) -> None:
    """Two PNG files are written into report/, and no temporary file is left beside them."""
    report = tmp_path / "report"
    coverage_path, width_path = write_plots(results_csv, report)

    assert coverage_path == report / COVERAGE_PLOT_NAME
    assert width_path == report / WIDTH_PLOT_NAME
    assert sorted(entry.name for entry in report.iterdir()) == [
        COVERAGE_PLOT_NAME,
        WIDTH_PLOT_NAME,
    ]
    for path in (coverage_path, width_path):
        data = path.read_bytes()
        assert data.startswith(PNG_SIGNATURE)
        assert len(data) > 10_000


def test_each_figure_has_one_panel_per_measurement_and_one_legend(results_csv: Path) -> None:
    """Each figure uses the Agg canvas, one panel per measurement, and one line per noise."""
    rows = read_result_rows(results_csv)
    for figure in (build_coverage_figure(rows), build_width_figure(rows)):
        assert isinstance(figure, Figure)
        assert isinstance(figure.canvas, FigureCanvasAgg)
        panels = figure.axes[:5]
        assert [panel.get_title() for panel in panels] == PANEL_TITLES
        for panel in panels:
            series = series_lines(panel)
            assert [line.get_label() for line in series] == list(SERIES_COLOURS)
            assert [line.get_color() for line in series] == list(SERIES_COLOURS.values())


def test_legends_name_the_series_and_the_coverage_reference_lines(results_csv: Path) -> None:
    """The coverage legend adds the band and nominal line; the width legend holds the series."""
    rows = read_result_rows(results_csv)
    assert legend_labels(build_width_figure(rows)) == list(SERIES_COLOURS)
    assert legend_labels(build_coverage_figure(rows)) == [
        *SERIES_COLOURS,
        "tolerance band 87% to 93%",
        "nominal 90%",
    ]


def test_coverage_panels_shade_the_band_and_mark_the_nominal_level(results_csv: Path) -> None:
    """Each coverage panel shades 87% to 93% and draws a dashed line at 90%; width panels do not."""
    rows = read_result_rows(results_csv)
    for panel in build_coverage_figure(rows).axes[:5]:
        assert len(panel.patches) == 1
        band = panel.patches[0]
        assert band.get_y() == pytest.approx(87.0)
        assert band.get_height() == pytest.approx(6.0)
        nominal = [line for line in panel.get_lines() if line.get_linestyle() == "--"]
        assert len(nominal) == 1
        assert all(value == pytest.approx(90.0) for value in nominal[0].get_ydata())
    for panel in build_width_figure(rows).axes[:5]:
        assert len(panel.patches) == 0


def test_series_plot_the_values_of_the_results_table(results_csv: Path) -> None:
    """The chest series match the table: coverage in percent and width in cm, by view count."""
    rows = read_result_rows(results_csv)
    chest_coverage = next(
        panel for panel in build_coverage_figure(rows).axes[:5] if panel.get_title() == "Chest"
    )
    chest_width = next(
        panel for panel in build_width_figure(rows).axes[:5] if panel.get_title() == "Chest"
    )
    views = [1, 2, 4]

    coverage_by_noise = {line.get_label(): line for line in series_lines(chest_coverage)}
    line = coverage_by_noise["0° placement noise"]
    assert list(line.get_xdata()) == views
    assert list(line.get_ydata()) == pytest.approx(
        [100.0 * synthetic_coverage(view, 1) for view in views]
    )

    width_by_noise = {line.get_label(): line for line in series_lines(chest_width)}
    line = width_by_noise["5° placement noise"]
    assert list(line.get_xdata()) == views
    assert list(line.get_ydata()) == pytest.approx(
        [synthetic_width(view, 1, 5.0) for view in views]
    )


def test_a_small_configuration_plots_one_series_per_panel(tmp_path: Path) -> None:
    """The test configuration plots views 1 and 4 at 0 degrees only; both files still write."""
    path = write_rows(tmp_path / "results.csv", design_rows((1, 4), (0.0,)))

    coverage_path, width_path = write_plots(path, tmp_path / "report")

    assert coverage_path.read_bytes().startswith(PNG_SIGNATURE)
    assert width_path.read_bytes().startswith(PNG_SIGNATURE)
    rows = read_result_rows(path)
    for panel in build_width_figure(rows).axes[:5]:
        assert [line.get_label() for line in series_lines(panel)] == ["0° placement noise"]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        # A percentage written where a fraction belongs would plot at 8900%.
        ("coverage", 89.0, "coverage must be a fraction between 0 and 1"),
        ("nominal_level", 90.0, "nominal_level must be a fraction strictly between 0 and 1"),
    ],
)
def test_refuses_a_fraction_column_written_as_a_percentage(
    tmp_path: Path, column: str, value: float, message: str
) -> None:
    """A coverage or nominal level given in percent instead of as a fraction is refused."""
    rows = design_rows((1,), (0.0,))
    rows[0][column] = value
    path = write_rows(tmp_path / "results.csv", rows)

    with pytest.raises(ValueError, match=message):
        read_result_rows(path)


def test_refuses_a_results_file_that_mixes_two_seeds(tmp_path: Path) -> None:
    """Rows from two runs cannot share one figure, so a file that mixes seeds is refused."""
    rows = design_rows((1, 4), (0.0,))
    rows[-1]["seed"] = 7
    path = write_rows(tmp_path / "results.csv", rows)

    with pytest.raises(ValueError, match="column seed differs between rows"):
        read_result_rows(path)
