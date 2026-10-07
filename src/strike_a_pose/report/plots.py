"""Coverage and median-width plots against view count, one subplot per measurement (FR-016).

``write_plots`` reads ``evaluate/results.csv`` (contracts/artifacts.md) and writes two PNG files
into ``report/``: ``coverage_vs_views.png`` with the empirical coverage in percent, and
``width_vs_views.png`` with the median interval width in centimetres. Each figure has one subplot
per measurement, one line per placement noise level, and one legend. The coverage figure shades
the 87% to 93% tolerance band of FR-013 and marks the nominal level, so each cell can be read
against the band. The figures are drawn on the Agg canvas (``FigureCanvasAgg`` of Matplotlib), so
no display is needed. The colours are the colour-blind-safe Okabe-Ito palette (Okabe and Ito,
Color Universal Design, https://jfly.uni-koeln.de/color/). Each PNG reaches its final name through
``checkpoint.atomic_path``, so an interrupted write leaves no partial figure under a plot name.
"""

import csv
import itertools
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from strike_a_pose.checkpoint import atomic_path

__all__ = [
    "COVERAGE_PLOT_NAME",
    "WIDTH_PLOT_NAME",
    "ResultRow",
    "build_coverage_figure",
    "build_width_figure",
    "read_result_rows",
    "write_plots",
]

# The two plot files of contracts/artifacts.md, under report/.
COVERAGE_PLOT_NAME = "coverage_vs_views.png"
WIDTH_PLOT_NAME = "width_vs_views.png"

# The tolerance band of FR-013 (spec.md Assumptions): 87% to 93% coverage at 90% nominal.
_BAND_LOW_PERCENT = 87.0
_BAND_HIGH_PERCENT = 93.0
_BAND_COLOUR = "#D9D9D9"  # light grey, so the band reads as background

# Measurements in the order of configuration key evaluate.measurements. Any other measurement
# name follows, in the order the file lists it.
_MEASUREMENT_ORDER = ("height", "chest", "waist", "hip", "thigh")

# Okabe-Ito colours for the 0, 2, and 5 degree series. Other noise levels take the spare colours
# in order, and none of the spare colours equals one of the three.
_NOISE_COLOURS = {0.0: "#0072B2", 2.0: "#E69F00", 5.0: "#009E73"}
_SPARE_COLOURS = ("#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")
# Distinct marker shapes, so the series stay apart when printed in greyscale.
_MARKERS = ("o", "s", "^", "D", "v")

_PANEL_COLUMNS = 3
_PANEL_WIDTH_INCHES = 4.0
_PANEL_HEIGHT_INCHES = 3.4
_LEGEND_ROW_EXTRA_INCHES = 0.6
_LINE_WIDTH = 1.8
_MARKER_SIZE = 6.0
_PNG_DPI = 150

# Every column the plots read from evaluate/results.csv (contracts/artifacts.md).
_REQUIRED_COLUMNS = (
    "cell_id",
    "views",
    "noise_deg",
    "measurement",
    "nominal_level",
    "n_cal",
    "n_test",
    "coverage",
    "median_width_cm",
    "seed",
    "config_hash",
)
# Fields that every row of one results file must share, because a figure states them once.
_PROVENANCE_FIELDS = ("nominal_level", "n_cal", "n_test", "seed", "config_hash")


@dataclass(frozen=True)
class ResultRow:
    """One row of evaluate/results.csv: one cell (view count and noise level) and one measurement.

    coverage is a fraction between 0 and 1, as data-model.md defines ResultCell. The figures show
    it in percent.
    """

    cell_id: str
    views: int
    noise_deg: float
    measurement: str
    nominal_level: float
    n_cal: int
    n_test: int
    coverage: float
    median_width_cm: float
    seed: int
    config_hash: str


@dataclass(frozen=True)
class _SeriesStyle:
    """The colour, marker, and legend label of one placement noise level."""

    colour: str
    marker: str
    label: str


def read_result_rows(path: str | os.PathLike[str]) -> list[ResultRow]:
    """Read evaluate/results.csv and return its rows, in file order.

    The file must carry every column the plots use, must hold at least one row, and must have
    the same provenance (nominal level, counts, seed, and configuration hash) on every row, so a
    figure never mixes two runs. A coverage outside [0, 1], such as a percentage written by
    mistake, and a non-finite or negative width are refused with a message that names the line.
    """
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in _REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{source}: missing column(s) {', '.join(missing)}")
        rows = [_parse_row(record, source, reader.line_num) for record in reader]
    if not rows:
        raise ValueError(f"{source}: no result rows to plot")
    for field in _PROVENANCE_FIELDS:
        if len({getattr(row, field) for row in rows}) > 1:
            raise ValueError(f"{source}: column {field} differs between rows; the file mixes runs")
    return rows


def _parse_row(record: Mapping[str, str | None], source: Path, line: int) -> ResultRow:
    """Convert one CSV record to a ResultRow, refusing a short row, a non-number, or a bad range."""
    values: dict[str, str] = {}
    for name in _REQUIRED_COLUMNS:
        value = record.get(name)
        if value is None:
            raise ValueError(f"{source}, line {line}: the row has fewer values than the header")
        values[name] = value
    try:
        row = ResultRow(
            cell_id=values["cell_id"],
            views=int(values["views"]),
            noise_deg=float(values["noise_deg"]),
            measurement=values["measurement"],
            nominal_level=float(values["nominal_level"]),
            n_cal=int(values["n_cal"]),
            n_test=int(values["n_test"]),
            coverage=float(values["coverage"]),
            median_width_cm=float(values["median_width_cm"]),
            seed=int(values["seed"]),
            config_hash=values["config_hash"],
        )
    except ValueError as error:
        raise ValueError(
            f"{source}, line {line}: a numeric column does not hold a number"
        ) from error
    if not 0.0 <= row.coverage <= 1.0:
        raise ValueError(
            f"{source}, line {line}: coverage must be a fraction between 0 and 1; "
            f"got {row.coverage!r}"
        )
    if not 0.0 < row.nominal_level < 1.0:
        raise ValueError(
            f"{source}, line {line}: nominal_level must be a fraction strictly between 0 and 1; "
            f"got {row.nominal_level!r}"
        )
    if not math.isfinite(row.median_width_cm) or row.median_width_cm < 0.0:
        raise ValueError(
            f"{source}, line {line}: median_width_cm must be a finite width of zero or more; "
            f"got {row.median_width_cm!r}"
        )
    return row


def build_coverage_figure(rows: Sequence[ResultRow]) -> Figure:
    """Return the coverage figure: empirical coverage in percent against view count.

    Each subplot shades the 87% to 93% band and draws a dashed line at the nominal level. The
    subplots share one vertical scale, so the cells compare directly.
    """
    _require_rows(rows)
    measurements = _measurement_names(rows)
    views = _view_counts(rows)
    styles = _series_styles(rows)
    nominal = 100.0 * rows[0].nominal_level
    coverages = [100.0 * row.coverage for row in rows]
    y_low = max(0.0, min(_BAND_LOW_PERCENT, min(coverages)) - 2.0)
    y_high = min(100.0, max(_BAND_HIGH_PERCENT, max(coverages)) + 2.0)

    figure, axes = _new_figure(len(measurements), share_y=True)
    for axis, measurement in zip(axes[: len(measurements)], measurements, strict=True):
        axis.axhspan(
            _BAND_LOW_PERCENT, _BAND_HIGH_PERCENT, color=_BAND_COLOUR, linewidth=0.0, zorder=0
        )
        axis.axhline(nominal, color="black", linestyle="--", linewidth=1.0, zorder=1)
        _draw_series(axis, rows, measurement, lambda row: 100.0 * row.coverage, styles)
        _format_panel(axis, measurement, views, "Coverage (%)")
        axis.set_ylim(y_low, y_high)
    handles = _series_handles(styles)
    handles.append(
        Patch(
            facecolor=_BAND_COLOUR,
            edgecolor="none",
            label=f"tolerance band {_BAND_LOW_PERCENT:g}% to {_BAND_HIGH_PERCENT:g}%",
        )
    )
    handles.append(
        Line2D([], [], color="black", linestyle="--", linewidth=1.0, label=f"nominal {nominal:g}%")
    )
    _fill_legend_cell(axes[len(measurements)], handles, rows[0])
    figure.suptitle(f"Empirical coverage at {nominal:g}% nominal by view count")
    return figure


def build_width_figure(rows: Sequence[ResultRow]) -> Figure:
    """Return the width figure: median interval width in centimetres against view count.

    The width axis starts at zero, so the subplots show true ratios between view counts. Each
    subplot has its own vertical scale, because the measurements differ in size.
    """
    _require_rows(rows)
    measurements = _measurement_names(rows)
    views = _view_counts(rows)
    styles = _series_styles(rows)

    figure, axes = _new_figure(len(measurements), share_y=False)
    for axis, measurement in zip(axes[: len(measurements)], measurements, strict=True):
        _draw_series(axis, rows, measurement, lambda row: row.median_width_cm, styles)
        _format_panel(axis, measurement, views, "Median interval width (cm)")
        axis.set_ylim(bottom=0.0)
    _fill_legend_cell(axes[len(measurements)], _series_handles(styles), rows[0])
    figure.suptitle("Median interval width by view count")
    return figure


def write_plots(
    results_path: str | os.PathLike[str], report_directory: str | os.PathLike[str]
) -> tuple[Path, Path]:
    """Read the results table and write both figures into report_directory.

    Returns the paths of coverage_vs_views.png and width_vs_views.png, in that order. The
    directory is created when it is missing.
    """
    rows = read_result_rows(results_path)
    directory = Path(report_directory)
    coverage_path = _write_png(build_coverage_figure(rows), directory / COVERAGE_PLOT_NAME)
    width_path = _write_png(build_width_figure(rows), directory / WIDTH_PLOT_NAME)
    return coverage_path, width_path


def _write_png(figure: Figure, destination: Path) -> Path:
    """Save the figure as PNG at destination, through a temporary name renamed at the end."""
    with atomic_path(destination) as temporary:
        figure.savefig(temporary, format="png", dpi=_PNG_DPI)
    return destination


def _require_rows(rows: Sequence[ResultRow]) -> None:
    """Refuse an empty table, because a figure needs at least one cell."""
    if not rows:
        raise ValueError("no result rows to plot")


def _measurement_names(rows: Sequence[ResultRow]) -> list[str]:
    """Return the measurement names in plot order: the configured order first, then the rest."""
    seen = list(dict.fromkeys(row.measurement for row in rows))
    known = [name for name in _MEASUREMENT_ORDER if name in seen]
    return known + [name for name in seen if name not in _MEASUREMENT_ORDER]


def _view_counts(rows: Sequence[ResultRow]) -> list[int]:
    """Return the distinct view counts in ascending order."""
    return sorted({row.views for row in rows})


def _series_styles(rows: Sequence[ResultRow]) -> dict[float, _SeriesStyle]:
    """Return a style per placement noise level, in ascending order of the noise level."""
    spare = itertools.cycle(_SPARE_COLOURS)
    styles: dict[float, _SeriesStyle] = {}
    for index, noise in enumerate(sorted({row.noise_deg for row in rows})):
        styles[noise] = _SeriesStyle(
            colour=_NOISE_COLOURS.get(noise) or next(spare),
            marker=_MARKERS[index % len(_MARKERS)],
            label=f"{noise:g}° placement noise",
        )
    return styles


def _series_handles(styles: Mapping[float, _SeriesStyle]) -> list[Line2D]:
    """Return one legend entry per noise level, drawn like its series."""
    return [
        Line2D(
            [],
            [],
            color=style.colour,
            marker=style.marker,
            linewidth=_LINE_WIDTH,
            markersize=_MARKER_SIZE,
            label=style.label,
        )
        for style in styles.values()
    ]


def _new_figure(panel_count: int, share_y: bool) -> tuple[Figure, list[Axes]]:
    """Return an Agg figure with a panel per measurement and one spare cell that holds the legend.

    The grid has three columns. The spare cell follows the panels, and any cells after it are
    hidden.
    """
    grid_rows = math.ceil((panel_count + 1) / _PANEL_COLUMNS)
    figure = Figure(
        figsize=(
            _PANEL_WIDTH_INCHES * _PANEL_COLUMNS,
            _PANEL_HEIGHT_INCHES * grid_rows + _LEGEND_ROW_EXTRA_INCHES,
        ),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    grid = figure.subplots(grid_rows, _PANEL_COLUMNS, sharey=share_y, squeeze=False)
    axes = list(grid.flat)
    for hidden in axes[panel_count + 1 :]:
        hidden.set_visible(False)
    return figure, axes


def _draw_series(
    axis: Axes,
    rows: Sequence[ResultRow],
    measurement: str,
    value_of: Callable[[ResultRow], float],
    styles: Mapping[float, _SeriesStyle],
) -> None:
    """Draw one line per noise level for one measurement, with the view counts on the x axis."""
    for noise, style in styles.items():
        points = sorted(
            (row.views, value_of(row))
            for row in rows
            if row.measurement == measurement and row.noise_deg == noise
        )
        if not points:
            continue
        views, values = zip(*points, strict=True)
        axis.plot(
            views,
            values,
            color=style.colour,
            marker=style.marker,
            linewidth=_LINE_WIDTH,
            markersize=_MARKER_SIZE,
            label=style.label,
        )


def _format_panel(axis: Axes, measurement: str, views: Sequence[int], y_label: str) -> None:
    """Set the title, the axis labels with units, and the view-count ticks of one panel."""
    axis.set_title(measurement.capitalize())
    axis.set_xlabel("View count (cameras)")
    axis.set_ylabel(y_label)
    axis.set_xticks(list(views))
    axis.set_xlim(views[0] - 0.4, views[-1] + 0.4)
    axis.grid(True, linewidth=0.5, color="0.85")


def _fill_legend_cell(axis: Axes, handles: Sequence[Line2D | Patch], shared_row: ResultRow) -> None:
    """Put the figure's only legend in the spare cell, with the sample sizes and seed beneath it.

    The provenance lines state the nominal level, the calibration and test counts, the seed, and
    the first twelve characters of the configuration hash, as constitution Principle III asks of
    every results table. Every row carries the same values, which read_result_rows checks.
    """
    axis.set_axis_off()
    axis.legend(handles=list(handles), loc="center", frameon=False, fontsize=10)
    provenance = (
        f"nominal {100.0 * shared_row.nominal_level:g}%, "
        f"n_cal {shared_row.n_cal}, n_test {shared_row.n_test}\n"
        f"seed {shared_row.seed}, config {shared_row.config_hash[:12]}"
    )
    axis.text(
        0.0,
        0.0,
        provenance,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.25",
    )
