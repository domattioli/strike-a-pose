"""Contiguous train, calibration, and test ranges of body ids, and the loss-monitoring slice.

Body ids 0 to n_train - 1 are training bodies. The next n_cal ids are calibration bodies, which
split conformal prediction needs disjoint from the training and test bodies (research R8). The
last n_test ids are test bodies. The ranges are written into the manifest, and ``check_disjoint``
checks that they cover every body id exactly once (research R9).

The last n_monitor training ids form a loss-monitoring slice. They keep the split ``train`` in the
manifest, but the training sampler draws only from ``sampler_range``, so they take no gradient
step, and no model selection reads them (research R9). This module implements no published method,
so it cites no algorithm source.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

__all__ = ["MONITOR_PERCENT", "DataSplit", "Split", "check_disjoint"]

# The loss-monitoring slice is this many percent of the training range, rounded up (research R9).
MONITOR_PERCENT = 5


class Split(str, Enum):
    """The split of a body. The values are the split names of the manifest and the predict files."""

    TRAIN = "train"
    CALIBRATION = "cal"
    TEST = "test"


@dataclass(frozen=True)
class DataSplit:
    """The sizes of the three splits, and the body ids that each one holds.

    Training bodies come first, then calibration bodies, then test bodies. The sizes are checked
    and stored as plain ints. A calibration or test split may be empty here; calibration itself
    refuses a calibration split below calibrate.min_cal (FR-011).
    """

    n_train: int
    n_cal: int
    n_test: int

    def __post_init__(self) -> None:
        for name in ("n_train", "n_cal", "n_test"):
            object.__setattr__(self, name, _checked_count(getattr(self, name), name))
        if self.n_train < 2:
            raise ValueError(
                f"n_train must be at least 2, not {self.n_train}, so that the loss-monitoring "
                "slice leaves at least one training body"
            )
        check_disjoint(self.ranges())

    @property
    def n_bodies(self) -> int:
        """The number of bodies in all three splits together."""
        return self.n_train + self.n_cal + self.n_test

    @property
    def n_monitor(self) -> int:
        """The size of the loss-monitoring slice: 5 percent of n_train, rounded up."""
        # Integer arithmetic gives the exact ceiling. -(-a // b) equals ceil(a / b).
        return -((-MONITOR_PERCENT * self.n_train) // 100)

    @property
    def sampler_range(self) -> range:
        """The training body ids that the training sampler may yield: [0, n_train - n_monitor)."""
        return range(0, self.n_train - self.n_monitor)

    @property
    def monitor_range(self) -> range:
        """The loss-monitoring slice: the last n_monitor training ids, with no gradient step."""
        return range(self.n_train - self.n_monitor, self.n_train)

    def ranges(self) -> dict[Split, range]:
        """Return the body ids of each split, in the order train, calibration, test."""
        calibration_start = self.n_train
        test_start = self.n_train + self.n_cal
        return {
            Split.TRAIN: range(0, calibration_start),
            Split.CALIBRATION: range(calibration_start, test_start),
            Split.TEST: range(test_start, self.n_bodies),
        }

    def range_of(self, split: Split | str) -> range:
        """Return the body ids of one split, given as a Split or as its name, such as "cal"."""
        return self.ranges()[Split(split)]

    def split_of(self, body_id: int) -> Split:
        """Return the split that holds body_id. Raise ValueError for an id outside the bodies."""
        if isinstance(body_id, bool) or not isinstance(body_id, Integral):
            raise TypeError(f"body_id must be an integer, not {type(body_id).__name__}")
        if not 0 <= body_id < self.n_bodies:
            raise ValueError(f"body_id {body_id} is outside 0 to {self.n_bodies - 1}")
        if body_id < self.n_train:
            return Split.TRAIN
        if body_id < self.n_train + self.n_cal:
            return Split.CALIBRATION
        return Split.TEST


def check_disjoint(ranges: Mapping[Split, range]) -> None:
    """Raise ValueError unless the split ranges tile the body ids from 0 with no overlap.

    Each range must be contiguous (step 1). Together the non-empty ranges must cover every body id
    from 0 up exactly once, so an overlap, a gap, or a first body id above 0 is an error. An empty
    range is allowed.
    """
    spans = []
    for split, body_ids in ranges.items():
        if body_ids.step != 1:
            raise ValueError(f"the {split.value} split is not a contiguous range of body ids")
        if len(body_ids) > 0:
            spans.append((body_ids.start, body_ids.stop, split))
    # Sorted by start, each span must begin exactly where the span before it ended.
    spans.sort(key=lambda span: span[0])
    covered_to = 0  # the first body id that no span so far covers
    covered_by: Split | None = None  # the split that holds body id covered_to - 1
    for start, stop, split in spans:
        if start < 0:
            raise ValueError(f"body ids start at 0, but the {split.value} split starts at {start}")
        if start > covered_to:
            raise ValueError(f"body ids {covered_to} to {start - 1} belong to no split")
        if start < covered_to:
            raise ValueError(
                f"the {split.value} split starts at body id {start}, but the "
                f"{covered_by.value} split already holds body id {covered_to - 1}"
            )
        covered_to = stop
        covered_by = split


def _checked_count(value: object, name: str) -> int:
    """Return value as an int, after checking that it is a non-negative integer and not a bool."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, not {value}")
    return int(value)
