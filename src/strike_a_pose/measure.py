"""Batched body measurements: height and the chest, waist, hip, and thigh circumferences in cm.

``measure_batch`` implements the five measurement definitions of research R6 (FR-004) for many
meshes at once, in torch on any device. Every mesh is in the canonical pose of its shape, so a
measurement depends on body shape only. Lengths go in as metres and come out as centimetres. The
vertical axis is y, as in ``body/base.py``.

Definitions (research R6). ``P(h, S)`` is the perimeter of the 2D convex hull of the points where
the horizontal plane ``y = h`` cuts the triangles that have at least one vertex in the part set
``S``. The torso is the pelvis, spine1, spine2, and spine3 parts. A part is the set of vertices
that one joint owns (``part_ids``), and ``y(j)`` is the height of joint ``j``.

    height   max y - min y over all vertices
    chest    max of P(h, torso) for h from y(spine2) up to y(spine3)
    waist    min of P(h, torso) for h from y(pelvis) up to y(spine2)
    hip      max of P(h, torso + left hip + right hip) for h from y(pelvis) - 0.10 height up to
             y(pelvis)
    thigh    P(h, left hip) at h = y(left_hip) - 0.30 (y(left_hip) - y(left_knee))

A search tries h = low, low + step, low + 2 step, and so on, up to and including high when high
lies on that grid (within one nanometre). The step is ``measure.step_cm``, the same for ground truth
and predictions. The convex hull is used because a tape measure spans concavities.

Plane test. The test is half-open: a vertex is above the plane when ``y > h`` and below it
otherwise, and an edge is cut only when its two vertices fall on different sides. A vertex that
sits exactly on the plane (vertex rings at joint heights do) therefore never adds a duplicate
point and never drops one. The cut points of a part set are the cut points of the unique edges of
its triangles; they are the end points of the cut segments of those triangles, listed once even
where two triangles share an edge.

NaN rule (research R6). A slice with fewer than 3 cut points (counted once per cut edge) is
degenerate. A measurement is NaN when its search range holds no slice or when any slice of the
range is degenerate, and the body is then flagged. ``slice_nan_flags`` finds those rows.

Perimeter. Cauchy's surface area formula, in the plane, says that the perimeter of a convex region
is the integral over the directions of a line, from 0 to pi, of the length of the region's
orthogonal projection onto that line (L. A. Santalo, "Integral Geometry and Geometric Probability",
Addison-Wesley, 1976). The projection length in a direction is the largest minus the smallest
value of ``x cos(a) + z sin(a)`` over the cut points, so the hull itself is never built. The
integral becomes a sum over 64 equally spaced directions (the rectangle rule). For a convex polygon
the projection length is the sum over its edges of (edge length / 2) |cos(a - edge angle)|, and by
the Fourier series of |cos| the rectangle rule misses each such term by between -0.020 and +0.010
percent of its integral, depending on where the edge angle falls among the sampled directions. The
perimeter is therefore off by at most about ``(pi / 64)^2 / 12``, which is 0.020 percent and below
the 0.1 percent of research R6. ``tests/test_measure.py`` reaches this bound with rectangles whose
sides lie along the sampled directions and stays under it for polygons and ellipses.

Memory and speed. The cuts of all meshes in a chunk and all slice heights are found with one
broadcast comparison of edge heights against slice heights. The cut points are then packed into one
padded row per slice, so every direction costs a few dense passes and no atomic operation.
``memory_budget_mb`` bounds one chunk: a larger batch is split into chunks of meshes, a search
with very many slices into blocks of slices, and the rows of cut points into blocks of rows.

Arithmetic. Everything is float64 elementwise arithmetic in a fixed order: no matrix product, no
fused multiply-add kernel, and no atomic or parallel floating-point sum (the sum over directions
is a fixed halving tree). The result for a mesh therefore does not depend on the batch size, the
chunk size, or the thread count. Gradients are not tracked.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.body.base import JOINT_INDEX, NUM_JOINTS

__all__ = [
    "DEFAULT_MEMORY_BUDGET_MB",
    "MEASUREMENT_NAMES",
    "NUM_DIRECTIONS",
    "NUM_MEASUREMENTS",
    "measure_batch",
    "measure_mesh",
    "slice_nan_flags",
]

# The five measurements in column order, as in ``evaluate.measurements`` (contracts/config.md).
MEASUREMENT_NAMES: tuple[str, ...] = ("height", "chest", "waist", "hip", "thigh")
NUM_MEASUREMENTS: int = len(MEASUREMENT_NAMES)

# Number of directions in the Cauchy sum, spaced pi / NUM_DIRECTIONS apart over half a turn. It
# must be a power of two (see _sum_in_fixed_order).
NUM_DIRECTIONS: int = 64

# Approximate working memory of one chunk, in megabytes (see ``measure_batch``).
DEFAULT_MEMORY_BUDGET_MB: float = 256.0

_METRES_TO_CENTIMETRES = 100.0

# Research R6: the hip search starts this fraction of the height below the pelvis joint, and the
# thigh slice lies this fraction of the hip-to-knee distance below the hip joint.
_HIP_RANGE_HEIGHT_FRACTION = 0.10
_THIGH_DROP_FRACTION = 0.30

# A slice needs at least this many cut points to have a perimeter.
_MINIMUM_SLICE_POINTS = 3

# A search range that ends within this distance (metres) above the last grid slice includes it.
_GRID_END_TOLERANCE_M = 1e-9

# Largest number of slices that one search may hold; a larger count is a unit mistake.
_MAXIMUM_SLICES = 1_000_000

# Memory accounting. One entry of the cut mask costs about 8 bytes once the cut points that come
# from it are counted. A padded entry costs 8 bytes for each of the two coordinate arrays and the
# three temporaries of one direction. The first chunk is sized for _NOMINAL_SLICES slices per
# search; a search with more slices is split into blocks.
_BYTES_PER_MASK_ENTRY = 8
_BYTES_PER_PADDED_ENTRY = 8 * 5
_NOMINAL_SLICES = 64

_PELVIS = JOINT_INDEX["pelvis"]
_SPINE2 = JOINT_INDEX["spine2"]
_SPINE3 = JOINT_INDEX["spine3"]
_LEFT_HIP = JOINT_INDEX["left_hip"]
_LEFT_KNEE = JOINT_INDEX["left_knee"]

# Part sets of research R6, as joint indices. The collar parts are left out of the torso on
# purpose: their vertices sit at shoulder level and would move the chest maximum to the shoulders.
_TORSO_PARTS = tuple(JOINT_INDEX[name] for name in ("pelvis", "spine1", "spine2", "spine3"))
_HIP_PARTS = _TORSO_PARTS + tuple(JOINT_INDEX[name] for name in ("left_hip", "right_hip"))
_THIGH_PARTS = (_LEFT_HIP,)


def _snap_to_zero(value: float) -> float:
    """Replace a value that is zero up to rounding error by an exact zero."""
    return 0.0 if abs(value) < 1e-12 else value


def _direction_components() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the cosines and sines of the NUM_DIRECTIONS angles k pi / NUM_DIRECTIONS."""
    angles = [math.pi * index / NUM_DIRECTIONS for index in range(NUM_DIRECTIONS)]
    return (
        tuple(_snap_to_zero(math.cos(angle)) for angle in angles),
        tuple(_snap_to_zero(math.sin(angle)) for angle in angles),
    )


_COSINES, _SINES = _direction_components()


@dataclass(frozen=True, eq=False)
class _Search:
    """One slice search: the unique edges of the triangles of a part set, and max or min."""

    edge_first: torch.Tensor  # (E,) vertex index of one end of each edge
    edge_second: torch.Tensor  # (E,) vertex index of the other end
    takes_maximum: bool


def _unique_edges(
    faces: NDArray[np.int64], part_ids: NDArray[np.int64], parts: tuple[int, ...], vertex_count: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the unique edges of the triangles that have a vertex in the part set, as vertex pairs.

    Each edge is listed once, from its lower vertex index to its higher one, in sorted order.
    """
    owned = np.isin(part_ids, parts)
    selected = faces[owned[faces].any(axis=1)]
    pairs = np.concatenate([selected[:, [0, 1]], selected[:, [1, 2]], selected[:, [2, 0]]])
    lower = pairs.min(axis=1)
    higher = pairs.max(axis=1)
    keys = np.unique((lower * vertex_count + higher)[lower != higher])
    return keys // vertex_count, keys % vertex_count


def _slice_counts(low: torch.Tensor, high: torch.Tensor, step: float) -> torch.Tensor:
    """Return the number of grid slices low, low + step, ... that do not pass high, shape (B,).

    The count is 0 when high is below low or when either end is not finite.
    """
    count = torch.floor((high - low + _GRID_END_TOLERANCE_M) / step) + 1.0
    count = torch.where(torch.isfinite(count), count, torch.zeros_like(count))
    return count.clamp(min=0.0, max=float(_MAXIMUM_SLICES + 1)).to(torch.int64)


def _sum_in_fixed_order(values: torch.Tensor) -> torch.Tensor:
    """Sum over the last axis by repeated halving, so the order of additions never varies.

    The length of the last axis must be a power of two, as NUM_DIRECTIONS is.
    """
    while values.shape[-1] > 1:
        half = values.shape[-1] // 2
        values = values[..., :half] + values[..., half:]
    return values[..., 0]


def _cauchy_perimeters(
    x: torch.Tensor,
    z: torch.Tensor,
    pair_row: torch.Tensor,
    counts: torch.Tensor,
    budget_bytes: int,
) -> torch.Tensor:
    """Return the Cauchy-formula perimeter of the hull of each row's cut points; NaN below 3 points.

    ``x`` and ``z`` hold the cut points of all rows one after another; ``pair_row`` is the row of
    each point (non-decreasing) and ``counts`` the number of points per row. Each row is padded to
    the same width with copies of its first point, which changes no maximum or minimum. Blocks of
    rows are padded and projected one after another, so the padded array stays inside the budget.
    """
    rows = counts.shape[0]
    perimeters = torch.full((rows,), math.nan, dtype=torch.float64, device=x.device)
    if x.shape[0] == 0:
        return perimeters
    starts = torch.cumsum(counts, dim=0) - counts
    widest = int(counts.max())
    rows_per_block = max(1, budget_bytes // _BYTES_PER_PADDED_ENTRY // widest)
    scale = math.pi / NUM_DIRECTIONS
    for first_row in range(0, rows, rows_per_block):
        last_row = min(rows, first_row + rows_per_block)
        block_counts = counts[first_row:last_row]
        if int(block_counts.max()) < _MINIMUM_SLICE_POINTS:
            continue
        begin = int(starts[first_row])
        end = int(starts[last_row - 1] + counts[last_row - 1])
        block_x = x[begin:end]
        block_z = z[begin:end]
        block_starts = starts[first_row:last_row] - begin
        width = int(block_counts.max())
        local_row = pair_row[begin:end] - first_row
        rank = torch.arange(end - begin, device=x.device) - block_starts[local_row]
        first_point = block_starts.clamp(max=end - begin - 1)
        padded_x = block_x[first_point].unsqueeze(1).repeat(1, width)
        padded_z = block_z[first_point].unsqueeze(1).repeat(1, width)
        padded_x[local_row, rank] = block_x
        padded_z[local_row, rank] = block_z
        largest = []
        smallest = []
        for cosine, sine in zip(_COSINES, _SINES, strict=True):
            projection = padded_x * cosine
            projection += padded_z * sine
            largest.append(projection.amax(dim=1))
            smallest.append(projection.amin(dim=1))
        widths = torch.stack(largest, dim=1) - torch.stack(smallest, dim=1)
        block_perimeters = _sum_in_fixed_order(widths) * scale
        perimeters[first_row:last_row] = torch.where(
            block_counts >= _MINIMUM_SLICE_POINTS,
            block_perimeters,
            torch.full_like(block_perimeters, math.nan),
        )
    return perimeters


def _slice_perimeters(
    vertices: torch.Tensor, search: _Search, heights: torch.Tensor, budget_bytes: int
) -> torch.Tensor:
    """Return the hull perimeter in metres of every slice, shape (B, K); NaN for a degenerate slice.

    ``vertices`` has shape (B, V, 3) and ``heights`` shape (B, K); a height of +inf is no slice.
    """
    batch, slices = heights.shape
    heights_of_vertices = vertices[:, :, 1]
    first_height = heights_of_vertices[:, search.edge_first]
    second_height = heights_of_vertices[:, search.edge_second]
    lowest = torch.minimum(first_height, second_height)
    highest = torch.maximum(first_height, second_height)

    # The half-open plane test: an edge is cut by the plane at h when lowest <= h < highest.
    level = heights.unsqueeze(2)
    is_cut = lowest.unsqueeze(1) <= level
    is_cut &= level < highest.unsqueeze(1)
    is_cut = is_cut.reshape(batch * slices, -1)
    counts = is_cut.sum(dim=1)
    pair_row, pair_edge = torch.nonzero(is_cut, as_tuple=True)
    del is_cut

    # Where each cut edge meets its plane, found by interpolating from the lower end upward.
    pair_mesh = pair_row // slices
    start = vertices[pair_mesh, search.edge_first[pair_edge]]
    stop = vertices[pair_mesh, search.edge_second[pair_edge]]
    swapped = (start[:, 1] > stop[:, 1]).unsqueeze(1)
    below = torch.where(swapped, stop, start)
    above = torch.where(swapped, start, stop)
    fraction = (heights.reshape(-1)[pair_row] - below[:, 1]) / (above[:, 1] - below[:, 1])
    cut_x = below[:, 0] + fraction * (above[:, 0] - below[:, 0])
    cut_z = below[:, 2] + fraction * (above[:, 2] - below[:, 2])
    perimeters = _cauchy_perimeters(cut_x, cut_z, pair_row, counts, budget_bytes)
    return perimeters.reshape(batch, slices)


def _search_extremum(
    vertices: torch.Tensor,
    search: _Search,
    low: torch.Tensor,
    high: torch.Tensor,
    step: float,
    budget_bytes: int,
) -> torch.Tensor:
    """Return the largest or smallest slice perimeter over each mesh's grid, in metres, shape (B,).

    The result is NaN for a mesh whose range holds no slice or has any degenerate slice.
    """
    batch = vertices.shape[0]
    device = vertices.device
    slice_counts = _slice_counts(low, high, step)
    total_slices = int(slice_counts.max())
    nan = torch.full((batch,), math.nan, dtype=torch.float64, device=device)
    edge_count = search.edge_first.shape[0]
    if edge_count == 0 or total_slices == 0:
        return nan
    if total_slices > _MAXIMUM_SLICES:
        raise ValueError(
            f"a search range holds more than {_MAXIMUM_SLICES} slices at step_cm={step * 100:g}; "
            "the joints or vertices are probably not in metres"
        )
    mask_entries = max(1, budget_bytes // _BYTES_PER_MASK_ENTRY)
    block = max(1, min(total_slices, mask_entries // (batch * edge_count)))
    worst = -math.inf if search.takes_maximum else math.inf
    extremum = torch.full((batch,), worst, dtype=torch.float64, device=device)
    degenerate = torch.zeros(batch, dtype=torch.bool, device=device)
    for first_slice in range(0, total_slices, block):
        indices = torch.arange(
            first_slice, min(total_slices, first_slice + block), device=device
        ).unsqueeze(0)
        active = indices < slice_counts.unsqueeze(1)
        grid = low.unsqueeze(1) + indices.to(torch.float64) * step
        heights = torch.where(active, grid, torch.full_like(grid, math.inf))
        perimeters = _slice_perimeters(vertices, search, heights, budget_bytes)
        missing = torch.isnan(perimeters)
        degenerate |= (active & missing).any(dim=1)
        usable = torch.where(active & ~missing, perimeters, torch.full_like(perimeters, worst))
        if search.takes_maximum:
            extremum = torch.maximum(extremum, usable.amax(dim=1))
        else:
            extremum = torch.minimum(extremum, usable.amin(dim=1))
    return torch.where(degenerate | (slice_counts == 0), nan, extremum)


def _measure_chunk(
    mesh: torch.Tensor,
    joints: torch.Tensor,
    searches: dict[str, _Search],
    step: float,
    budget_bytes: int,
) -> torch.Tensor:
    """Apply the five definitions of research R6 to a chunk; return centimetres, shape (B, 5).

    ``mesh`` has shape (B, V, 3) and ``joints`` shape (B, J, 3), both float64 and in metres.
    """
    vertex_height = mesh[:, :, 1]
    height = vertex_height.amax(dim=1) - vertex_height.amin(dim=1)
    joint_height = joints[:, :, 1]
    pelvis = joint_height[:, _PELVIS]
    spine2 = joint_height[:, _SPINE2]
    spine3 = joint_height[:, _SPINE3]
    left_hip = joint_height[:, _LEFT_HIP]
    left_knee = joint_height[:, _LEFT_KNEE]
    thigh_level = left_hip - _THIGH_DROP_FRACTION * (left_hip - left_knee)
    hip_bottom = pelvis - _HIP_RANGE_HEIGHT_FRACTION * height
    values = [
        height,
        _search_extremum(mesh, searches["chest"], spine2, spine3, step, budget_bytes),
        _search_extremum(mesh, searches["waist"], pelvis, spine2, step, budget_bytes),
        _search_extremum(mesh, searches["hip"], hip_bottom, pelvis, step, budget_bytes),
        _search_extremum(mesh, searches["thigh"], thigh_level, thigh_level, step, budget_bytes),
    ]
    return torch.stack(values, dim=1) * _METRES_TO_CENTIMETRES


def _prepare_searches(
    faces: NDArray[np.int64],
    part_ids: NDArray[np.int64],
    vertex_count: int,
    device: torch.device,
) -> dict[str, _Search]:
    """Build the chest, waist, hip, and thigh searches of research R6 for one mesh topology."""
    definitions = {
        "chest": (_TORSO_PARTS, True),
        "waist": (_TORSO_PARTS, False),
        "hip": (_HIP_PARTS, True),
        "thigh": (_THIGH_PARTS, True),
    }
    edges_by_parts: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    searches = {}
    for name, (parts, takes_maximum) in definitions.items():
        if parts not in edges_by_parts:
            first, second = _unique_edges(faces, part_ids, parts, vertex_count)
            edges_by_parts[parts] = (
                torch.from_numpy(first).to(device),
                torch.from_numpy(second).to(device),
            )
        edge_first, edge_second = edges_by_parts[parts]
        searches[name] = _Search(edge_first, edge_second, takes_maximum)
    return searches


def _checked_inputs(
    vertices: ArrayLike | torch.Tensor,
    faces: ArrayLike | torch.Tensor,
    part_ids: ArrayLike | torch.Tensor,
    joints: ArrayLike | torch.Tensor,
    step_cm: float,
    memory_budget_mb: float,
) -> tuple[torch.Tensor, NDArray[np.int64], NDArray[np.int64], torch.Tensor, float, int]:
    """Convert and check the arguments of ``measure_batch``.

    Returns the vertices as a tensor (still on their own device and dtype), the faces and part ids
    as int64 arrays, the joints as a tensor, the step in metres, and the budget in bytes.
    """
    step = float(step_cm)
    if not (math.isfinite(step) and step > 0.0):
        raise ValueError(f"step_cm must be a positive finite number, got {step_cm!r}")
    budget = float(memory_budget_mb)
    if not (math.isfinite(budget) and budget > 0.0):
        raise ValueError(f"memory_budget_mb must be a positive finite number, got {budget!r}")

    vertex_tensor = _as_tensor(vertices)
    if vertex_tensor.ndim != 3 or vertex_tensor.shape[2] != 3 or vertex_tensor.shape[1] < 1:
        raise ValueError(
            "vertices must have shape (meshes, vertices, 3) with at least one vertex, "
            f"got {tuple(vertex_tensor.shape)}"
        )
    batch, vertex_count = vertex_tensor.shape[0], vertex_tensor.shape[1]

    joint_tensor = _as_tensor(joints)
    if (
        joint_tensor.ndim != 3
        or joint_tensor.shape[0] != batch
        or joint_tensor.shape[1] < NUM_JOINTS
        or joint_tensor.shape[2] != 3
    ):
        raise ValueError(
            f"joints must have shape ({batch}, at least {NUM_JOINTS}, 3) to match vertices, "
            f"got {tuple(joint_tensor.shape)}"
        )

    face_array = _as_index_array(faces, "faces")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError(f"faces must have shape (faces, 3), got {face_array.shape}")
    if face_array.size and (face_array.min() < 0 or face_array.max() >= vertex_count):
        raise ValueError(
            f"faces must index vertices 0 to {vertex_count - 1}, "
            f"got the range {face_array.min()} to {face_array.max()}"
        )
    part_array = _as_index_array(part_ids, "part_ids")
    if part_array.shape != (vertex_count,):
        raise ValueError(
            f"part_ids must have one entry per vertex, shape ({vertex_count},), "
            f"got {part_array.shape}"
        )
    return (
        vertex_tensor,
        face_array,
        part_array,
        joint_tensor,
        step / _METRES_TO_CENTIMETRES,
        max(1, int(budget * 2**20)),
    )


def _as_tensor(values: ArrayLike | torch.Tensor) -> torch.Tensor:
    """Return a tensor for a tensor, array, or nested list; lists of floats stay float64."""
    if isinstance(values, torch.Tensor):
        return values
    array = np.ascontiguousarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array)


def _as_index_array(values: ArrayLike | torch.Tensor, name: str) -> NDArray[np.int64]:
    """Convert integer indices from a list, array, or tensor into an int64 array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must hold integers, got dtype {array.dtype}")
    return array.astype(np.int64, copy=False)


@torch.no_grad()
def measure_batch(
    vertices: ArrayLike | torch.Tensor,
    faces: ArrayLike | torch.Tensor,
    part_ids: ArrayLike | torch.Tensor,
    joints: ArrayLike | torch.Tensor,
    step_cm: float,
    *,
    device: str | torch.device | None = None,
    memory_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB,
) -> torch.Tensor:
    """Measure B canonical-pose meshes and return height, chest, waist, hip, and thigh in cm.

    Args:
        vertices: B meshes of V vertices in metres, shape (B, V, 3), y up. A NumPy array, a nested
            list, or a tensor on any device; it is widened to float64 one chunk at a time.
        faces: triangle vertex indices shared by all meshes, shape (F, 3).
        part_ids: for each vertex the index in ``JOINT_NAMES`` of the joint that owns it, shape
            (V,) (``BodyModel.part_ids``).
        joints: the body joints of each mesh in metres, shape (B, J, 3) with J at least 22, in the
            order of ``JOINT_NAMES`` (``BodyModel.joints``). Joints after the 22nd are ignored.
        step_cm: the search step in centimetres (``measure.step_cm``). It must be positive.
        device: where to compute. By default the device of ``vertices``, which is the CPU for
            NumPy input. Inputs on another device are moved to it one chunk at a time.
        memory_budget_mb: approximate working memory of one chunk in megabytes, for body meshes
            whose slices cut a small share of the edges. A batch that needs more is split into
            chunks of meshes, which changes no result.

    Returns:
        A float64 tensor of shape (B, 5) on the compute device, with the columns of
        ``MEASUREMENT_NAMES``, in centimetres. An entry is NaN when its slice is degenerate (see
        the module docstring); ``slice_nan_flags`` finds those rows.

    Raises:
        ValueError: when an argument has the wrong shape or type, or ``step_cm`` is not positive.
    """
    (vertex_tensor, face_array, part_array, joint_tensor, step, budget_bytes) = _checked_inputs(
        vertices, faces, part_ids, joints, step_cm, memory_budget_mb
    )
    if device is None:
        compute_device = vertex_tensor.device
    else:
        compute_device = torch.device(device)
    batch, vertex_count = vertex_tensor.shape[0], vertex_tensor.shape[1]
    result = torch.empty((batch, NUM_MEASUREMENTS), dtype=torch.float64, device=compute_device)
    if batch == 0:
        return result
    searches = _prepare_searches(face_array, part_array, vertex_count, compute_device)
    joint_tensor = joint_tensor.to(device=compute_device, dtype=torch.float64)

    most_edges = max(search.edge_first.shape[0] for search in searches.values())
    mask_entries = max(1, budget_bytes // _BYTES_PER_MASK_ENTRY)
    chunk = max(1, min(batch, mask_entries // max(1, most_edges * _NOMINAL_SLICES)))
    for first in range(0, batch, chunk):
        last = min(batch, first + chunk)
        mesh = vertex_tensor[first:last].to(device=compute_device, dtype=torch.float64)
        result[first:last] = _measure_chunk(
            mesh, joint_tensor[first:last], searches, step, budget_bytes
        )
    return result


def measure_mesh(
    vertices: ArrayLike | torch.Tensor,
    faces: ArrayLike | torch.Tensor,
    part_ids: ArrayLike | torch.Tensor,
    joints: ArrayLike | torch.Tensor,
    step_cm: float,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Measure one canonical-pose mesh; return its five measurements in cm, shape (5,).

    ``vertices`` has shape (V, 3) and ``joints`` shape (J, 3); the other arguments are those of
    ``measure_batch``, which this calls with a batch of one.
    """
    batch_of_one = measure_batch(
        _as_tensor(vertices).unsqueeze(0),
        faces,
        part_ids,
        _as_tensor(joints).unsqueeze(0),
        step_cm,
        device=device,
    )
    return batch_of_one[0]


def slice_nan_flags(measurements: ArrayLike | torch.Tensor) -> torch.Tensor:
    """Flag the rows of a measurement array that hold a NaN or another non-finite value.

    A degenerate slice gives NaN (see the module docstring), and a body with such a row is
    flagged ``slice_nan`` (data model, BodySample). ``measurements`` has shape (..., 5); the result
    is a bool tensor of the shape without the last axis.
    """
    return ~torch.isfinite(_as_tensor(measurements)).all(dim=-1)
