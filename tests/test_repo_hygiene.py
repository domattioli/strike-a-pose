"""FR-022 hygiene: no licensed asset, derived render, per-subject table, or weight is tracked."""

import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

DENIED_SUFFIXES = frozenset({".npz", ".pkl", ".pt", ".pth", ".ckpt", ".h5", ".obj", ".ply"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg"})
CSV_FORBIDDEN_HEADER_TERMS = ("body_id", "subject_id", "pose_body_0")

# Decimal sizes: 1 MB is 1,000,000 bytes and 100 kB is 100,000 bytes, the stricter reading.
MAX_TRACKED_BYTES = 1_000_000
MAX_FIXTURE_BYTES = 100_000

# Index mode of a submodule. It records a commit of another repository, not a file.
SUBMODULE_MODE = "160000"


class TrackedFile(NamedTuple):
    """A git index entry: POSIX path from the repository root, size in bytes, and CSV header."""

    path: str
    size: int
    header: str = ""


class Violation(NamedTuple):
    """One tracked file that breaks one FR-022 rule, with the reason."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def _suffix(path: str) -> str:
    return PurePosixPath(path).suffix.lower()


def _in_results_dir(parts: tuple[str, ...]) -> bool:
    """True for specs/<feature>/results/<anything>, where the wildcard is one directory name."""
    return len(parts) >= 4 and parts[0] == "specs" and parts[2] == "results"


def _in_fixtures_dir(parts: tuple[str, ...]) -> bool:
    return len(parts) >= 3 and parts[0] == "tests" and parts[1] == "fixtures"


def csv_header(data: bytes) -> str:
    """Return the first line of CSV bytes, without a byte order mark or line ending."""
    text = data.decode("utf-8-sig", errors="replace")
    return text.split("\n", 1)[0].rstrip("\r")


def denied_extension_violations(files: Sequence[TrackedFile]) -> list[Violation]:
    """Flag every tracked file whose extension is on the deny list, in any directory."""
    return [
        Violation(f.path, f"{_suffix(f.path)} files are never tracked (FR-022)")
        for f in files
        if _suffix(f.path) in DENIED_SUFFIXES
    ]


def image_location_violations(files: Sequence[TrackedFile]) -> list[Violation]:
    """Flag every tracked png or jpg file outside specs/*/results/ and tests/fixtures/."""
    found = []
    for f in files:
        parts = PurePosixPath(f.path).parts
        allowed = _in_results_dir(parts) or _in_fixtures_dir(parts)
        if _suffix(f.path) in IMAGE_SUFFIXES and not allowed:
            found.append(
                Violation(
                    f.path,
                    "png and jpg files are tracked only under specs/*/results/ "
                    "and tests/fixtures/ (FR-022)",
                )
            )
    return found


def csv_header_violations(files: Sequence[TrackedFile]) -> list[Violation]:
    """Flag every tracked CSV whose header contains a per-body or per-subject column name."""
    found = []
    for f in files:
        if _suffix(f.path) != ".csv":
            continue
        header = f.header.lower()
        hits = [term for term in CSV_FORBIDDEN_HEADER_TERMS if term in header]
        if hits:
            found.append(Violation(f.path, f"CSV header contains {', '.join(hits)} (FR-022)"))
    return found


def size_violations(files: Sequence[TrackedFile]) -> list[Violation]:
    """Flag every tracked file over 1 MB."""
    return [
        Violation(f.path, f"{f.size} bytes is over the 1 MB limit (FR-022)")
        for f in files
        if f.size > MAX_TRACKED_BYTES
    ]


def fixture_size_violations(files: Sequence[TrackedFile]) -> list[Violation]:
    """Flag every tracked file under tests/fixtures/ that is over 100 kB."""
    return [
        Violation(
            f.path,
            f"{f.size} bytes is over the 100 kB fixture limit "
            "(constitution, Assets, Data, and Compute Constraints)",
        )
        for f in files
        if _in_fixtures_dir(PurePosixPath(f.path).parts) and f.size > MAX_FIXTURE_BYTES
    ]


RULES = (
    denied_extension_violations,
    image_location_violations,
    csv_header_violations,
    size_violations,
    fixture_size_violations,
)


def all_violations(files: Sequence[TrackedFile]) -> list[Violation]:
    """Apply every FR-022 rule and return the violations in rule order."""
    return [violation for rule in RULES for violation in rule(files)]


def _git(root: Path, *args: str, stdin: bytes = b"") -> bytes:
    """Run git in root and return its standard output; raise RuntimeError with git's message."""
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        input=stdin,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {message}")
    return completed.stdout


def _in_git_work_tree(root: Path) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def _blob_sizes(root: Path, blob_ids: Sequence[str]) -> dict[str, int]:
    """Return the byte size of each git blob, read from the object database."""
    if not blob_ids:
        return {}
    report = _git(root, "cat-file", "--batch-check", stdin="\n".join(blob_ids).encode() + b"\n")
    sizes: dict[str, int] = {}
    for blob_id, line in zip(blob_ids, report.decode("ascii").splitlines(), strict=True):
        fields = line.split()
        if len(fields) != 3:
            raise RuntimeError(f"git cannot read object {blob_id}: {line}")
        sizes[blob_id] = int(fields[2])
    return sizes


def list_tracked_files(root: Path = REPO_ROOT) -> list[TrackedFile]:
    """List the files in the git index of root: the set that version control holds."""
    blob_of_path: dict[str, str] = {}
    for record in _git(root, "ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        meta, _, raw_path = record.partition(b"\t")
        mode, blob_id, _stage = meta.decode("ascii").split(" ")
        if mode == SUBMODULE_MODE:
            continue
        blob_of_path.setdefault(raw_path.decode("utf-8", errors="replace"), blob_id)
    sizes = _blob_sizes(root, sorted(set(blob_of_path.values())))
    files = []
    for path in sorted(blob_of_path):
        blob_id = blob_of_path[path]
        header = ""
        if _suffix(path) == ".csv":
            header = csv_header(_git(root, "cat-file", "blob", blob_id))
        files.append(TrackedFile(path, sizes[blob_id], header))
    return files


def _report(violations: Sequence[Violation]) -> str:
    return "\n".join(str(violation) for violation in violations)


@pytest.fixture(scope="module")
def tracked_in_repo() -> list[TrackedFile]:
    if not _in_git_work_tree(REPO_ROOT):
        pytest.skip("the repository is not a git work tree, so no file is under version control")
    return list_tracked_files(REPO_ROOT)


# Rule checks on synthetic inputs, without git.


@pytest.mark.parametrize("suffix", sorted(DENIED_SUFFIXES))
def test_denied_extension_is_flagged_in_any_directory(suffix: str) -> None:
    files = [
        TrackedFile(f"model{suffix}", 10),
        TrackedFile(f"deep/nested/model{suffix}", 10),
        TrackedFile(f"specs/001-kill-test-mvp/results/model{suffix.upper()}", 10),
    ]
    assert [v.path for v in denied_extension_violations(files)] == [f.path for f in files]


def test_ordinary_extensions_pass_the_deny_list() -> None:
    files = [
        TrackedFile("configs/tiny.yaml", 100),
        TrackedFile("src/strike_a_pose/cli.py", 100),
        TrackedFile("specs/001-kill-test-mvp/results/run_record.json", 100),
    ]
    assert denied_extension_violations(files) == []


@pytest.mark.parametrize(
    "path",
    [
        "specs/001-kill-test-mvp/results/report/coverage.png",
        "specs/002-next-feature/results/width.jpg",
        "tests/fixtures/tiny.png",
    ],
)
def test_images_under_results_and_fixtures_pass(path: str) -> None:
    assert image_location_violations([TrackedFile(path, 10)]) == []


@pytest.mark.parametrize(
    "path",
    [
        "docs/figure.png",
        "docs/FIGURE.PNG",
        "specs/001-kill-test-mvp/figure.png",
        "specs/results/figure.png",
        "specs/001-kill-test-mvp/deep/results/figure.png",
        "tests/figure.png",
        "tests/fixtures.png",
        "src/strike_a_pose/fixtures/figure.jpg",
    ],
)
def test_images_elsewhere_are_flagged(path: str) -> None:
    assert [v.path for v in image_location_violations([TrackedFile(path, 10)])] == [path]


@pytest.mark.parametrize(
    "header",
    [
        "body_id,height_cm",
        "subject_id,chest_cm",
        "pose_body_0,pose_body_1",
        "Body_ID,height_cm",
        "cell_id,pose_body_0_angle",
    ],
)
def test_csv_header_naming_a_body_or_subject_is_flagged(header: str) -> None:
    assert len(csv_header_violations([TrackedFile("out/table.csv", 10, header)])) == 1


@pytest.mark.parametrize(
    "header",
    [
        "cell_id,views,noise_deg,measurement",
        "dataset,split,mask_source,cell_id",
        "pose_root_0,pose_root_1,betas_0",
    ],
)
def test_csv_header_without_body_or_subject_passes(header: str) -> None:
    assert csv_header_violations([TrackedFile("out/table.csv", 10, header)]) == []


def test_only_csv_files_have_their_header_checked() -> None:
    assert csv_header_violations([TrackedFile("notes.txt", 10, "body_id,x")]) == []


def test_csv_header_is_the_first_line_without_bom_or_line_ending() -> None:
    assert csv_header(b"\xef\xbb\xbfbody_id,x\r\n1,2\r\n") == "body_id,x"
    assert csv_header(b"a,b\n1,2") == "a,b"
    assert csv_header(b"") == ""


def test_one_megabyte_limit_is_one_million_bytes() -> None:
    at_limit = TrackedFile("data/table.bin", MAX_TRACKED_BYTES)
    over_limit = TrackedFile("data/table.bin", MAX_TRACKED_BYTES + 1)
    assert size_violations([at_limit]) == []
    assert [v.path for v in size_violations([over_limit])] == ["data/table.bin"]


def test_fixture_limit_is_100_kilobytes_and_covers_only_tests_fixtures() -> None:
    files = [
        TrackedFile("tests/fixtures/at_limit.txt", MAX_FIXTURE_BYTES),
        TrackedFile("tests/fixtures/over_limit.txt", MAX_FIXTURE_BYTES + 1),
        TrackedFile("tests/other/half_megabyte.txt", 500_000),
    ]
    assert [v.path for v in fixture_size_violations(files)] == ["tests/fixtures/over_limit.txt"]


def test_violations_are_reported_only_for_tracked_files(tmp_path: Path) -> None:
    """Run the full check on a scratch git repository, so the git plumbing is exercised too."""
    repo = tmp_path / "scratch"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    contents = {
        "data/manifest.npz": b"not a real archive",
        "docs/figure.png": b"png bytes",
        "specs/001-kill-test-mvp/results/figure.png": b"png bytes",
        "tests/fixtures/small.jpg": b"jpg bytes",
        "out/bodies.csv": b"body_id,height_cm\n0,170.5\n",
        "out/cells.csv": b"cell_id,coverage\n1,0.9\n",
        "big/archive.txt": b"x" * (MAX_TRACKED_BYTES + 1),
        "tests/fixtures/large.txt": b"x" * (MAX_FIXTURE_BYTES + 1),
        "untracked/weights.pt": b"written to the working tree but never added",
    }
    for relative, data in contents.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _git(repo, "add", "--", *(p for p in contents if not p.startswith("untracked/")))

    violations = all_violations(list_tracked_files(repo))

    assert {v.path for v in violations} == {
        "data/manifest.npz",
        "docs/figure.png",
        "out/bodies.csv",
        "big/archive.txt",
        "tests/fixtures/large.txt",
    }


# Rule checks on the files that the repository tracks.


def test_repository_listing_contains_a_known_file(tracked_in_repo: list[TrackedFile]) -> None:
    assert "pyproject.toml" in {f.path for f in tracked_in_repo}


def test_no_denied_extension_is_under_version_control(tracked_in_repo: list[TrackedFile]) -> None:
    violations = denied_extension_violations(tracked_in_repo)
    assert not violations, _report(violations)


def test_png_and_jpg_are_tracked_only_under_results_and_fixtures(
    tracked_in_repo: list[TrackedFile],
) -> None:
    violations = image_location_violations(tracked_in_repo)
    assert not violations, _report(violations)


def test_no_tracked_csv_header_names_a_body_or_subject(
    tracked_in_repo: list[TrackedFile],
) -> None:
    violations = csv_header_violations(tracked_in_repo)
    assert not violations, _report(violations)


def test_no_tracked_file_is_over_one_megabyte(tracked_in_repo: list[TrackedFile]) -> None:
    violations = size_violations(tracked_in_repo)
    assert not violations, _report(violations)


def test_tracked_fixture_files_are_at_most_100_kilobytes(
    tracked_in_repo: list[TrackedFile],
) -> None:
    violations = fixture_size_violations(tracked_in_repo)
    assert not violations, _report(violations)
