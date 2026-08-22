"""Integration tests against real OmniFold HDF5 files when available locally."""

from pathlib import Path

import pytest

from omnifold_publication.validation import validate_package
from omnifold_publication.writer import write_package


@pytest.mark.parametrize(
    ("input_path", "output_name"),
    [
        (Path("data/multifold.h5"), "zjets_nominal"),
        (Path("data/multifold_sherpa.h5"), "zjets_sherpa"),
        (Path("data/multifold_nonDY.h5"), "zjets_nonDY"),
    ],
)
def test_write_package_real_data(input_path, output_name, tmp_path):
    """write_package works against real OmniFold HDF5 files."""

    if not input_path.exists():
        pytest.skip(f"real data not available: {input_path}")

    out = write_package(
        input_path=input_path,
        output_dir=tmp_path / output_name,
        event_count=50_000,
    )

    errors = validate_package(out)
    assert errors == []
