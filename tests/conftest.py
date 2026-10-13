"""Shared fixtures for the E-path test suite."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from support import write_synthetic_mmcif, write_synthetic_pdb


@pytest.fixture(autouse=True)
def matplotlib_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep matplotlib from writing its cache into the user's home directory."""
    previous = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(tmp_path_factory.mktemp("mpl"))
    yield
    if previous is None:
        os.environ.pop("MPLCONFIGDIR", None)
    else:
        os.environ["MPLCONFIGDIR"] = previous


@pytest.fixture(scope="session")
def synthetic_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic crystal as a PDB file."""
    return write_synthetic_pdb(tmp_path_factory.mktemp("structures") / "synthetic.pdb")


@pytest.fixture(scope="session")
def synthetic_mmcif(
    tmp_path_factory: pytest.TempPathFactory, synthetic_pdb: Path
) -> Path:
    """The synthetic crystal converted to mmCIF."""
    return write_synthetic_mmcif(
        tmp_path_factory.mktemp("structures") / "synthetic.cif", synthetic_pdb
    )
