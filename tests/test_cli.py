"""The command-line interface: parsing, single runs, batches, and the launcher."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from support import write_synthetic_pdb

import xtal.rcsb as rcsb
from epath import __version__
from epath.cli import main
from epath.cli.args import (
    REMOVED_BDAMAGE_COMMAND_MESSAGE,
    bdamage_workflow_options_from_args,
    epath_workflow_options_from_args,
    fraction_float,
    parse_args,
    percent_float,
    positive_float,
    positive_int,
)
from epath.cli.run import batch_output_prefix_path, output_label_for_input
from xtal.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI in-process and return its exit code, stdout, and stderr."""
    stdout, stderr = io.StringIO(), io.StringIO()
    code = main(argv, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def _rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into one dict per row."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_defaults_mirror_the_workflow_options() -> None:
    args = parse_args(["model.pdb"])

    assert (
        args.structure_inputs == ["model.pdb"] and args.structure_input == "model.pdb"
    )
    assert args.percentile == 2.0 and args.chains is None
    assert args.cluster_distance == 4.0 and args.minimum_cluster_size == 8
    assert args.packing_density_threshold == 7.0 and args.window_size_fraction == 0.02
    assert args.minimum_window_size == 11 and args.translation_range == 1
    assert args.output_prefix is None and args.output_dir == "output"
    assert not (
        args.no_csv or args.no_cif or args.no_plot or args.no_reports or args.quiet
    )

    options = epath_workflow_options_from_args(
        parse_args(
            [
                "a.pdb",
                "--percentile",
                "5",
                "--no-reports",
                "--no-plot",
                "--chains",
                "B, A,B",
            ]
        ),
        output_prefix=Path("x"),
    )
    assert options.selection.percentile == 5.0
    assert options.output.display_chains == ("B", "A")
    assert options.output.output_prefix == Path("x")
    assert not (
        options.output.write_cluster_csv
        or options.output.write_step_csv
        or options.output.write_run_csv
        or options.output.write_summary_report
    )
    assert (
        not options.output.write_z_score_plot
        and options.output.write_combined_csv
        and options.output.write_structure_files
    )

    bdamage = bdamage_workflow_options_from_args(
        parse_args(
            [
                "a.pdb",
                "--packing-density-threshold",
                "6.5",
                "--translation-range",
                "2",
                "--materialize-translated-block",
            ]
        )
    )
    assert (
        bdamage.packing_density_threshold == 6.5
        and bdamage.translation_range == 2
        and bdamage.materialize_translated_block
    )

    batch = parse_args(["a.pdb", "b.pdb"])
    assert (
        batch.structure_inputs == ["a.pdb", "b.pdb"] and batch.structure_input is None
    )


@pytest.mark.parametrize(
    ("converter", "good", "bad"),
    [
        (positive_float, ("0.5", 0.5), ("0", "-1", "inf", "x")),
        (fraction_float, ("0.25", 0.25), ("0", "1", "nan")),
        (percent_float, ("100", 100.0), ("-0.1", "100.1", "x")),
        (positive_int, ("3", 3), ("0", "-2", "1.5")),
    ],
)
def test_argument_validators(
    converter: Callable[[str], float],
    good: tuple[str, float],
    bad: tuple[str, ...],
) -> None:
    import argparse

    assert converter(good[0]) == good[1]
    for value in bad:
        with pytest.raises(argparse.ArgumentTypeError):
            converter(value)


def test_argparse_rejects_bad_values() -> None:
    with pytest.raises(SystemExit):
        parse_args(["a.pdb", "--percentile", "150"])
    for chains in ("", "A,,B", " "):
        with pytest.raises(SystemExit):
            parse_args(["a.pdb", "--chains", chains])
    with pytest.raises(SystemExit) as info:
        parse_args(["bdamage", "a.pdb"])
    assert info.value.code == REMOVED_BDAMAGE_COMMAND_MESSAGE


def test_informational_invocations(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run([])
    assert code == 2 and "missing required argument" in err and out == ""

    code, out, _ = _run(["--citations"])
    assert code == 0 and "RABDAM" in out and "Nat Commun" in out

    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"epath {__version__}"


def test_single_run_writes_every_output(synthetic_pdb: Path, tmp_path: Path) -> None:
    prefix = tmp_path / "out" / "run"
    code, out, err = _run(
        [
            str(synthetic_pdb),
            "-o",
            str(prefix),
            "--percentile",
            "30",
            "--minimum-cluster-size",
            "3",
        ]
    )

    assert code == 0, err
    assert "Done." in out and "Total runtime" in out
    assert "Pathways:" in out and "Cofactor sites: FMN" in out
    assert "Calculating BDamage scores..." in err and "done in" in err
    written = sorted(p.name for p in prefix.parent.iterdir())
    assert written == [
        "run_clusters.csv",
        "run_epath.csv",
        "run_model.cif",
        "run_path01.cif",
        "run_path02.cif",
        "run_path03.cif",
        "run_run.csv",
        "run_selected.cif",
        "run_steps.csv",
        "run_summary.txt",
        "run_zfalloff.png",
    ]
    assert len(_rows(prefix.parent / "run_epath.csv")) == 77
    assert len(_rows(prefix.parent / "run_clusters.csv")) == 3
    summary = (prefix.parent / "run_summary.txt").read_text()
    assert "Path count: 3" in summary and "Window size used: 11 atoms" in summary


def test_output_groups_can_be_disabled_and_quiet_silences_progress(
    synthetic_pdb: Path, tmp_path: Path
) -> None:
    prefix = tmp_path / "quiet" / "run"
    code, out, err = _run(
        [
            str(synthetic_pdb),
            "-o",
            str(prefix),
            "-q",
            "--no-plot",
            "--no-cif",
            "--no-reports",
        ]
    )

    assert code == 0 and out == "" and err == ""
    assert sorted(p.name for p in prefix.parent.iterdir()) == ["run_epath.csv"]

    code, _, _ = _run(
        [
            str(synthetic_pdb),
            "-o",
            str(tmp_path / "none" / "run"),
            "-q",
            "--no-plot",
            "--no-cif",
            "--no-reports",
            "--no-csv",
        ]
    )
    assert code == 0 and not (tmp_path / "none").exists()


def test_default_output_prefix_uses_the_input_stem(
    synthetic_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code, out, _ = _run([str(synthetic_pdb), "-q", "--no-plot", "--preview-count", "3"])

    assert code == 0
    assert (tmp_path / "output" / "synthetic" / "synthetic_summary.txt").is_file()
    assert (tmp_path / "output" / "synthetic" / "synthetic_epath.csv").is_file()

    code, out, _ = _run(
        [
            str(synthetic_pdb),
            "--output-dir",
            str(tmp_path / "elsewhere"),
            "--no-plot",
            "--preview-count",
            "3",
        ]
    )
    assert (
        code == 0
        and (tmp_path / "elsewhere" / "synthetic" / "synthetic_run.csv").is_file()
    )
    assert "BDamage preview:" in out and "Packing-density counts, first 3:" in out


def test_invalid_structures_fail_cleanly(tmp_path: Path) -> None:
    no_cell = write_synthetic_pdb(tmp_path / "nocell.pdb", include_cryst1=False)
    code, _, err = _run([str(no_cell), "-o", str(tmp_path / "x"), "-q"])
    assert code == 1 and "epath: error:" in err and "unit-cell" in err

    code, _, err = _run([str(tmp_path / "missing.pdb"), "-q"])
    assert code == 1 and "Could not resolve structure input" in err

    assembly = write_synthetic_pdb(
        tmp_path / "assembly.pdb",
        cell=(6.0, 6.0, 6.0, 90.0, 90.0, 90.0),
        space_group="P 21 21 21",
    )
    code, _, err = _run([str(assembly), "-o", str(tmp_path / "y"), "-q"])
    assert code == 1 and "cannot fit in the unit cell" in err
    assert not (tmp_path / "y_epath.csv").exists()


def test_pdb_id_is_downloaded_once_and_labelled_by_id(
    synthetic_mmcif: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = synthetic_mmcif.read_bytes()
    calls: list[str] = []

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(url: str, timeout: float) -> Response:
        calls.append(url)
        return Response(payload)

    monkeypatch.setattr(rcsb, "urlopen", fake_urlopen)
    download_dir = tmp_path / "downloads"
    argv = [
        "1abc",
        "--output-dir",
        str(tmp_path / "out"),
        "--download-dir",
        str(download_dir),
        "-q",
        "--no-plot",
    ]

    code, _, err = _run(argv)
    assert code == 0, err
    assert calls == ["https://files.rcsb.org/download/1abc.cif"]
    assert (download_dir / "1ABC.cif").read_bytes() == payload
    assert (tmp_path / "out" / "1ABC" / "1ABC_summary.txt").is_file()
    assert (
        "Input structure: " + str((download_dir / "1ABC.cif").resolve())
        in (tmp_path / "out" / "1ABC" / "1ABC_summary.txt").read_text()
    )

    assert _run(argv)[0] == 0 and len(calls) == 1
    assert _run([*argv, "--overwrite-download"])[0] == 0 and len(calls) == 2


def test_batch_run_writes_a_cohort_table_with_unique_prefixes(
    synthetic_pdb: Path, synthetic_mmcif: Path, tmp_path: Path
) -> None:
    duplicate = tmp_path / "copy" / "synthetic.pdb"
    duplicate.parent.mkdir()
    shutil.copy(synthetic_pdb, duplicate)
    out_dir = tmp_path / "batch"

    code, out, err = _run(
        [
            str(synthetic_pdb),
            str(synthetic_mmcif),
            str(duplicate),
            "--output-dir",
            str(out_dir),
            "--no-plot",
            "--percentile",
            "30",
        ]
    )

    assert code == 0, err
    assert "Batch item 1/3" in err and "Batch item 3/3" in err
    assert "Successful: 3" in out and "Failed: 0" in out
    rows = _rows(out_dir / "epath_runs.csv")
    assert [row["structure"] for row in rows] == ["synthetic", "synthetic", "synthetic"]
    assert [row["input_path"] for row in rows] == [
        str(synthetic_pdb.resolve()),
        str(synthetic_mmcif.resolve()),
        str(duplicate.resolve()),
    ]
    assert (out_dir / "synthetic" / "synthetic_summary.txt").is_file()
    assert (out_dir / "synthetic_1" / "synthetic_summary.txt").is_file()
    assert (out_dir / "synthetic_2" / "synthetic_summary.txt").is_file()
    assert rows[0]["atoms_selected"] == rows[1]["atoms_selected"] == "23"


def test_batch_continues_past_a_failing_input(
    synthetic_pdb: Path, tmp_path: Path
) -> None:
    broken = write_synthetic_pdb(tmp_path / "broken.pdb", include_cryst1=False)
    out_dir = tmp_path / "batch"

    code, out, err = _run(
        [str(synthetic_pdb), str(broken), "--output-dir", str(out_dir), "-q"]
    )

    assert code == 1
    assert out == "" and f"epath: error: {broken}:" in err
    rows = _rows(out_dir / "epath_runs.csv")
    assert len(rows) == 1 and rows[0]["structure"] == "synthetic"

    code, _, err = _run(
        [str(synthetic_pdb), str(synthetic_pdb), "-o", str(tmp_path / "p"), "-q"]
    )
    assert (
        code == 1 and "--output-prefix can only be used with one structure input" in err
    )

    code, _, _ = _run(
        [
            str(synthetic_pdb),
            str(broken),
            "--output-dir",
            str(tmp_path / "noreports"),
            "-q",
            "--no-reports",
            "--no-plot",
        ]
    )
    assert code == 1 and not (tmp_path / "noreports" / "epath_runs.csv").exists()


def test_output_labels_and_prefixes(tmp_path: Path) -> None:
    local = ResolvedStructureInput(
        "weird name!.pdb",
        StructureSourceType.LOCAL_FILE,
        StructureFileFormat.PDB,
        local_path=tmp_path / "weird name!.pdb",
    )
    assert output_label_for_input(local) == "weird_name"
    downloaded = ResolvedStructureInput(
        "1abc",
        StructureSourceType.LOCAL_FILE,
        StructureFileFormat.MMCIF,
        local_path=tmp_path / "1ABC.cif",
        structure_id="1abc",
    )
    assert output_label_for_input(downloaded) == "1ABC"
    dots = ResolvedStructureInput(
        "...",
        StructureSourceType.LOCAL_FILE,
        StructureFileFormat.PDB,
        local_path=tmp_path / "....pdb",
    )
    assert output_label_for_input(dots) == "structure"

    used: set[Path] = set()
    assert (
        batch_output_prefix_path(
            downloaded, output_dir=tmp_path, used_output_prefixes=used
        )
        == tmp_path / "1ABC" / "1ABC"
    )
    assert (
        batch_output_prefix_path(
            downloaded, output_dir=tmp_path, used_output_prefixes=used
        )
        == tmp_path / "1ABC_1" / "1ABC"
    )
    # Run directories already on disk are skipped, even outside a batch.
    (tmp_path / "1ABC").mkdir()
    (tmp_path / "1ABC_1").mkdir()
    assert (
        batch_output_prefix_path(
            downloaded, output_dir=tmp_path, used_output_prefixes=None
        )
        == tmp_path / "1ABC_2" / "1ABC"
    )


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell launcher")
def test_launcher_runs_the_package_from_a_clone(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(REPO_ROOT / "epath"), "--version"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"epath {__version__}"

    failed = subprocess.run(
        [str(REPO_ROOT / "epath")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert failed.returncode == 2 and "missing required argument" in failed.stderr
