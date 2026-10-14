"""The command-line interface: parsing, single runs, batches, and the launcher."""

import argparse
import ast
import configparser
import io
import json
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from support import read_csv_rows, write_synthetic_pdb

import xtal.rcsb as rcsb
from epath import __version__
from epath.cli import main
from epath.cli.args import (
    bdamage_pipeline_options_from_args,
    epath_pipeline_options_from_args,
    fraction_float,
    parse_args,
    percent_float,
    percentile_list,
    positive_float,
    positive_int,
)
from epath.cli.run import default_output_prefix, output_label_for_input
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


def test_defaults_mirror_the_pipeline_options() -> None:
    args = parse_args(["model.pdb"])

    assert args.structure_inputs == ["model.pdb"]
    assert args.percentile == (2.0,) and args.chains is None
    assert args.cluster_distance == 4.0 and args.minimum_cluster_size == 8
    assert args.packing_density_threshold == 7.0 and args.window_size_fraction == 0.02
    assert args.minimum_window_size == 11 and args.translation_range == 1
    assert args.output_prefix is None and args.output_dir == "output"
    assert not (
        args.no_csv or args.no_cif or args.no_plot or args.no_reports or args.quiet
    )

    parsed = parse_args(
        [
            "a.pdb",
            "--percentile",
            "5",
            "--no-reports",
            "--no-plot",
            "--chains",
            "B, A,B",
        ]
    )
    assert parsed.percentile == (5.0,)
    options = epath_pipeline_options_from_args(
        parsed, percentile=parsed.percentile[0], output_prefix=Path("x")
    )
    assert options.selection.percentile == 5.0
    assert options.output.display_chains == ("B", "A")
    assert options.output.output_prefix == Path("x")
    assert not options.output.write_reports
    assert (
        not options.output.write_z_score_plot
        and options.output.write_combined_csv
        and options.output.write_structure_files
    )

    bdamage = bdamage_pipeline_options_from_args(
        parse_args(
            [
                "a.pdb",
                "--packing-density-threshold",
                "6.5",
                "--translation-range",
                "2",
            ]
        )
    )
    assert bdamage.packing_density_threshold == 6.5 and bdamage.translation_range == 2

    # Several percentiles keep their order, with a repeat kept once.
    assert parse_args(["a.pdb", "--percentile", "3, 1,3.0,2"]).percentile == (
        3.0,
        1.0,
        2.0,
    )

    batch = parse_args(["a.pdb", "b.pdb"])
    assert batch.structure_inputs == ["a.pdb", "b.pdb"]


@pytest.mark.parametrize(
    ("converter", "good", "bad"),
    [
        (positive_float, ("0.5", 0.5), ("0", "-1", "inf", "x")),
        (fraction_float, ("0.25", 0.25), ("0", "1", "nan")),
        (percent_float, ("100", 100.0), ("-0.1", "100.1", "x")),
        (percentile_list, ("1,2.5", (1.0, 2.5)), ("", "1,,2", "1,150", "1,x")),
        (positive_int, ("3", 3), ("0", "-2", "1.5")),
    ],
)
def test_argument_validators(
    converter: Callable[[str], object],
    good: tuple[str, object],
    bad: tuple[str, ...],
) -> None:
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
    assert len(read_csv_rows(prefix.parent / "run_epath.csv")) == 77
    assert len(read_csv_rows(prefix.parent / "run_clusters.csv")) == 3
    summary = (prefix.parent / "run_summary.txt").read_text()
    assert "Path count: 3" in summary and "Window size used: 11 atoms" in summary


def test_several_percentiles_share_one_bdamage_calculation(
    synthetic_pdb: Path, tmp_path: Path
) -> None:
    prefix = tmp_path / "multi" / "run"
    code, out, err = _run(
        [
            str(synthetic_pdb),
            "-o",
            str(prefix),
            "--percentile",
            "20,30",
            "--minimum-cluster-size",
            "3",
            "--no-plot",
        ]
    )

    assert code == 0, err
    assert err.count("Calculating BDamage scores...") == 1
    assert err.count("Calculating Bnet and Bnet percentile...") == 1
    assert err.count("Selecting E-path atoms...") == 2
    assert "Pathways at 20%:" in out and "Pathways at 30%:" in out
    assert out.count("RABDAM Bnet results:") == 1
    assert sorted(p.name for p in prefix.parent.iterdir()) == [
        "percentile_20",
        "percentile_30",
        "run_runs.csv",
    ]
    rows = read_csv_rows(prefix.parent / "run_runs.csv")
    assert [row["cutoff_percent"] for row in rows] == ["20", "30"]
    assert [row["atoms_selected"] for row in rows] == ["15", "23"]
    assert rows[0]["bnet"] == rows[1]["bnet"]
    for percentile, row in zip(("20", "30"), rows, strict=True):
        directory = prefix.parent / f"percentile_{percentile}"
        assert read_csv_rows(directory / "run_run.csv") == [row]
        assert (directory / "run_summary.txt").is_file()
        assert (directory / "run_path01.cif").is_file()

    # Switching reports off drops the percentile table with the other reports.
    prefix = tmp_path / "multi_noreports" / "run"
    code, _, _ = _run(
        [str(synthetic_pdb), "-o", str(prefix), "-q", "--percentile", "20,30"]
        + ["--no-reports", "--no-plot", "--no-cif"]
    )
    assert code == 0
    assert sorted(p.name for p in prefix.parent.iterdir()) == [
        "percentile_20",
        "percentile_30",
    ]


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


def test_chain_filter_is_checked_before_anything_is_written(
    synthetic_pdb: Path, tmp_path: Path
) -> None:
    for extra in ([], ["--no-cif"]):
        prefix = tmp_path / f"bad{len(extra)}" / "run"
        code, _, err = _run(
            [str(synthetic_pdb), "-o", str(prefix), "-q", "--chains", "A,Z", *extra]
        )
        assert code == 1 and not prefix.parent.exists()
        assert "not in the model: Z. The model has chains A, B." in err

    # Without coordinate files there is nothing for the filter to apply to.
    prefix = tmp_path / "nocif" / "run"
    code, _, err = _run(
        [str(synthetic_pdb), "-o", str(prefix), "-q", "--chains", "B", "--no-cif"]
    )
    assert code == 0, err
    assert (
        "Chains in coordinate files"
        not in (prefix.parent / "run_summary.txt").read_text()
    )


def test_default_output_prefix_uses_the_input_stem(
    synthetic_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code, _, _ = _run([str(synthetic_pdb), "-q", "--no-plot"])

    assert code == 0
    assert (tmp_path / "output" / "synthetic" / "synthetic_summary.txt").is_file()
    assert (tmp_path / "output" / "synthetic" / "synthetic_epath.csv").is_file()

    code, _, _ = _run(
        [str(synthetic_pdb), "--output-dir", str(tmp_path / "elsewhere"), "--no-plot"]
    )
    assert (
        code == 0
        and (tmp_path / "elsewhere" / "synthetic" / "synthetic_run.csv").is_file()
    )


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
    assert not (tmp_path / "y").exists()


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
    synthetic_pdb.copy(duplicate)
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
    rows = read_csv_rows(out_dir / "epath_runs.csv")
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

    # A second batch into the same directory overwrites nothing of the first.
    first_cohort = (out_dir / "epath_runs.csv").read_text()
    code, _, _ = _run(
        [str(synthetic_pdb), str(synthetic_mmcif), "--output-dir", str(out_dir)]
        + ["-q", "--no-plot", "--no-cif"]
    )
    assert code == 0
    assert (out_dir / "epath_runs.csv").read_text() == first_cohort
    assert len(read_csv_rows(out_dir / "epath_runs_1.csv")) == 2
    assert (out_dir / "synthetic_4" / "synthetic_run.csv").is_file()


def test_batch_runs_every_percentile_of_every_structure(
    synthetic_pdb: Path, synthetic_mmcif: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "batch"
    code, _, err = _run(
        [str(synthetic_pdb), str(synthetic_mmcif), "--output-dir", str(out_dir)]
        + ["-q", "--no-plot", "--no-cif", "--percentile", "20,30"]
    )

    assert code == 0, err
    rows = read_csv_rows(out_dir / "epath_runs.csv")
    assert [(row["input_path"], row["cutoff_percent"]) for row in rows] == [
        (str(synthetic_pdb.resolve()), "20"),
        (str(synthetic_pdb.resolve()), "30"),
        (str(synthetic_mmcif.resolve()), "20"),
        (str(synthetic_mmcif.resolve()), "30"),
    ]
    for label in ("synthetic", "synthetic_1"):
        assert len(read_csv_rows(out_dir / label / "synthetic_runs.csv")) == 2
        assert (out_dir / label / "percentile_30" / "synthetic_run.csv").is_file()


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
    rows = read_csv_rows(out_dir / "epath_runs.csv")
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
    # Only the extension is left once the stem is cleaned, and an extension is
    # not a name, so the generic label is used.
    assert output_label_for_input(dots) == "structure"

    assert (
        default_output_prefix(downloaded, output_dir=tmp_path)
        == tmp_path / "1ABC" / "1ABC"
    )
    # Run directories already on disk are skipped.
    (tmp_path / "1ABC").mkdir()
    (tmp_path / "1ABC_1").mkdir()
    assert (
        default_output_prefix(downloaded, output_dir=tmp_path)
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


def _run_launcher(
    launcher: Path, *python_flags: str
) -> subprocess.CompletedProcess[str]:
    """Run a launcher through this interpreter, so the test also works on Windows."""
    return subprocess.run(
        [sys.executable, *python_flags, str(launcher), "--version"],
        capture_output=True,
        text=True,
        cwd=launcher.parent,
        check=False,
    )


def test_launcher_reports_missing_libraries() -> None:
    # Without the site module, none of the installed libraries can be found.
    completed = _run_launcher(REPO_ROOT / "epath", "-S")
    assert completed.returncode == 1
    assert (
        "missing required libraries: gemmi numpy scipy matplotlib" in completed.stderr
    )


def test_launcher_reports_an_old_python(tmp_path: Path) -> None:
    source = (REPO_ROOT / "epath").read_text(encoding="utf-8")
    assert "MINIMUM_PYTHON = (3, 14)" in source
    launcher = tmp_path / "epath"
    launcher.write_text(
        source.replace("MINIMUM_PYTHON = (3, 14)", "MINIMUM_PYTHON = (99, 0)"),
        encoding="utf-8",
    )
    completed = _run_launcher(launcher)
    assert completed.returncode == 1
    assert "Python 99.0 or newer is required" in completed.stderr


def test_launcher_copied_out_of_the_clone_says_so(tmp_path: Path) -> None:
    launcher = tmp_path / "epath"
    shutil.copy(REPO_ROOT / "epath", launcher)
    completed = _run_launcher(launcher)
    assert completed.returncode == 1
    assert "cannot find the E-path sources" in completed.stderr


def test_launcher_stays_readable_by_old_pythons() -> None:
    # The launcher must parse and reach its version check on Python 2.7 and
    # early Python 3, which the linters (all targeting 3.14) cannot check.
    tree = ast.parse((REPO_ROOT / "epath").read_text(encoding="utf-8"))
    imports_before_main = [
        alias.name if isinstance(node, ast.Import) else node.module
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    ]
    assert imports_before_main == ["__future__", "sys"]

    too_new = (
        ast.JoinedStr,  # f-strings: 3.6
        ast.AnnAssign,  # variable annotations: 3.6
        ast.Starred,  # several unpackings in one call: 3.5; keep it simple
        ast.NamedExpr,  # 3.8
        ast.Match,  # 3.10
        ast.AsyncFunctionDef,  # 3.5
        ast.Await,  # 3.5
    )
    for node in ast.walk(tree):
        assert not isinstance(node, too_new), ast.dump(node)
        if isinstance(node, ast.FunctionDef):
            assert node.returns is None
            assert all(arg.annotation is None for arg in node.args.args)
            assert not node.args.kwonlyargs


def test_launcher_minimum_python_matches_the_project_configuration() -> None:
    tree = ast.parse((REPO_ROOT / "epath").read_text(encoding="utf-8"))
    (minimum,) = (
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "MINIMUM_PYTHON" for t in node.targets
        )
    )
    version = "{}.{}".format(*minimum)

    mypy = configparser.ConfigParser()
    mypy.read(REPO_ROOT / "mypy.ini", encoding="utf-8")
    assert mypy["mypy"]["python_version"] == version

    pyright = json.loads((REPO_ROOT / "pyrightconfig.json").read_text(encoding="utf-8"))
    assert pyright["pythonVersion"] == version

    ruff = tomllib.loads((REPO_ROOT / "ruff.toml").read_text(encoding="utf-8"))
    assert ruff["target-version"] == "py" + version.replace(".", "")

    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert set(re.findall(r'python-version: "([\d.]+)"', ci)) == {version}

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Python {version} or newer" in readme
