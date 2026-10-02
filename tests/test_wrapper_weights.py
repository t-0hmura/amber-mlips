from pathlib import Path
import importlib
import importlib.util
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def cli():
    directory = Path(__file__).resolve().parents[1] / "plugins"
    name = "amber_weights_wrapper"
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + ".cli_amber")


@pytest.fixture
def inputs(tmp_path):
    weights = tmp_path / "local weights.pt"
    weights.write_bytes(b"offline fixture")
    mdin = tmp_path / "input.in"
    mdin.write_text(
        "ML/MM\n&cntrl\n ifqnt=1,\n/\n"
        "&qmmm\n qm_theory='orb',\n ml_keywords='--model orb-v3-conservative-omol',\n/\n"
    )
    return mdin, weights


@pytest.mark.parametrize("flag", ["-w", "--weights-file"])
def test_wrapper_forwards_weights_to_model_server(cli, inputs, monkeypatch, tmp_path, flag):
    mdin, weights = inputs
    commands = []
    keywords = []
    monkeypatch.setattr(cli, "_resolve_sander_bin", lambda *a, **k: "/sander")
    monkeypatch.setattr(cli, "_stage_qchem_shim", lambda *a, **k: (str(tmp_path), str(tmp_path / "qchem")))
    def start_server(*, ml_keywords, **kwargs):
        keywords.append(ml_keywords)
        return str(tmp_path / "model.sock")
    monkeypatch.setattr(cli, "_start_model_server", start_server)
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    assert cli.main([flag, str(weights), "-O", "-i", str(mdin)]) == 0
    assert len(keywords) == len(commands) == 1
    shim = importlib.import_module(cli.__package__ + ".nonmpi_qc_shim")
    parsed = shim._parse_keywords("orb", keywords[0])
    assert parsed.weights_file == str(weights)
    assert parsed.model == "orb-v3-conservative-omol"
    assert flag not in commands[0]
    assert str(weights) not in commands[0]


def test_wrapper_rejects_conflicting_mdin_weights(cli, inputs, tmp_path, capsys):
    mdin, weights = inputs
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    mdin.write_text(
        "ML/MM\n&cntrl\n ifqnt=1,\n/\n"
        "&qmmm\n qm_theory='orb',\n ml_keywords='--weights-file other.pt',\n/\n"
    )
    assert cli.main(["--weights-file", str(weights), "-i", str(mdin)]) == 1
    assert "--weights-file conflicts with ml_keywords" in capsys.readouterr().err
