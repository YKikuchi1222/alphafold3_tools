import json
from pathlib import Path

import pytest

from alphafold3tools.jsontoyaml import boltz_yaml_dumps, convert_json_to_boltz_schema, jsontoyaml


def test_convert_json_to_boltz_schema_quotes_smiles_and_maps_entities(tmp_path):
    input_json = tmp_path / "input.json"
    output_yaml = tmp_path / "output.yaml"
    data = {
        "dialect": "alphafold3",
        "version": 4,
        "name": "job",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSTN",
                    "modifications": [{"ptmType": "SEP", "ptmPosition": 2}],
                    "unpairedMsaPath": "protein.a3m",
                    "pairedMsa": "",
                    "templates": [],
                }
            },
            {
                "rna": {
                    "id": "C",
                    "sequence": "AGCU",
                    "modifications": [{"modificationType": "2MG", "basePosition": 1}],
                }
            },
            {"dna": {"id": "D", "sequence": "ACTG", "modifications": []}},
            {"ligand": {"id": "E", "smiles": r"CCC\C=C"}},
            {"ligand": {"id": ["F", "G"], "ccdCodes": ["NAG", "FUC"]}},
        ],
        "modelSeeds": [1],
        "bondedAtomPairs": [[["A", 2, "CA"], ["F", 1, "C1"]]],
        "userCCD": None,
    }
    input_json.write_text(json.dumps(data))
    (tmp_path / "protein.a3m").write_text(">query\nMSTN\n")

    converted = convert_json_to_boltz_schema(
        data,
        json_path=input_json.resolve(),
        output_path=output_yaml.resolve(),
    )
    yaml_text = boltz_yaml_dumps(converted)

    assert "version: 1" in yaml_text
    assert "smiles: 'CCC\\C=C'" in yaml_text
    assert "ccd: [ NAG, FUC ]" in yaml_text
    assert "id: [ A, B ]" in yaml_text
    assert "msa: protein.a3m" in yaml_text
    assert "position: 2" in yaml_text
    assert "ccd: SEP" in yaml_text
    assert "atom1: [ A, 2, CA ]" in yaml_text
    assert "atom2: [ F, 1, C1 ]" in yaml_text


def test_jsontoyaml_writes_inline_msa_sidecar_and_yaml(tmp_path):
    input_json = tmp_path / "input.json"
    output_yaml = tmp_path / "output.yaml"
    data = {
        "dialect": "alphafold3",
        "version": 4,
        "name": "job",
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "MSTN",
                    "modifications": [],
                    "unpairedMsa": ">query\nMSTN\n",
                    "pairedMsa": "",
                    "templates": [],
                }
            }
        ],
        "modelSeeds": [1],
        "bondedAtomPairs": None,
        "userCCD": None,
    }
    input_json.write_text(json.dumps(data))

    jsontoyaml(str(input_json), str(output_yaml))

    yaml_text = output_yaml.read_text()
    assert "msa: output_msas/A.a3m" in yaml_text
    assert (tmp_path / "output_msas" / "A.a3m").read_text() == ">query\nMSTN\n"


def test_jsontoyaml_ignores_userccd_and_warns_to_stderr(tmp_path, capsys):
    input_json = tmp_path / "input.json"
    output_yaml = tmp_path / "output.yaml"
    data = {
        "dialect": "alphafold3",
        "version": 4,
        "name": "job",
        "sequences": [{"ligand": {"id": "A", "ccdCodes": ["MY-PRD"]}}],
        "modelSeeds": [1],
        "bondedAtomPairs": None,
        "userCCD": "data_MY-PRD",
    }
    input_json.write_text(json.dumps(data))

    jsontoyaml(str(input_json), str(output_yaml))

    yaml_text = output_yaml.read_text()
    captured = capsys.readouterr()
    assert "ccd: MY-PRD" in yaml_text
    assert "userCCD" in captured.err
