import json
from pathlib import Path

import pytest

from alphafold3tools.yamltojson import convert_yaml_to_af3_json, yamltojson


def test_convert_yaml_to_af3_json_maps_entities_and_bonds(tmp_path):
    input_yaml = tmp_path / "input.yaml"
    output_json = tmp_path / "output.json"
    input_yaml.write_text(
        """version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MSTN
      msa: protein.a3m
      modifications:
        - position: 2
          ccd: SEP
  - rna:
      id: C
      sequence: AGCU
      modifications:
        - position: 1
          ccd: 2MG
  - dna:
      id: D
      sequence: ACTG
  - ligand:
      id: E
      smiles: 'CCC\\C=C'
  - ligand:
      id: [F, G]
      ccd: [NAG, FUC]
constraints:
  - bond:
      atom1: [A, 2, CA]
      atom2: [F, 1, C1]
"""
    )
    (tmp_path / "protein.a3m").write_text(">query\nMSTN\n")
    data = convert_yaml_to_af3_json(
        {"version": 1,
         "sequences": [
             {"protein": {"id": ["A", "B"], "sequence": "MSTN", "msa": "protein.a3m", "modifications": [{"position": 2, "ccd": "SEP"}]}},
             {"rna": {"id": "C", "sequence": "AGCU", "modifications": [{"position": 1, "ccd": "2MG"}]}},
             {"dna": {"id": "D", "sequence": "ACTG"}},
             {"ligand": {"id": "E", "smiles": r"CCC\C=C"}},
             {"ligand": {"id": ["F", "G"], "ccd": ["NAG", "FUC"]}},
         ],
         "constraints": [{"bond": {"atom1": ["A", 2, "CA"], "atom2": ["F", 1, "C1"]}}],
        },
        yaml_path=input_yaml.resolve(),
        output_path=output_json.resolve(),
    )
    assert data["dialect"] == "alphafold3"
    assert data["version"] == 4
    assert data["modelSeeds"] == [1]
    protein = data["sequences"][0]["protein"]
    assert protein["unpairedMsaPath"] == "protein.a3m"
    assert protein["pairedMsa"] == ""
    assert protein["modifications"] == [{"ptmType": "SEP", "ptmPosition": 2}]
    assert data["sequences"][1]["rna"]["modifications"] == [
        {"modificationType": "2MG", "basePosition": 1}
    ]
    assert data["sequences"][3]["ligand"]["smiles"] == r"CCC\C=C"
    assert data["sequences"][4]["ligand"]["ccdCodes"] == ["NAG", "FUC"]
    assert data["bondedAtomPairs"] == [[["A", 2, "CA"], ["F", 1, "C1"]]]


def test_yamltojson_writes_json_and_warns_for_ignored_fields(tmp_path, capsys):
    input_yaml = tmp_path / "input.yaml"
    output_json = tmp_path / "output.json"
    input_yaml.write_text(
        """sequences:
  - protein:
      id: A
      sequence: MSTN
      msa: empty
constraints:
  - pocket:
      binder: A
      contacts: [[A, 1]]
properties:
  - affinity:
      binder: A
templates:
  - cif: template.cif
"""
    )

    yamltojson(str(input_yaml), str(output_json))

    written = json.loads(output_json.read_text())
    stderr = capsys.readouterr().err
    assert written["sequences"][0]["protein"]["unpairedMsa"] == ""
    assert written["sequences"][0]["protein"]["pairedMsa"] == ""
    assert written["bondedAtomPairs"] is None
    assert "properties" in stderr
    assert "constraints.pocket" in stderr
    assert "Templates were ignored" in stderr


def test_convert_yaml_to_af3_json_rejects_csv_msa(tmp_path):
    with pytest.raises(ValueError, match="CSV"):
        convert_yaml_to_af3_json(
            {
                "sequences": [
                    {"protein": {"id": "A", "sequence": "MSTN", "msa": "paired.csv"}}
                ]
            },
            yaml_path=(tmp_path / "input.yaml").resolve(),
            output_path=(tmp_path / "output.json").resolve(),
        )
