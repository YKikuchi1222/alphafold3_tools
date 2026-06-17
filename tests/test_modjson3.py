import json
from pathlib import Path

import pytest

from alphafold3tools.modjson3 import (
    add_entities,
    collect_entity_additions,
    modjson3,
    parse_entity_file,
)


def test_add_entities_continues_existing_ids_and_preserves_order():
    data = {
        "dialect": "alphafold3",
        "version": 4,
        "name": "testprotein",
        "sequences": [
            {"protein": {"id": ["A", "B"], "sequence": "MSN", "templates": []}},
            {"ligand": {"id": "C", "ccdCodes": ["ATP"]}},
        ],
    }
    additions = collect_entity_additions(
        ["-a", "rna", "AGCU", "-a", "dna", "ACTG", "-a", "ccdCode", "MG"]
    )
    added_data = add_entities(data, additions)
    assert added_data["sequences"][3]["dna"]["id"] == "E"
    assert added_data["sequences"][2]["rna"]["id"] == "D"
    assert added_data["sequences"][4]["ligand"]["id"] == "F"
    assert added_data["sequences"][2]["rna"]["sequence"] == "AGCU"
    assert added_data["sequences"][3]["dna"]["sequence"] == "ACTG"
    assert added_data["sequences"][4]["ligand"]["ccdCodes"] == ["MG"]


def test_add_entities_smiles_is_json_escaped_when_written(tmp_path):
    data = {
        "dialect": "alphafold3",
        "version": 4,
        "name": "testprotein",
        "sequences": [],
    }
    smiles = r"CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\CO"
    added_data = add_entities(data, [collect_entity_additions(["-a", "smiles", smiles])[0]])
    output_file = tmp_path / "out.json"
    output_file.write_text(json.dumps(added_data, indent=2))
    written = output_file.read_text()
    assert "\\\\C=C\\\\" in written
    loaded = json.loads(written)
    assert loaded["sequences"][0]["ligand"]["smiles"] == smiles


def test_parse_entity_file_accepts_optional_quotes(tmp_path):
    entity_file = tmp_path / "entities.txt"
    entity_file.write_text('rna "AGCU"\ndna ACTG\nccdCode "MG"\n')
    additions = parse_entity_file(str(entity_file))
    assert additions[0].entity_type == "rna"
    assert additions[0].value == "AGCU"
    assert additions[1].value == "ACTG"
    assert additions[2].value == "MG"


def test_parse_entity_file_rejects_three_columns_with_line_number(tmp_path):
    entity_file = tmp_path / "entities.txt"
    entity_file.write_text("rna AGCU extra\n")
    with pytest.raises(ValueError, match=r"line 1"):
        parse_entity_file(str(entity_file))


def test_collect_entity_additions_preserves_cli_appearance_order(tmp_path):
    entity_file = tmp_path / "entities.txt"
    entity_file.write_text("dna ACTG\n")
    additions = collect_entity_additions(
        ["-a", "rna", "AGCU", "-f", str(entity_file), "-a", "ccdCode", "ATP"]
    )
    assert [(addition.entity_type, addition.value) for addition in additions] == [
        ("rna", "AGCU"),
        ("dna", "ACTG"),
        ("ccdCode", "ATP"),
    ]


def test_modjson3_writes_double_quoted_json_strings(tmp_path):
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.json"
    input_json.write_text(
        json.dumps(
            {
                "dialect": "alphafold3",
                "version": 4,
                "name": "testprotein",
                "sequences": [{"protein": {"id": "A", "sequence": "MSN", "templates": []}}],
                "modelSeeds": [1],
                "bondedAtomPairs": None,
                "userCCD": None,
            }
        )
    )
    modjson3(
        str(input_json),
        str(output_json),
        entity_additions=collect_entity_additions(
            ["-a", "smiles", r"CCC\C=C", "-a", "rna", "AGCU", "-a", "ccdCode", "MG"]
        ),
    )
    written = output_json.read_text()
    assert '"dialect": "alphafold3"' in written
    assert '"name": "testprotein"' in written
    assert '"id": "B"' in written
    assert '"smiles": "CCC\\\\C=C"' in written
    assert '"sequence": "AGCU"' in written
    assert '"ccdCodes": [' in written
    assert '"MG"' in written
