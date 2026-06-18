from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.utils import add_version_option, write_af3_json


def read_yaml_data(yaml_path: str | Path) -> dict[str, Any]:
    with Path(yaml_path).open("r") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Input YAML must contain a top-level mapping.")
    return data


def _warn_ignored_yaml_only_fields(data: dict[str, Any]) -> None:
    ignored = []
    if data.get("properties"):
        ignored.append("properties")

    constraints = data.get("constraints") or []
    if any("pocket" in constraint for constraint in constraints):
        ignored.append("constraints.pocket")
    if any("contact" in constraint for constraint in constraints):
        ignored.append("constraints.contact")

    if ignored:
        print(
            "Warning: AF3 JSON does not support "
            + ", ".join(ignored)
            + ". These field(s) were ignored in the JSON output.",
            file=sys.stderr,
        )


def _resolve_input_path(raw_path: str, yaml_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (yaml_path.parent / path).resolve()


def _to_output_relative(raw_path: str, yaml_path: Path, output_path: Path) -> str:
    resolved = _resolve_input_path(raw_path, yaml_path)
    return os.path.relpath(resolved, output_path.parent)


def _convert_modifications(
    entity_type: str, modifications: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if entity_type == "protein":
        return [
            {"ptmType": mod["ccd"], "ptmPosition": mod["position"]}
            for mod in modifications
        ]
    return [
        {"modificationType": mod["ccd"], "basePosition": mod["position"]}
        for mod in modifications
    ]


def _convert_polymer_entity(
    entity_type: str,
    entity: dict[str, Any],
    *,
    yaml_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": entity["id"],
        "sequence": entity["sequence"],
        "modifications": _convert_modifications(entity_type, entity.get("modifications", [])),
    }

    if entity.get("cyclic"):
        print(
            f"Warning: AF3 JSON does not support cyclic {entity_type} directly. "
            f"The cyclic flag for chain {entity['id']} was ignored.",
            file=sys.stderr,
        )

    if entity_type == "protein":
        msa = entity.get("msa")
        if msa == "empty":
            out["unpairedMsa"] = ""
            out["pairedMsa"] = ""
        elif msa:
            msa_path = _to_output_relative(str(msa), yaml_path, output_path)
            if Path(str(msa)).suffix.lower() == ".csv":
                raise ValueError(
                    "Boltz protein msa CSV cannot be converted safely to AF3 JSON. "
                    "Please provide per-chain A3M instead."
                )
            out["unpairedMsaPath"] = msa_path
            out["pairedMsa"] = ""
        out["templates"] = []

    elif entity_type == "rna":
        msa = entity.get("msa")
        if msa:
            out["unpairedMsaPath"] = _to_output_relative(str(msa), yaml_path, output_path)

    return {entity_type: out}


def _convert_ligand_entity(entity: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": entity["id"]}
    if "smiles" in entity:
        out["smiles"] = entity["smiles"]
    elif "ccd" in entity:
        ccd = entity["ccd"]
        out["ccdCodes"] = ccd if isinstance(ccd, list) else [ccd]
    else:
        raise ValueError("Ligand must contain either smiles or ccd.")
    if entity.get("cyclic"):
        print(
            f"Warning: AF3 JSON does not support cyclic ligand directly. "
            f"The cyclic flag for chain {entity['id']} was ignored.",
            file=sys.stderr,
        )
    return {"ligand": out}


def _convert_sequences(
    data: dict[str, Any],
    *,
    yaml_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    sequences = []
    for sequence_entry in data.get("sequences", []):
        if not isinstance(sequence_entry, dict) or len(sequence_entry) != 1:
            raise ValueError(f"Invalid sequence entry: {sequence_entry}")
        entity_type = next(iter(sequence_entry))
        entity = sequence_entry[entity_type]
        if entity_type in {"protein", "rna", "dna"}:
            sequences.append(
                _convert_polymer_entity(
                    entity_type,
                    entity,
                    yaml_path=yaml_path,
                    output_path=output_path,
                )
            )
        elif entity_type == "ligand":
            sequences.append(_convert_ligand_entity(entity))
        else:
            raise ValueError(f"Unsupported entity type in YAML: {entity_type}")
    return sequences


def _convert_constraints(data: dict[str, Any]) -> list[list[list[Any]]] | None:
    bonds = []
    for constraint in data.get("constraints", []):
        if "bond" in constraint:
            bond = constraint["bond"]
            bonds.append([bond["atom1"], bond["atom2"]])
    return bonds or None


def _warn_and_ignore_templates(data: dict[str, Any]) -> None:
    templates = data.get("templates") or []
    if templates:
        print(
            "Warning: Boltz templates cannot be converted safely to AF3 templates "
            "because residue alignment mappings are not available. Templates were ignored.",
            file=sys.stderr,
        )


def convert_yaml_to_af3_json(
    data: dict[str, Any],
    *,
    yaml_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _warn_ignored_yaml_only_fields(data)
    _warn_and_ignore_templates(data)

    return {
        "dialect": "alphafold3",
        "version": 4,
        "name": output_path.stem,
        "sequences": _convert_sequences(
            data,
            yaml_path=yaml_path,
            output_path=output_path,
        ),
        "modelSeeds": [1],
        "bondedAtomPairs": _convert_constraints(data),
        "userCCD": None,
    }


def yamltojson(input_yaml: str, output_json: str) -> None:
    yaml_path = Path(input_yaml).resolve()
    output_path = Path(output_json).resolve()
    logger.info(f"Reading input YAML file: {yaml_path}")
    data = read_yaml_data(yaml_path)
    converted = convert_yaml_to_af3_json(
        data,
        yaml_path=yaml_path,
        output_path=output_path,
    )
    logger.info(f"Writing output JSON file: {output_path}")
    write_af3_json(output_path, converted)


def main():
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=(
            "Convert a Boltz YAML file into AlphaFold3-compatible input JSON."
        ),
    )
    add_version_option(parser)
    parser.add_argument(
        "-i",
        "--input_yaml",
        help="Input Boltz YAML file. Mandatory.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output AlphaFold3 JSON file. Mandatory.",
        type=str,
        required=True,
        metavar="output.json",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        dest="loglevel",
        action="store_const",
        const="DEBUG",
        default="SUCCESS",
    )
    args = parser.parse_args()
    log_setup(args.loglevel)
    yamltojson(args.input_yaml, args.out)


if __name__ == "__main__":
    main()
