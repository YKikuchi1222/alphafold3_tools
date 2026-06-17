#!/usr/bin/env python3
import copy
import re
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import Literal, cast

from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.modjson import (
    add_userccd,
    modify_name,
    purge_ligand,
    read_json_data,
    remove_ccdcodes,
)
from alphafold3tools.utils import add_version_option, int_id_to_str_id, write_af3_json

EntityType = Literal["smiles", "rna", "dna", "ccdCode"]


@dataclass(frozen=True)
class EntityAddition:
    entity_type: EntityType
    value: str


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _validate_entity_type(entity_type: str) -> EntityType:
    valid_types = {"smiles", "rna", "dna", "ccdCode"}
    if entity_type not in valid_types:
        raise ValueError(
            f"Invalid entity type: {entity_type}. "
            "The entity type must be one of 'smiles', 'rna', 'dna', or 'ccdCode'."
        )
    return cast(EntityType, entity_type)


def _count_existing_ids(data: dict) -> int:
    count = 0
    for sequence_content in data["sequences"]:
        for value in sequence_content.values():
            seq_id = value.get("id")
            if isinstance(seq_id, list):
                count += len(seq_id)
            elif isinstance(seq_id, str):
                count += 1
    return count


def _make_sequence_entry(entity_type: EntityType, value: str, seq_id: str) -> dict:
    if entity_type == "smiles":
        return {"ligand": {"id": seq_id, "smiles": value}}
    if entity_type == "ccdCode":
        return {"ligand": {"id": seq_id, "ccdCodes": [value]}}
    if entity_type == "rna":
        return {"rna": {"id": seq_id, "sequence": value, "modifications": []}}
    if entity_type == "dna":
        return {"dna": {"id": seq_id, "sequence": value, "modifications": []}}
    raise ValueError(f"Unsupported entity type: {entity_type}")


def add_entities(data: dict, entity_additions: list[EntityAddition]) -> dict:
    new_data = copy.deepcopy(data)
    sequence_contents = new_data["sequences"]
    id_counter = _count_existing_ids(new_data) + 1
    for entity_addition in entity_additions:
        seq_id = int_id_to_str_id(id_counter)
        logger.info(
            f"Adding {entity_addition.entity_type}: {entity_addition.value} as chain {seq_id}"
        )
        sequence_contents.append(
            _make_sequence_entry(
                entity_addition.entity_type,
                entity_addition.value,
                seq_id,
            )
        )
        id_counter += 1
    return new_data


def parse_entity_file(file_path: str) -> list[EntityAddition]:
    additions = []
    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            columns = re.split(r"\s+", stripped)
            if len(columns) > 2:
                raise ValueError(
                    f"Invalid entity file format at line {line_number}: {line.rstrip()}"
                )
            if len(columns) < 2:
                raise ValueError(
                    f"Missing value in entity file at line {line_number}: {line.rstrip()}"
                )
            entity_type = _validate_entity_type(columns[0])
            value = _strip_wrapping_quotes(columns[1])
            additions.append(EntityAddition(entity_type, value))
    return additions


def collect_entity_additions(argv: list[str]) -> list[EntityAddition]:
    additions: list[EntityAddition] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-a", "--add_entity"):
            if i + 2 >= len(argv):
                raise ValueError(f"Option {token} requires two arguments.")
            entity_type = _validate_entity_type(argv[i + 1])
            additions.append(EntityAddition(entity_type, argv[i + 2]))
            i += 3
            continue
        if token in ("-f", "--add_entity_file"):
            if i + 1 >= len(argv):
                raise ValueError(f"Option {token} requires a file path.")
            additions.extend(parse_entity_file(argv[i + 1]))
            i += 2
            continue
        i += 1
    return additions


def modjson3(
    input_json: str,
    output: str,
    entity_additions: list[EntityAddition] | None = None,
    purging: bool = False,
    ligands_to_be_removed: list[str] | None = None,
    name: str | None = None,
    userccd_to_be_added: list[str] | None = None,
) -> None:
    logger.info(f"Reading input JSON file: {input_json}")
    data = read_json_data(input_json)
    if purging:
        logger.info("Purging current ligand entities from the input JSON file.")
        data = purge_ligand(data)
    if ligands_to_be_removed:
        logger.info("Removing ligand entities from the input JSON file.")
        data = remove_ccdcodes(data, ligands_to_be_removed)
    if entity_additions:
        logger.info("Adding entities to the input JSON file.")
        data = add_entities(data, entity_additions)
    if name:
        logger.info(f"Setting the job name to: {name}")
        data = modify_name(data, name)
    if userccd_to_be_added:
        logger.info("Adding user provided ccdCodes to the input JSON file.")
        data = add_userccd(data, userccd_to_be_added)
    logger.info(f"Output JSON file: {output}")
    write_af3_json(output, data, indent=4)


def main():
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=(
            "Add rna/dna/smiles/ccdCode entities or remove ligands in an "
            "AlphaFold3 input JSON file."
        ),
    )
    add_version_option(parser)
    parser.add_argument(
        "-i",
        "--input_json",
        help="Input AlphaFold3 JSON file. Mandatory.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output JSON file. Mandatory.",
        type=str,
        required=True,
        metavar="output.json",
    )
    parser.add_argument(
        "-a",
        "--add_entity",
        help="Add an entity to the input JSON file.\n"
        "Provide 'entity type' and 'value'.\n"
        "The entity type must be one of 'smiles', 'rna', 'dna', or 'ccdCode'.\n"
        "Multiple entities can be added.\n"
        "e.g. -a smiles 'CCO' -a rna AGCU -a ccdCode ATP",
        type=str,
        nargs=2,
        action="append",
        dest="entities_to_be_added",
        metavar=("entity_type", "value"),
    )
    parser.add_argument(
        "-f",
        "--add_entity_file",
        help="Read entities to add from a two-column file with no header.\n"
        "Column 1: smiles/rna/dna/ccdCode\n"
        "Column 2: the corresponding value.\n"
        "Columns are split by one or more spaces.",
        type=str,
        action="append",
        dest="entity_files",
        metavar="file.txt",
    )
    parser.add_argument(
        "-p",
        "--purge_ligand",
        dest="purging",
        help="Purge all ligands from the input JSON file at first.",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--remove_ccdcodes",
        help="Remove ligands with ccdcodes from the input JSON file. Multiple ccdcodes "
        "can be provided.\n"
        "e.g. -r PRD ATP",
        dest="ligands_to_be_removed",
        type=str,
        nargs="*",
        metavar="ccdcode",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Set the job name in the input JSON file. i.e. data['name'] = name",
        type=str,
        metavar="new prediction name",
    )
    parser.add_argument(
        "-u",
        "--add_userccd",
        help="Add user provided ccdCodes to the input JSON file.\n"
        "Multiple files can be provided.\n"
        "e.g. -u userccd1.cif userccd2.cif",
        type=str,
        dest="userccd_to_be_added",
        nargs="*",
        metavar="userccd_file",
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
    entity_additions = collect_entity_additions(sys.argv[1:])
    modjson3(
        args.input_json,
        args.out,
        entity_additions=entity_additions,
        purging=args.purging,
        ligands_to_be_removed=args.ligands_to_be_removed,
        name=args.name,
        userccd_to_be_added=args.userccd_to_be_added,
    )


if __name__ == "__main__":
    main()
