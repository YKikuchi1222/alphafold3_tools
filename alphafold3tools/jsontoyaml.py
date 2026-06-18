from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Any

from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.modjson import read_json_data
from alphafold3tools.utils import add_version_option


BLOCK_LIST_KEYS = {"sequences", "constraints", "templates", "properties", "modifications"}
INLINE_LIST_KEYS = {
    "id",
    "atom1",
    "atom2",
    "token1",
    "token2",
    "contacts",
    "chain_id",
    "template_id",
    "ccd",
}


def _resolve_input_path(raw_path: str, json_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (json_path.parent / path).resolve()


def _to_output_relative(raw_path: str, json_path: Path, output_path: Path) -> str:
    resolved = _resolve_input_path(raw_path, json_path)
    return os.path.relpath(resolved, output_path.parent)


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _quote_yaml_string(value: str, *, force_single: bool = False) -> str:
    if force_single:
        return "'" + value.replace("'", "''") + "'"
    if value == "":
        return "''"
    plain_safe_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:-+"
    )
    if set(value) <= plain_safe_chars and not value.lower() in {"null", "true", "false"}:
        return value
    return "'" + value.replace("'", "''") + "'"


def _format_inline_scalar(value: Any, key: str | None = None) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    force_single = key == "smiles"
    return _quote_yaml_string(str(value), force_single=force_single)


def _is_inline_list(value: list[Any], key: str | None = None) -> bool:
    if key in INLINE_LIST_KEYS:
        return True
    return all(
        not isinstance(item, (dict, list)) or (
            isinstance(item, list) and all(not isinstance(sub, (dict, list)) for sub in item)
        )
        for item in value
    )


def _format_inline_list(value: list[Any], key: str | None = None) -> str:
    items = []
    for item in value:
        if isinstance(item, list):
            items.append(_format_inline_list(item))
        else:
            items.append(_format_inline_scalar(item, key=key))
    return f"[ {', '.join(items)} ]"


def _yaml_lines(value: Any, indent: int = 0, key: str | None = None) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for child_key, child_value in value.items():
            if isinstance(child_value, dict):
                lines.append(f"{prefix}{child_key}:")
                lines.extend(_yaml_lines(child_value, indent + 2, key=child_key))
            elif isinstance(child_value, list):
                if not child_value:
                    lines.append(f"{prefix}{child_key}: []")
                elif _is_inline_list(child_value, key=child_key):
                    lines.append(f"{prefix}{child_key}: {_format_inline_list(child_value, key=child_key)}")
                else:
                    lines.append(f"{prefix}{child_key}:")
                    for item in child_value:
                        if isinstance(item, dict):
                            if len(item) == 1:
                                first_key = next(iter(item))
                                first_value = item[first_key]
                                if isinstance(first_value, dict):
                                    lines.append(f"{prefix}  - {first_key}:")
                                    lines.extend(_yaml_lines(first_value, indent + 6, key=first_key))
                                else:
                                    lines.append(f"{prefix}  - {first_key}: {_format_inline_scalar(first_value, key=first_key)}")
                            else:
                                first_key = next(iter(item))
                                lines.append(f"{prefix}  - {first_key}: {_format_inline_scalar(item[first_key], key=first_key)}")
                                for extra_key, extra_value in list(item.items())[1:]:
                                    lines.append(f"{prefix}    {extra_key}: {_format_inline_scalar(extra_value, key=extra_key)}")
                        else:
                            lines.append(f"{prefix}  - {_format_inline_scalar(item, key=child_key)}")
            else:
                lines.append(f"{prefix}{child_key}: {_format_inline_scalar(child_value, key=child_key)}")
        return lines
    if isinstance(value, list):
        if _is_inline_list(value, key=key):
            return [f"{prefix}{_format_inline_list(value, key=key)}"]
        lines: list[str] = []
        for item in value:
            lines.append(f"{prefix}- {_format_inline_scalar(item, key=key)}")
        return lines
    return [f"{prefix}{_format_inline_scalar(value, key=key)}"]


def _convert_modifications(entity_type: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    modifications = data.get("modifications", [])
    if entity_type == "protein":
        return [
            {"position": mod["ptmPosition"], "ccd": mod["ptmType"]}
            for mod in modifications
        ]
    return [
        {"position": mod["basePosition"], "ccd": mod["modificationType"]}
        for mod in modifications
    ]


def _choose_protein_msa_path(
    protein: dict[str, Any],
    *,
    chain_label: str,
    json_path: Path,
    output_path: Path,
    msa_dir: Path,
) -> str | None:
    unpaired_path = protein.get("unpairedMsaPath")
    paired_path = protein.get("pairedMsaPath")
    if unpaired_path:
        return _to_output_relative(unpaired_path, json_path, output_path)
    if paired_path:
        logger.warning(
            f"Chain {chain_label}: using pairedMsaPath because unpairedMsaPath is absent."
        )
        return _to_output_relative(paired_path, json_path, output_path)

    unpaired_msa = protein.get("unpairedMsa")
    paired_msa = protein.get("pairedMsa")
    if unpaired_msa:
        msa_path = msa_dir / f"{chain_label}.a3m"
        _write_text_file(msa_path, unpaired_msa)
        return os.path.relpath(msa_path, output_path.parent)
    if paired_msa:
        logger.warning(
            f"Chain {chain_label}: using pairedMsa text because unpairedMsa is absent."
        )
        msa_path = msa_dir / f"{chain_label}.a3m"
        _write_text_file(msa_path, paired_msa)
        return os.path.relpath(msa_path, output_path.parent)
    if unpaired_msa == "" or paired_msa == "":
        return "empty"
    return None


def _convert_sequence_entry(
    sequence_entry: dict[str, Any],
    *,
    json_path: Path,
    output_path: Path,
    msa_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entity_type = next(iter(sequence_entry))
    entity = sequence_entry[entity_type]
    out: dict[str, Any] = {"id": entity["id"]}
    templates: list[dict[str, Any]] = []

    if entity_type in {"protein", "rna", "dna"}:
        out["sequence"] = entity["sequence"]
        modifications = _convert_modifications(entity_type, entity)
        if modifications:
            out["modifications"] = modifications
        if entity.get("cyclic"):
            out["cyclic"] = True

    if entity_type == "protein":
        chain_label = (
            entity["id"][0] if isinstance(entity["id"], list) else entity["id"]
        )
        msa_path = _choose_protein_msa_path(
            entity,
            chain_label=chain_label,
            json_path=json_path,
            output_path=output_path,
            msa_dir=msa_dir,
        )
        if msa_path is not None:
            out["msa"] = msa_path
        for idx, template in enumerate(entity.get("templates", []), start=1):
            template_record: dict[str, Any] = {}
            if template.get("mmcifPath"):
                template_record["cif"] = _to_output_relative(
                    template["mmcifPath"], json_path, output_path
                )
            elif template.get("mmcif"):
                template_dir = output_path.parent / f"{output_path.stem}_templates"
                template_dir.mkdir(parents=True, exist_ok=True)
                template_path = template_dir / f"{chain_label}_{idx}.cif"
                _write_text_file(template_path, template["mmcif"])
                template_record["cif"] = os.path.relpath(template_path, output_path.parent)
            else:
                continue
            template_record["chain_id"] = entity["id"]
            templates.append(template_record)

    elif entity_type == "ligand":
        if "smiles" in entity:
            out["smiles"] = entity["smiles"]
        elif "ccdCodes" in entity:
            ccd_codes = entity["ccdCodes"]
            out["ccd"] = ccd_codes[0] if len(ccd_codes) == 1 else ccd_codes
        else:
            raise ValueError("Ligand must contain smiles or ccdCodes.")

    return {entity_type: out}, templates


def _convert_constraints(data: dict[str, Any]) -> list[dict[str, Any]]:
    bonded_atom_pairs = data.get("bondedAtomPairs") or []
    constraints = []
    for bond in bonded_atom_pairs:
        atom1, atom2 = bond
        constraints.append(
            {
                "bond": {
                    "atom1": atom1,
                    "atom2": atom2,
                }
            }
        )
    return constraints


def _warn_ignored_af3_only_fields(data: dict[str, Any]) -> None:
    ignored_fields = []
    if data.get("userCCD") is not None:
        ignored_fields.append("userCCD")
    if data.get("userCCDPath") is not None:
        ignored_fields.append("userCCDPath")
    if ignored_fields:
        print(
            "Warning: Boltz YAML does not support "
            + ", ".join(ignored_fields)
            + ". These field(s) were ignored in the YAML output.",
            file=sys.stderr,
        )


def convert_json_to_boltz_schema(
    data: dict[str, Any],
    *,
    json_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {"version": 1, "sequences": []}
    templates: list[dict[str, Any]] = []
    msa_dir = output_path.parent / f"{output_path.stem}_msas"

    for sequence_entry in data["sequences"]:
        converted_entry, entry_templates = _convert_sequence_entry(
            sequence_entry,
            json_path=json_path,
            output_path=output_path,
            msa_dir=msa_dir,
        )
        output["sequences"].append(converted_entry)
        templates.extend(entry_templates)

    constraints = _convert_constraints(data)
    if constraints:
        output["constraints"] = constraints
    if templates:
        output["templates"] = templates
    return output


def boltz_yaml_dumps(data: dict[str, Any]) -> str:
    return "\n".join(_yaml_lines(data)) + "\n"


def jsontoyaml(input_json: str, output_yaml: str) -> None:
    json_path = Path(input_json).resolve()
    output_path = Path(output_yaml).resolve()
    logger.info(f"Reading input JSON file: {json_path}")
    data = read_json_data(str(json_path))
    _warn_ignored_af3_only_fields(data)
    converted = convert_json_to_boltz_schema(
        data,
        json_path=json_path,
        output_path=output_path,
    )
    logger.info(f"Writing output YAML file: {output_path}")
    output_path.write_text(boltz_yaml_dumps(converted))


def main():
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=(
            "Convert a completed AlphaFold3 input JSON into Boltz-compatible YAML."
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
        help="Output Boltz YAML file. Mandatory.",
        type=str,
        required=True,
        metavar="output.yaml",
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
    jsontoyaml(args.input_json, args.out)


if __name__ == "__main__":
    main()
