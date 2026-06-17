import concurrent.futures
import datetime
import os
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.msatojson import (
    Seq,
    convert_msas_to_str,
    get_paired_and_unpaired_msa,
    get_residuelens_stoichiometries,
    search_templates,
    to_json,
)
from alphafold3tools.utils import add_version_option, int_id_to_str_id


def _write_msa_file(path: Path, msas: list[Seq]) -> None:
    path.write_text(convert_msas_to_str(msas))


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)


def generate_input_json_content(
    name: str,
    cardinality: int,
    stoichiometries: list[int],
    pairedmsas: list[list[Seq]],
    unpairedmsas: list[list[Seq]],
    msa_output_dir: Path,
    json_parent_dir: Path,
    includetemplates: bool = False,
    savehmmsto: bool = False,
    pdb_database_path: str | os.PathLike[str] | None = None,
    seqres_database_path: str | os.PathLike[str] | None = None,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    max_subsequence_ratio: float | None = 0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
) -> dict[str, Any]:
    sequences = []
    chain_id_count = 0
    msa_output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(cardinality):
        query_seq = unpairedmsas[i][0].sequence
        chain_ids = [
            int_id_to_str_id(chain_id_count + j + 1) for j in range(stoichiometries[i])
        ]
        chain_id_count += stoichiometries[i]

        unpaired_path = msa_output_dir / f"chain_{i + 1}_unpaired.a3m"
        _write_msa_file(unpaired_path, unpairedmsas[i])

        protein: dict[str, Any] = {
            "id": chain_ids,
            "sequence": query_seq,
            "modifications": [],
            "unpairedMsaPath": _relative_path(unpaired_path, json_parent_dir),
        }

        if pairedmsas[i]:
            paired_path = msa_output_dir / f"chain_{i + 1}_paired.a3m"
            _write_msa_file(paired_path, pairedmsas[i])
            protein["pairedMsaPath"] = _relative_path(paired_path, json_parent_dir)
        else:
            protein["pairedMsa"] = ""

        if includetemplates:
            logger.info(
                f"Searching templates for chain {i + 1} with sequence length {len(query_seq)}..."
            )
            templates_list = search_templates(
                msa_a3m_string=convert_msas_to_str(unpairedmsas[i]),
                pdb_database_path=pdb_database_path,
                seqres_database_path=seqres_database_path,
                savehmmsto=savehmmsto,
                max_template_date=max_template_date,
                max_subsequence_ratio=max_subsequence_ratio,
                hmmbuild_binary_path=hmmbuild_binary_path,
                hmmsearch_binary_path=hmmsearch_binary_path,
            )
        else:
            templates_list = []
        protein["templates"] = templates_list
        sequences.append({"protein": protein})

    return {
        "dialect": "alphafold3",
        "version": 4,
        "name": f"{name}",
        "sequences": sequences,
        "modelSeeds": [1],
        "bondedAtomPairs": None,
        "userCCD": None,
    }


def write_input_json_file(
    inputmsafile: str | Path,
    name: str,
    outputjsonfile: str | Path,
    includetemplates: bool = False,
    savehmmsto: bool = False,
    pdb_database_path: str | os.PathLike[str] | None = None,
    seqres_database_path: str | os.PathLike[str] | None = None,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    max_subsequence_ratio: float | None = 0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
) -> None:
    inputmsafile = Path(inputmsafile)
    outputjsonfile = Path(outputjsonfile)
    with inputmsafile.open("r") as f:
        lines = f.readlines()
    residue_lens, stoichiometries = get_residuelens_stoichiometries(lines)
    if len(residue_lens) != len(stoichiometries):
        raise ValueError("Length of residue_lens and stoichiometries must be the same.")
    cardinality = len(residue_lens)
    logger.info(
        f"The input MSA file contains {cardinality} distinct polypeptide chains."
    )
    logger.info(f"Residue lengths: {residue_lens}")
    logger.info(f"Stoichiometries: {stoichiometries}")
    pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
        lines, residue_lens, cardinality
    )
    msa_output_dir = outputjsonfile.parent / f"{outputjsonfile.stem}_msas"
    content = generate_input_json_content(
        name=f"{name}",
        cardinality=cardinality,
        stoichiometries=stoichiometries,
        pairedmsas=pairedmsas,
        unpairedmsas=unpairedmsas,
        msa_output_dir=msa_output_dir,
        json_parent_dir=outputjsonfile.parent,
        includetemplates=includetemplates,
        savehmmsto=savehmmsto,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=max_template_date,
        max_subsequence_ratio=max_subsequence_ratio,
        hmmbuild_binary_path=hmmbuild_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
    )
    outputjsonfile.write_text(to_json(content))


def _process_a3m_file(
    a3m_file: Path,
    output_dir: Path,
    includetemplates: bool,
    savehmmsto: bool,
    pdb_database_path: str | os.PathLike[str] | None,
    seqres_database_path: str | os.PathLike[str] | None,
    max_template_date: datetime.date,
    max_subsequence_ratio: float | None,
    hmmbuild_binary_path: str | None,
    hmmsearch_binary_path: str | None,
) -> None:
    name = a3m_file.stem
    output_file = output_dir / f"{name}.json"
    write_input_json_file(
        inputmsafile=a3m_file,
        name=name,
        outputjsonfile=output_file,
        includetemplates=includetemplates,
        savehmmsto=savehmmsto,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=max_template_date,
        max_subsequence_ratio=max_subsequence_ratio,
        hmmbuild_binary_path=hmmbuild_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
    )


def process_a3m_directory(
    input_dir: Path,
    output_dir: Path,
    includetemplates: bool,
    savehmmsto: bool,
    pdb_database_path: str | os.PathLike[str] | None,
    seqres_database_path: str | os.PathLike[str] | None,
    max_template_date: datetime.date,
    max_subsequence_ratio: float | None,
    hmmbuild_binary_path: str | None,
    hmmsearch_binary_path: str | None,
) -> None:
    if output_dir.suffix == ".json":
        raise ValueError(
            "Now the input is directory, so output name must be a directory."
        )
    logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    a3m_files = list(input_dir.glob("*.a3m"))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _process_a3m_file,
                a3m_file,
                output_dir,
                includetemplates,
                savehmmsto,
                pdb_database_path,
                seqres_database_path,
                max_template_date,
                max_subsequence_ratio,
                hmmbuild_binary_path,
                hmmsearch_binary_path,
            )
            for a3m_file in a3m_files
        ]
        concurrent.futures.wait(futures)


def process_single_a3m_file(
    inputmsafile: Path,
    outputjsonfile: Path,
    name: str | None = None,
    includetemplates: bool = False,
    savehmmsto: bool = False,
    pdb_database_path: str | os.PathLike[str] | None = None,
    seqres_database_path: str | os.PathLike[str] | None = None,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    max_subsequence_ratio: float | None = 0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
) -> None:
    if inputmsafile.suffix != ".a3m":
        raise ValueError("Input file must have .a3m extension.")
    logger.info(f"Input A3M file: {inputmsafile}")
    if outputjsonfile.suffix != ".json":
        raise ValueError("Output file must have .json extension.")
    logger.info(f"Output JSON file: {outputjsonfile}")
    write_input_json_file(
        inputmsafile=inputmsafile,
        name=name or inputmsafile.stem,
        outputjsonfile=outputjsonfile,
        includetemplates=includetemplates,
        savehmmsto=savehmmsto,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=max_template_date,
        max_subsequence_ratio=max_subsequence_ratio,
        hmmbuild_binary_path=hmmbuild_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
    )


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "Converts a3m-format MSA file to AlphaFold3 v4 input JSON file "
            "using per-chain MSA path fields."
        ),
    )
    add_version_option(parser)
    parser.add_argument(
        "-i",
        "--input",
        help="Input A3M file or directory containing A3M files. e.g. 1bjp.a3m",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Name of the protein complex.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-o", "--out", help="Output directory or JSON file.", type=str, required=True
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
    parser.add_argument(
        "--include_templates",
        help="Include template search results in the output JSON file.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_hmmsto",
        help="Save intermediate HMM sto files used for template search.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pdb_database_path",
        help="Path to the PDB mmCIF database for template search.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seqres_database_path",
        help="Path to the PDB SEQRES database for template search.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_template_date",
        help="Maximum template date for template search in YYYY-MM-DD format.",
        type=lambda s: datetime.date.fromisoformat(s),
        default=datetime.date(2099, 12, 31),
    )
    parser.add_argument(
        "--max_subsequence_ratio",
        help="Maximum subsequence ratio for template search. "
        "If set to 1.0, no templates will be excluded based on subsequence ratio. "
        "Default is 0.95.",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--hmmbuild_binary_path",
        help="Path to the hmmbuild binary. Default is to use the hmmbuild in PATH.",
        type=str,
        default=shutil.which("hmmbuild"),
    )
    parser.add_argument(
        "--hmmsearch_binary_path",
        help="Path to the hmmsearch binary. Default is to use the hmmsearch in PATH.",
        type=str,
        default=shutil.which("hmmsearch"),
    )
    args = parser.parse_args()
    log_setup(args.loglevel)
    input_path = Path(args.input)
    output_path = Path(args.out)

    if input_path.is_dir():
        process_a3m_directory(
            input_dir=input_path,
            output_dir=output_path,
            includetemplates=args.include_templates,
            savehmmsto=args.save_hmmsto,
            pdb_database_path=args.pdb_database_path,
            seqres_database_path=args.seqres_database_path,
            max_template_date=args.max_template_date,
            max_subsequence_ratio=args.max_subsequence_ratio,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            hmmsearch_binary_path=args.hmmsearch_binary_path,
        )
    else:
        process_single_a3m_file(
            inputmsafile=input_path,
            outputjsonfile=output_path,
            name=args.name or input_path.stem,
            includetemplates=args.include_templates,
            savehmmsto=args.save_hmmsto,
            pdb_database_path=args.pdb_database_path,
            seqres_database_path=args.seqres_database_path,
            max_template_date=args.max_template_date,
            max_subsequence_ratio=args.max_subsequence_ratio,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            hmmsearch_binary_path=args.hmmsearch_binary_path,
        )
