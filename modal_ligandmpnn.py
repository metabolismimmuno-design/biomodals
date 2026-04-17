# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
from __future__ import annotations
"""LigandMPNN (superseding ProteinMPNN) for protein design on Modal.

LigandMPNN: https://github.com/dauparas/LigandMPNN

- By default, calc_score is False, because it's quite slow.

## Example EGFR binder
- Design chain C but include chains A and C

```
modal run modal_ligandmpnn.py --input-pdb in/ligandmpnn/1IVO_edited.pdb --extract-chains AC \
--params-str '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt" \
--chains_to_design "C" --save_stats 1'
```

## Example EGFR binder
- Outputs will have only chain C
- 15 sequences total (3x5)

```
modal run modal_ligandmpnn.py --input-pdb in/ligandmpnn/1IVO_edited.pdb \
--params-str '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt" \
--parse_these_chains_only "C" --save_stats 1 --batch_size 3 --number_of_batches 5'
```

"""

import os
from pathlib import Path
from typing import Optional

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

DEFAULT_PARAMS = '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt" --save_stats 1'

image = (
    Image.micromamba(python_version="3.11")
    .apt_install(["git", "wget", "gcc", "g++", "libffi-dev"])
    .pip_install(
        [
            "biopython==1.79",
            "filelock==3.13.1",
            "fsspec==2024.3.1",
            "Jinja2==3.1.3",
            "MarkupSafe==2.1.5",
            "mpmath==1.3.0",
            "networkx==3.2.1",
            "numpy==1.23.5",
        ]
    )
    .pip_install(
        [
            "nvidia-cublas-cu12==12.1.3.1",
            "nvidia-cuda-cupti-cu12==12.1.105",
            "nvidia-cuda-nvrtc-cu12==12.1.105",
            "nvidia-cuda-runtime-cu12==12.1.105",
            "nvidia-cudnn-cu12==8.9.2.26",
            "nvidia-cufft-cu12==11.0.2.54",
            "nvidia-curand-cu12==10.3.2.106",
            "nvidia-cusolver-cu12==11.4.5.107",
            "nvidia-cusparse-cu12==12.1.0.106",
            "nvidia-nccl-cu12==2.19.3",
            "nvidia-nvjitlink-cu12==12.4.99",
            "nvidia-nvtx-cu12==12.1.105",
        ]
    )
    .pip_install(
        [
            "ProDy==2.4.1",
            "pyparsing==3.1.1",
            "scipy==1.12.0",
            "sympy==1.12",
            "torch==2.2.1",
            "triton==2.2.0",
            "typing_extensions==4.10.0",
            "ml-collections==0.1.1",
            "dm-tree==0.1.8",
        ]
    )
    .run_commands(
        "git clone https://github.com/dauparas/LigandMPNN.git /LigandMPNN"
        " && cd /LigandMPNN"
        ' && bash get_model_params.sh "./model_params"'
        # AbMPNN: ProteinMPNN fine-tuned on SAbDab antibody structures (Exscientia, arXiv:2310.19513).
        # 20 MB weights downloaded from Zenodo at image build time.
        ' && wget -q "https://zenodo.org/record/8164693/files/abmpnn.pt"'
        ' -O "/LigandMPNN/model_params/abmpnn.pt"'
    )
)

app = App("LigandMPNN", image=image)


def extract_chains_inplace(pdb_file: str, extract_chains: str) -> str:
    """Extract specific chains from a PDB file in-place.
    
    Args:
        pdb_file (str): Path to the PDB file.
        extract_chains (str): Chain identifiers to extract.
        
    Returns:
        str: Path to the modified PDB file.
    """
    from prody import parsePDB, writePDB

    chains = parsePDB(pdb_file, chain=extract_chains.replace(",", ""))
    writePDB(pdb_file, chains)
    return pdb_file


# 松约束：只固定骆驼科签名位点（结构核心 + VHH 单域稳定性关键位点）
# 不依赖 CDR 边界，适用于允许 FR 有一定变化但保留折叠必需残基的场景
# 注意：这些是顺序编号（PDB residue number），近似对应 Kabat 22/37/44/45/47/92
VHH_LOOSE_FIXED = [22, 37, 44, 45, 47, 92]

# 紧约束：动态计算 FR = 所有 vhh_chain 残基 − CDR 位点
# 需要用户通过 cdr_positions 传入 CDR 位点（JSON 格式），适用于严格保持 VHH 框架保守性的场景
# 示例：'{"B": [26,27,28,29,30,31,32,33,52,53,54,55,56,57,58,98,99,100,101,102,103,104,105,106,107,108]}'


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def ligandmpnn(
    input_pdb_str: str,
    input_pdb_name: str,
    params_str: Optional[str] = None,
    calc_score: bool = False,
    score_params_str: Optional[str] = None,
    extract_chains: Optional[str] = None,
    vhh_framework_mode: Optional[str] = None,
    cdr_positions: Optional[str] = None,
    vhh_chain: str = "A",
) -> list:
    """Runs LigandMPNN on a PDB input string.

    Args:
        input_pdb_str (str): PDB file content as a string.
        input_pdb_name (str): Name for the PDB file.
        params_str (str | None): Optional string of additional parameters for LigandMPNN.
        calc_score (bool): Whether to calculate scores for the output.
        score_params_str (str | None): Optional parameters for scoring.
        extract_chains (str | None): Chain identifiers to extract from the PDB.
        vhh_framework_mode (str | None): VHH framework constraint mode.
            "loose" — fix camelid signature positions only (22, 37, 44, 45, 47, 92 on vhh_chain).
                      No cdr_positions needed.
            "strict" — fix all FR positions = (all vhh_chain residues) − CDR positions.
                       Requires cdr_positions to be provided.
            None — no framework constraints (default, all positions free).
        cdr_positions (str | None): JSON string specifying CDR residue numbers per chain.
            Required for strict mode. Key must match vhh_chain. Example (VHH on chain B):
            '{"B": [26,27,28,29,30,31,32,33,52,53,54,55,56,57,58,98,99,100,101,102,103,104,105,106,107,108]}'
        vhh_chain (str): Chain ID of the VHH in the PDB. Default "A". Set to "B" when
            VHH comes from RFdiffusion output (chain B = binder).

    Returns:
        list[tuple[Path, bytes]]: A list of tuples containing output file paths and their byte content.
    """
    import json
    from subprocess import run

    out_dir = "./out"

    open(input_pdb_name, "w").write(input_pdb_str)
    if extract_chains is not None:
        input_pdb_name = extract_chains_inplace(input_pdb_name, extract_chains)

    # VHH framework constraint: generate fixed_positions_jsonl if mode is specified
    if vhh_framework_mode is not None:
        if vhh_framework_mode not in ("loose", "strict"):
            raise ValueError(f"vhh_framework_mode must be 'loose', 'strict', or None. Got: {vhh_framework_mode!r}")

        if vhh_framework_mode == "loose":
            fixed_positions = VHH_LOOSE_FIXED

        else:  # strict: FR = all vhh_chain residues − CDR positions
            if cdr_positions is None:
                raise ValueError("vhh_framework_mode='strict' requires cdr_positions (JSON string of CDR residue numbers).")

            from Bio.PDB import PDBParser

            cdr_set = set(json.loads(cdr_positions).get(vhh_chain, []))

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("nb", input_pdb_name)
            all_vhh_residues = set()
            for model in structure:
                if vhh_chain in model:
                    for residue in model[vhh_chain]:
                        if residue.id[0] == " ":
                            all_vhh_residues.add(residue.id[1])
                break

            fixed_positions = sorted(all_vhh_residues - cdr_set)
            print(f"[strict] chain {vhh_chain} total={len(all_vhh_residues)}, CDR={len(cdr_set)}, FR fixed={len(fixed_positions)}")

        # Use --fixed_residues flag (--fixed_positions_jsonl is not supported in current LigandMPNN)
        fixed_residues_str = " ".join(f"{vhh_chain}{pos}" for pos in fixed_positions)
        extra = f' --fixed_residues "{fixed_residues_str}"'
        params_str = (params_str if params_str is not None else DEFAULT_PARAMS) + extra
        print(f"[VHH framework mode: {vhh_framework_mode}] fixed {len(fixed_positions)} positions on chain {vhh_chain}")

    # --------------------------------------------------------------------------
    # Run LigandMPNN
    # By default, use a protein model
    if params_str is None:
        params_str = DEFAULT_PARAMS

    cmd = f'python /LigandMPNN/run.py --pdb_path "{input_pdb_name}" --out_folder "{out_dir}" {params_str}'
    print(cmd)
    run(cmd, shell=True, check=True)

    # --------------------------------------------------------------------------
    # Score the output from LigandMPNN
    # Defaults from https://github.com/dauparas/LigandMPNN, not sure what some of these do
    #
    if calc_score:
        if score_params_str is None:
            score_params_str = (
                f' --seed 111 --model_type "protein_mpnn" --checkpoint_protein_mpnn {ckpt}'
                " --single_aa_score 1 --use_sequence 1 --batch_size 1 --number_of_batches 10"
            )

        for backbone in (Path(out_dir) / "backbones").glob("*.pdb"):
            score_params_str_ = score_params_str + f' --pdb_path "{backbone}"'

            cmd_score = f'python /LigandMPNN/score.py --out_folder "{out_dir}" {score_params_str_}'
            print(cmd_score)
            run(cmd_score, shell=True, check=True)

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*")
        if Path(out_file).is_file()
    ]


@app.local_entrypoint()
def main(
    input_pdb: str,
    params_str: Optional[str] = None,
    calc_score: bool = False,
    score_params_str: Optional[str] = None,
    extract_chains: Optional[str] = None,
    run_name: Optional[str] = None,
    out_dir: str = "./out/ligandmpnn",
    vhh_framework_mode: Optional[str] = None,
    cdr_positions: Optional[str] = None,
    vhh_chain: str = "A",
):
    """Local entrypoint to run LigandMPNN predictions using Modal.

    Args:
        input_pdb (str): Path to an input PDB file.
        params_str (str | None): Optional string of additional parameters for LigandMPNN.
        calc_score (bool): Whether to calculate scores for the output.
        score_params_str (str | None): Optional parameters for scoring.
        extract_chains (str | None): Chain identifiers to extract from the PDB.
        run_name (str | None): Optional name for the run, used for organizing output files.
                               If None, a timestamp-based name is used.
        out_dir (str): Directory to save the output files. Defaults to "./out/ligandmpnn".
        vhh_framework_mode (str | None): VHH framework constraint mode.
            "loose" — fix camelid signature positions only (22, 37, 44, 45, 47, 92 on vhh_chain).
            "strict" — fix all FR = (all vhh_chain residues) − CDR positions. Requires cdr_positions.
            None — no framework constraints (default).
        cdr_positions (str | None): JSON string of CDR residue numbers per chain. Required for strict mode.
            Key must match vhh_chain. Example (VHH on chain B): '{"B": [26,27,28,...,108]}'
        vhh_chain (str): Chain ID of the VHH in the PDB. Default "A". Use "B" for RFdiffusion output.

    Returns:
        None
    """
    from datetime import datetime

    input_pdb_str = open(input_pdb).read()

    outputs = ligandmpnn.remote(
        input_pdb_str, Path(input_pdb).name, params_str, calc_score, score_params_str, extract_chains,
        vhh_framework_mode, cdr_positions, vhh_chain,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        with open((Path(out_dir_full) / Path(out_file)), "wb") as out:
            out.write(out_content)
