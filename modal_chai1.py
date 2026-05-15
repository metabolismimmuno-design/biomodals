# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
from __future__ import annotations

"""Runs Chai-1, a protein-ligand co-folding model, on Modal.

Chai-1r: https://github.com/chaidiscovery/chai-lab

Example fasta
```
>protein|name=insulin
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS
>ligand|name=caffeine
CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```
```
modal run modal_chai1.py --input-faa test_chai1.faa
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))


def download_models():
    """Downloads Chai-1 models by running a minimal inference.

    Args:
        None

    Returns:
        None
    """
    import torch
    from tempfile import TemporaryDirectory
    from chai_lab.chai1 import run_inference

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        with open(f"{td_in}/tmp.faa", "w") as out:
            out.write(
                ">protein|name=pro\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSL\n"
                ">ligand|name=lig\nCC\n"
            )

        _ = run_inference(
            fasta_file=Path(f"{td_in}/tmp.faa"),
            output_dir=Path(f"{td_out}"),
            num_trunk_recycles=1,
            num_diffn_timesteps=10,
            seed=1,
            device=torch.device("cuda:0"),
            use_esm_embeddings=True,
        )


image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .uv_pip_install("chai_lab==0.6.1")
    .run_function(download_models, gpu="a100")
)

app = App("chai1", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def chai1(
    input_faa_str: str,
    input_faa_name: str = "input.faa",
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_a3m_by_seq: dict | None = None,
    chai1_kwargs: dict = {},
) -> list:
    """Runs Chai-1 on a FASTA file string and returns the output files.

    Args:
        input_faa_str (str): Content of the input FASTA file as a string.
        input_faa_name (str): Original name of the input FASTA file (for naming outputs).
                              Defaults to "input.faa".
        num_trunk_recycles (int): Number of trunk recycles for the model. Defaults to 3.
        num_diffn_timesteps (int): Number of diffusion timesteps. Defaults to 200.
        seed (int): Random seed for reproducibility. Defaults to 42.
        use_esm_embeddings (bool): Whether to use ESM embeddings. Defaults to True.
        use_msa_server (bool): Whether to fetch MSA from ColabFold server. Defaults to False
                               (ESM-only). Set True to enable MSA-conditioned prediction —
                               critical for accurate antigen folding (raises antigen pTM
                               typically from ~0.4 ESM-only to ~0.7+ with MSA).
        msa_a3m_by_seq (dict | None): Per-chain pre-computed MSA. Keys are chain
                              protein sequences (case-insensitive, uppercased internally);
                              values are a3m file content as string. For each entry,
                              chai_lab's `merge_a3m_in_directory` converts a3m → aligned.pqt
                              keyed by sha256(seq.upper()), which `run_inference` picks up
                              via msa_directory. **Chains absent from the dict fall back to
                              ESM** — this is the natural mechanism for "antigen with MSA,
                              VHH single-sequence". Mutually exclusive with `use_msa_server=True`.
                              Defaults to None.
        chai1_kwargs (dict): Additional keyword arguments to pass to `run_inference`.
                             Defaults to an empty dict. Note: do not duplicate
                             `use_msa_server` here if passed explicitly above.

    Returns:
        list[tuple[Path, bytes]]: A list of tuples, where each tuple contains the relative
                                  output file path and its byte content.
    """
    import torch
    from tempfile import TemporaryDirectory
    from chai_lab.chai1 import run_inference

    # 防止 escape hatch 与一等参数重复：若 chai1_kwargs 显式带 use_msa_server，则覆盖外层
    kw = dict(chai1_kwargs) if chai1_kwargs else {}
    if "use_msa_server" in kw:
        use_msa_server = kw.pop("use_msa_server")

    if msa_a3m_by_seq and use_msa_server:
        raise ValueError(
            "msa_a3m_by_seq and use_msa_server=True are mutually exclusive — "
            "Chai-1 accepts either pre-computed msa_directory OR live ColabFold."
        )

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out, TemporaryDirectory() as td_msa:
        fasta_path = Path(td_in) / input_faa_name
        fasta_path.write_text(input_faa_str)

        msa_directory = None
        if msa_a3m_by_seq:
            from chai_lab.data.parsing.msas.aligned_pqt import merge_a3m_in_directory
            msa_dir = Path(td_msa)
            for i, (seq, a3m_str) in enumerate(msa_a3m_by_seq.items()):
                if not a3m_str or not a3m_str.strip():
                    print(f"[MSA] seq#{i} ({seq[:20]}...): empty a3m, skipping", flush=True)
                    continue
                # Per-seq staging subdir; merge_a3m_in_directory will produce
                # <sha256(seq.upper())>.aligned.pqt into msa_dir.
                stage = Path(td_msa) / f"_stage_{i}"
                stage.mkdir()
                (stage / "hits_uniref90.a3m").write_text(a3m_str)
                merge_a3m_in_directory(str(stage), output_directory=str(msa_dir))
                print(f"[MSA] seq#{i} ({seq[:20]}...): wrote pqt for {len(a3m_str)} chars a3m", flush=True)
            pqt_files = list(msa_dir.glob("*.aligned.pqt"))
            print(f"[MSA] msa_directory={msa_dir} contains {len(pqt_files)} pqt file(s)", flush=True)
            msa_directory = msa_dir

        run_kwargs = dict(
            fasta_file=Path(fasta_path),
            output_dir=Path(td_out),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            seed=seed,
            device=torch.device("cuda:0"),
            use_esm_embeddings=use_esm_embeddings,
            use_msa_server=use_msa_server,
        )
        if msa_directory is not None:
            run_kwargs["msa_directory"] = msa_directory
        run_kwargs.update(kw)

        _ = run_inference(**run_kwargs)

        return [
            (out_file.relative_to(td_out), open(out_file, "rb").read())
            for out_file in Path(td_out).glob("**/*")
            if Path(out_file).is_file()
        ]


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/chai1",
    run_name=None,
    chai1_kwargs=None,
    msa_a3m_dir: str = "",
):
    """Local entrypoint for running Chai-1 predictions using Modal.

    Args:
        input_faa (str): Path to the input FASTA file.
        out_dir (str): Directory to save the output files. Defaults to "./out/chai1".
        run_name (str | None): Optional name for the run, used in the output directory structure.
        chai1_kwargs (str | None): Optional string representation of a dictionary for additional
                                   Chai-1 keyword arguments (e.g., '{"key": "value"}').

    Returns:
        None
    """
    from datetime import datetime

    input_faa_str = open(input_faa).read()

    # Optional CLI mode: --msa-a3m-dir <dir> where filenames are <SEQ>.a3m
    # (SEQ uppercase). Keys in msa_a3m_by_seq dict = uppercased chain sequence.
    msa_a3m_by_seq = None
    if msa_a3m_dir:
        msa_a3m_by_seq = {}
        for p in Path(msa_a3m_dir).glob("*.a3m"):
            msa_a3m_by_seq[p.stem.upper()] = p.read_text()

    outputs = chai1.remote(
        input_faa_str,
        input_faa_name=Path(input_faa).name,
        msa_a3m_by_seq=msa_a3m_by_seq,
        chai1_kwargs=dict(eval(chai1_kwargs)) if chai1_kwargs else {},
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        (Path(out_dir_full) / Path(out_file)).write_bytes(out_content)
