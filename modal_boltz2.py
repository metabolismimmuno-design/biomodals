# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Boltz-2 protein structure and affinity prediction on Modal.

Boltz-2 (https://github.com/jwohlwend/boltz) is a biomolecular
structure prediction model supporting structure + binding affinity.

## Input formats

### Structure prediction (YAML):
```yaml
sequences:
  - protein:
      id: A
      sequence: TDKLIFGKGTRVTVEP
```

### Affinity prediction (YAML, requires protein + ligand):
```yaml
sequences:
  - protein:
      id: A
      sequence: MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEREQESRKEAAG
  - ligand:
      id: B
      smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
```

Ligand can be specified as `smiles` or `ccd` (CCD code, e.g. ATP).

## Usage:
```
modal run modal_boltz2.py --input-yaml prot.yaml
modal run modal_boltz2.py --input-yaml affinity.yaml --affinity
```

## Deploy for MCP:
```
modal deploy modal_boltz2.py
```
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))  # minutes

BOLTZ_VOLUME_NAME = "boltz2-models"
BOLTZ_MODEL_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

BOLTZ_VERSION = "2.0.3"


def _download_boltz2_models():
    """Download Boltz-2 model weights to persistent volume."""
    import subprocess

    cache = Path(CACHE_DIR)
    # Boltz-2 downloads weights to ~/.boltz on first run; redirect to volume
    os.environ["BOLTZ_CACHE"] = str(cache)

    # Check if already downloaded (boltz caches model_weights dir)
    model_dir = cache / "boltz2"
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Boltz-2 models already cached at {model_dir}")
        return

    print("Downloading Boltz-2 weights (~4 GB)...")
    # Run a minimal prediction to trigger download
    import yaml
    from tempfile import TemporaryDirectory

    test_yaml = {
        "sequences": [
            {"protein": {"id": "A", "sequence": "MAWTPLLLL"}}
        ]
    }
    with TemporaryDirectory() as td:
        input_path = Path(td) / "input.yaml"
        input_path.write_text(yaml.dump(test_yaml))
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        result = subprocess.run(
            ["boltz", "predict", str(input_path), "--out_dir", str(out_dir),
             "--accelerator", "gpu", "--devices", "1"],
            capture_output=True, text=True, timeout=300,
        )
        print("STDOUT:", result.stdout[-1000:] if result.stdout else "")
        print("STDERR:", result.stderr[-500:] if result.stderr else "")
    print("Boltz-2 model weights downloaded.")


image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential", "libffi-dev")
    .pip_install(
        "torch==2.5.1",
        f"boltz=={BOLTZ_VERSION}",
        "pyyaml",
        "pandas",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"BOLTZ_CACHE": CACHE_DIR})
    .run_function(
        _download_boltz2_models,
        gpu=GPU,
        timeout=20 * 60,
        volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME},
    )
)

app = App("boltz2", image=image)


# ─────────────────────────────────────────────────────────────────────────────
# Structure Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME},
)
def run_structure_prediction(
    yaml_str: str,
    run_name: str = "boltz2_structure",
    use_msa_server: bool = True,
    use_potentials: bool = False,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    seed: int = 42,
    template_cif_str: str | None = None,
    template_pdb_str: str | None = None,
    msa_files: dict | None = None,
) -> bytes:
    """Run Boltz-2 structure prediction and return zip of outputs.

    Args:
        yaml_str: YAML string with sequences (protein, RNA, DNA, ligand, etc.)
        run_name: Name prefix for output files.
        use_msa_server: Use ColabFold MSA server. Only applies to protein chains
            with `msa: auto` (or msa unset). Chains with explicit `msa: empty`
            or `msa: __MSA_<key>__` are respected per-chain.
        use_potentials: Enable inference-time potentials (slower, higher quality).
        recycling_steps: Number of structure recycling steps (default 3).
        diffusion_samples: Number of diffusion samples (default 1).
        seed: Random seed.
        template_cif_str: Optional CIF content as string. If provided, written to
            container-side file; yaml_str must contain placeholder `__TEMPLATE_CIF__`
            which will be substituted with the container path before invocation.
        msa_files: Optional dict {placeholder_key: a3m_content_str}. For each key,
            file is written to a temp path and yaml `__MSA_<key>__` placeholders are
            substituted with that path. Use this for per-chain MSA control: antigen
            chain → `msa: __MSA_A__`; VHH chain → `msa: empty` (single-sequence,
            required for de novo CDR designs to avoid template-pulling bias).

    Returns:
        bytes: ZIP archive containing PDB/CIF structures + confidence JSON files.
    """
    import io
    import subprocess
    import zipfile
    from tempfile import TemporaryDirectory

    os.environ["BOLTZ_CACHE"] = CACHE_DIR

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        if template_cif_str is not None:
            tpl_path = Path(in_dir) / "template.cif"
            tpl_path.write_text(template_cif_str)
            yaml_str = yaml_str.replace("__TEMPLATE_CIF__", str(tpl_path))
            print(f"Template CIF staged at {tpl_path} ({len(template_cif_str)} chars)")
        if template_pdb_str is not None:
            tpl_path = Path(in_dir) / "template.pdb"
            tpl_path.write_text(template_pdb_str)
            yaml_str = yaml_str.replace("__TEMPLATE_PDB__", str(tpl_path))
            print(f"Template PDB staged at {tpl_path} ({len(template_pdb_str)} chars)")
        if msa_files:
            for key, a3m_str in msa_files.items():
                placeholder = f"__MSA_{key}__"
                if placeholder not in yaml_str:
                    print(f"[MSA] WARNING: placeholder {placeholder} not found in yaml_str", flush=True)
                    continue
                msa_path = Path(in_dir) / f"msa_{key}.a3m"
                # Sanitize: ColabFold a3m sometimes ends with a trailing NULL byte
                # which boltz's a3m parser rejects (KeyError on '\x00').
                clean = a3m_str.replace("\x00", "")
                msa_path.write_text(clean)
                yaml_str = yaml_str.replace(placeholder, str(msa_path))
                print(f"[MSA] {placeholder} → {msa_path} ({len(clean)} chars, stripped {len(a3m_str)-len(clean)} NULL bytes)", flush=True)
            # Boltz #627: cached a3m + multi-protein-chain → silent cross-chain
            # paired-MSA failure. parse_a3m() lacks a taxonomy DB at inference, so
            # every cached sequence gets taxonomy_id=-1; construct_paired_msa()
            # then treats the chains as evolutionarily unrelated. MSA depth and
            # monomer pTM look normal — only a pos/neg iPTM Δ exposes the loss of
            # discrimination. Fix: key=0 forced-pairing CSV MSA, or the
            # Novel-Therapeutics/boltz-community fork (commit 51d54c4).
            # https://github.com/jwohlwend/boltz/issues/627
            try:
                import yaml as _msa_yaml
                _seqs = (_msa_yaml.safe_load(yaml_str) or {}).get("sequences", [])
                _n_prot = sum(1 for s in _seqs if isinstance(s, dict) and "protein" in s)
            except Exception:
                _n_prot = 0
            if _n_prot >= 2:
                print(f"[MSA] WARNING (Boltz #627): cached a3m supplied for a "
                      f"{_n_prot}-protein-chain complex — cross-chain paired-MSA may "
                      f"silently collapse (taxonomy_id=-1), making iPTM lose pos/neg "
                      f"discrimination. Verify with a pos/neg control, or switch to "
                      f"key=0 CSV MSA / the boltz-community fork. "
                      f"https://github.com/jwohlwend/boltz/issues/627", flush=True)
        input_path = Path(in_dir) / "input.yaml"
        input_path.write_text(yaml_str)

        if template_cif_str is not None or template_pdb_str is not None:
            print(f"=== Final YAML (post-substitution) ===\n{yaml_str}\n=== /YAML ===", flush=True)
            # Boltz CLI catches template parse errors and prints only the brief message.
            # Call the parsing pipeline directly to surface the full traceback.
            try:
                import yaml as _yaml
                import pickle
                from boltz.data.parse.schema import parse_boltz_schema
                from boltz.main import CCD_PATH  # path to ccd.pkl in boltz cache
                with open(input_path) as f:
                    schema_dict = _yaml.safe_load(f)
                # Load ccd from default cache path used by boltz CLI
                ccd_path = Path(CACHE_DIR) / "ccd.pkl"
                ccd = None
                for trial in (ccd_path, Path(CCD_PATH)):
                    if trial.exists():
                        with open(trial, "rb") as f:
                            ccd = pickle.load(f)
                        break
                if ccd is None:
                    print("[dryrun] could not locate ccd.pkl; skipping dryrun", flush=True)
                else:
                    print(f"[dryrun] running parse_boltz_schema(boltz_2=True)...", flush=True)
                    parse_boltz_schema("input", schema_dict, ccd, mol_dir=None, boltz_2=True)
                    print("[dryrun] parse OK", flush=True)
            except Exception as e:
                import traceback as _tb
                print(f"[dryrun] PARSE ERROR: {type(e).__name__}: {e}", flush=True)
                _tb.print_exc()

        cmd = [
            "boltz", "predict", str(input_path),
            "--out_dir", out_dir,
            "--seed", str(seed),
            "--recycling_steps", str(recycling_steps),
            "--diffusion_samples", str(diffusion_samples),
            "--accelerator", "gpu",
            "--devices", "1",
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        if use_potentials:
            cmd.append("--use_potentials")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT * 55)

        if result.returncode != 0:
            print("STDERR:", result.stderr[-2000:])
            raise RuntimeError(f"boltz predict failed (rc={result.returncode}): {result.stderr[-500:]}")

        if result.stderr:
            print("STDERR (rc=0):", result.stderr[-3000:], flush=True)
        print(result.stdout[-1000:] if result.stdout else "")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in Path(out_dir).rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(out_dir))
        buf.seek(0)
        return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Affinity Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME},
)
def run_affinity_prediction(
    yaml_str: str,
    run_name: str = "boltz2_affinity",
    use_msa_server: bool = True,
    use_potentials: bool = False,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    seed: int = 42,
) -> bytes:
    """Run Boltz-2 affinity prediction (protein-ligand) and return zip of outputs.

    YAML must contain a protein + ligand (smiles or ccd).

    Args:
        yaml_str: YAML string with protein + ligand sequences.
        run_name: Name prefix for output files.
        use_msa_server: Use ColabFold MSA server for better accuracy.
        use_potentials: Enable inference-time potentials.
        recycling_steps: Number of recycling steps (default 3).
        diffusion_samples: Number of diffusion samples (default 1).
        seed: Random seed.

    Returns:
        bytes: ZIP archive containing complex structures + affinity scores.
    """
    import io
    import subprocess
    import zipfile
    from tempfile import TemporaryDirectory

    os.environ["BOLTZ_CACHE"] = CACHE_DIR

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        input_path = Path(in_dir) / "input.yaml"
        input_path.write_text(yaml_str)

        cmd = [
            "boltz", "predict", str(input_path),
            "--out_dir", out_dir,
            "--property", "affinity",
            "--seed", str(seed),
            "--recycling_steps", str(recycling_steps),
            "--diffusion_samples", str(diffusion_samples),
            "--accelerator", "gpu",
            "--devices", "1",
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        if use_potentials:
            cmd.append("--use_potentials")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT * 55)

        if result.returncode != 0:
            print("STDERR:", result.stderr[-2000:])
            raise RuntimeError(f"boltz predict failed (rc={result.returncode}): {result.stderr[-500:]}")

        print(result.stdout[-1000:] if result.stdout else "")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in Path(out_dir).rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(out_dir))
        buf.seek(0)
        return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoints (modal run modal_boltz2.py ...)
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    input_yaml: str = "",
    affinity: bool = False,
    no_msa_server: bool = False,
    use_potentials: bool = False,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    seed: int = 42,
    out_dir: str = "./boltz2_output",
):
    """CLI wrapper for quick testing.

    Usage:
        modal run modal_boltz2.py --input-yaml prot.yaml
        modal run modal_boltz2.py --input-yaml affinity.yaml --affinity
    """
    import zipfile
    from pathlib import Path

    if not input_yaml:
        print("Usage: modal run modal_boltz2.py --input-yaml <file.yaml>")
        return

    yaml_str = Path(input_yaml).read_text()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"{'Affinity' if affinity else 'Structure'} prediction via Boltz-2...")
    fn = run_affinity_prediction if affinity else run_structure_prediction
    zip_bytes = fn.remote(
        yaml_str=yaml_str,
        use_msa_server=not no_msa_server,
        use_potentials=use_potentials,
        recycling_steps=recycling_steps,
        diffusion_samples=diffusion_samples,
        seed=seed,
    )

    import io
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(out_path)

    files = list(out_path.rglob("*"))
    print(f"\nOutputs ({len(files)} files) → {out_path}")
    for f in sorted(files)[:20]:
        if f.is_file():
            print(f"  {f.relative_to(out_path)}")
