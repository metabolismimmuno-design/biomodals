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
) -> bytes:
    """Run Boltz-2 structure prediction and return zip of outputs.

    Args:
        yaml_str: YAML string with sequences (protein, RNA, DNA, ligand, etc.)
        run_name: Name prefix for output files.
        use_msa_server: Use ColabFold MSA server for better accuracy.
        use_potentials: Enable inference-time potentials (slower, higher quality).
        recycling_steps: Number of structure recycling steps (default 3).
        diffusion_samples: Number of diffusion samples (default 1).
        seed: Random seed.

    Returns:
        bytes: ZIP archive containing PDB/CIF structures + confidence JSON files.
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
