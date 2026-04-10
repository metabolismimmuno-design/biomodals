# /// script
# requires-python = ">=3.9"
# dependencies = ["modal>=1.0"]
# ///
from __future__ import annotations
import modal
import os
from pathlib import Path

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_MODEL_VOLUME = modal.Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boltz==0.4.1", "pyyaml")
)

app = modal.App("debug-boltz", image=image)


@app.function(volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME})
def check_volume():
    files = list(Path(CACHE_DIR).glob("**/*"))
    print(f"Volume contents ({len(files)} files):")
    for f in files[:20]:
        print(f"  {f}")
    return files


@app.function(
    gpu="L40S",
    volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME},
)
def run_predict_debug(yaml_str: str):
    from subprocess import run, PIPE
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        open(fixed_yaml := Path(in_dir) / "fixed.yaml", "w").write(yaml_str)
        r = run(
            f'boltz predict "{fixed_yaml}" --out_dir "{out_dir}" --cache "{CACHE_DIR}" --seed 42',
            shell=True, capture_output=True, text=True
        )
        print("STDOUT:", r.stdout[-1000:])
        print("STDERR:", r.stderr[-1000:])
        print("Return code:", r.returncode)


YAML_STR = """sequences:
  - protein:
      id: A
      sequence: SARMLGDVMAVSTCVPVAADNVIVQNSMRISSRPGACYSRPLVSFRYEDQGPLVEGQLGENNELRLTRDAIEPCTVGHRRYFTFGGGYVYFEEYAYSHQLSRADITTVSTF
  - protein:
      id: B
      sequence: GQVQLVESGGGLVQPGGSLRLSCTASGGDASAYDIIWFRQAPGREREFVAQITADGSHTNYADSVKDRFTISRDNAKNVVYLQMNSLKPEDTAVYYCAAVPGTGRTTLIPLSSLKNAYWGQGTQVTVSSG
"""


@app.local_entrypoint()
def main():
    print("=== Volume contents ===")
    check_volume.remote()
    print("\n=== Predict debug ===")
    run_predict_debug.remote(YAML_STR)
