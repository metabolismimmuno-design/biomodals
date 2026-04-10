# /// script
# requires-python = ">=3.9"
# dependencies = ["modal>=1.0"]
# ///
from __future__ import annotations
import modal

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_MODEL_VOLUME = modal.Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boltz==0.4.1", "pyyaml")
)

app = modal.App("debug-boltz2", image=image)

YAML_STR = """sequences:
  - protein:
      id: A
      sequence: SARMLGDVMAVSTCVPVAADNVIVQNSMRISSRPGACYSRPLVSFRYEDQGPLVEGQLGENNELRLTRDAIEPCTVGHRRYFTFGGGYVYFEEYAYSHQLSRADITTVSTF
  - protein:
      id: B
      sequence: GQVQLVESGGGLVQPGGSLRLSCTASGGDASAYDIIWFRQAPGREREFVAQITADGSHTNYADSVKDRFTISRDNAKNVVYLQMNSLKPEDTAVYYCAAVPGTGRTTLIPLSSLKNAYWGQGTQVTVSSG
"""


@app.function(
    gpu="L40S",
    timeout=900,
    volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME},
)
def predict_with_msa_server():
    """Test with MSA server enabled."""
    from subprocess import run
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        open(fixed_yaml := Path(in_dir) / "fixed.yaml", "w").write(YAML_STR)
        r = run(
            f'boltz predict "{fixed_yaml}" --out_dir "{out_dir}" --cache "{CACHE_DIR}" --use_msa_server --seed 42',
            shell=True, capture_output=True, text=True
        )
        print("Return code:", r.returncode)
        print("STDOUT:", r.stdout[-2000:])
        print("STDERR:", r.stderr[-2000:])
        return r.returncode


@app.function(gpu="L40S", timeout=300, volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME})
def show_predict_help():
    from subprocess import run
    r = run("boltz predict --help", shell=True, capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)


@app.local_entrypoint()
def main():
    print("=== boltz predict --help ===")
    show_predict_help.remote()
    print("\n=== predict with MSA server ===")
    predict_with_msa_server.remote()
