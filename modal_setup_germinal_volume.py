"""One-time setup script to populate the germinal-models Modal Volume with AlphaFold2 parameters.

Run this before using modal_mber.py or modal_germinal.py for the first time:

    modal run modal_setup_germinal_volume.py

This downloads the AlphaFold-Multimer parameters (~3-4 GB) from Google Storage
into the 'germinal-models' Volume at /germinal-models/alphafold_params.
"""

from modal import App, Image, Volume

GERMINAL_VOLUME_NAME = "germinal-models"
GERMINAL_MODEL_VOLUME = Volume.from_name(GERMINAL_VOLUME_NAME, create_if_missing=True)
AF_PARAMS_DIR = f"/{GERMINAL_VOLUME_NAME}/alphafold_params"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("wget", "tar")
)

app = App("germinal-volume-setup", image=image)


@app.function(
    timeout=60 * 60,  # 1 hour timeout for download
    volumes={f"/{GERMINAL_VOLUME_NAME}": GERMINAL_MODEL_VOLUME},
)
def download_alphafold_params():
    """Download AlphaFold-Multimer parameters into the germinal-models volume."""
    import subprocess
    from pathlib import Path

    params_dir = Path(AF_PARAMS_DIR)
    params_dir.mkdir(parents=True, exist_ok=True)

    sentinel = params_dir / "params_model_1_multimer_v3.npz"
    if sentinel.exists():
        print("AF2 params already present, skipping download.")
        return

    tar_path = params_dir / "alphafold_params_2022-12-06.tar"
    print("Downloading AlphaFold-Multimer parameters (~3.5 GB)...")
    subprocess.run(
        [
            "wget",
            "-q",
            "--show-progress",
            "-O",
            str(tar_path),
            "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
        ],
        check=True,
    )
    print("Download complete. Extracting...")
    subprocess.run(
        ["tar", "-xf", str(tar_path), "-C", str(params_dir)],
        check=True,
    )
    tar_path.unlink()  # remove tar after extraction
    GERMINAL_MODEL_VOLUME.commit()
    print(f"Done. AF2 params are at {params_dir}")
    # List extracted files
    files = list(params_dir.iterdir())
    print(f"  {len(files)} files in {params_dir}")


@app.local_entrypoint()
def main():
    download_alphafold_params.remote()
    print("\nVolume setup complete. You can now run modal_mber.py.")
