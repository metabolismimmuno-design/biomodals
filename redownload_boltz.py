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

app = modal.App("redownload-boltz", image=image)


@app.function(volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME}, timeout=600)
def force_redownload():
    from pathlib import Path

    cache = Path(CACHE_DIR)
    for f in cache.iterdir():
        print(f"Deleting {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        f.unlink()

    print("Re-downloading Boltz-1 weights...")
    from boltz.main import download
    download(cache)
    BOLTZ_MODEL_VOLUME.commit()

    for f in cache.iterdir():
        print(f"Downloaded {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


@app.function(gpu="L40S", timeout=600, volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME})
def test_load_ckpt():
    from pathlib import Path
    import torch

    ckpt = Path(CACHE_DIR) / "boltz1_conf.ckpt"
    print(f"Loading {ckpt} ({ckpt.stat().st_size/1e6:.1f} MB)...")
    try:
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        print("Load OK, keys:", list(ck.keys())[:5])
    except Exception as e:
        print(f"Load FAILED: {e}")


@app.local_entrypoint()
def main():
    print("=== Force re-download ===")
    force_redownload.remote()
    print("\n=== Test checkpoint loading ===")
    test_load_ckpt.remote()
