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

app = modal.App("fix-boltz-cache", image=image)


@app.function(volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME}, timeout=600)
def check_and_redownload():
    from pathlib import Path
    import os

    cache = Path(CACHE_DIR)
    ckpt = cache / "boltz1_conf.ckpt"
    ccd = cache / "ccd.pkl"

    # Check sizes
    print(f"boltz1_conf.ckpt: {ckpt.stat().st_size / 1e6:.1f} MB" if ckpt.exists() else "boltz1_conf.ckpt: MISSING")
    print(f"ccd.pkl: {ccd.stat().st_size / 1e6:.1f} MB" if ccd.exists() else "ccd.pkl: MISSING")

    # boltz1_conf.ckpt should be ~560 MB
    ckpt_ok = ckpt.exists() and ckpt.stat().st_size > 500_000_000
    ccd_ok = ccd.exists() and ccd.stat().st_size > 100_000

    print(f"Checkpoint valid: {ckpt_ok}")
    print(f"CCD valid: {ccd_ok}")

    if not ckpt_ok or not ccd_ok:
        print("Re-downloading...")
        if ckpt.exists():
            ckpt.unlink()
        if ccd.exists():
            ccd.unlink()
        from boltz.main import download
        download(cache)
        BOLTZ_MODEL_VOLUME.commit()
        print(f"Re-downloaded. ckpt size: {ckpt.stat().st_size / 1e6:.1f} MB")
    else:
        print("Cache OK, no re-download needed.")


@app.local_entrypoint()
def main():
    check_and_redownload.remote()
