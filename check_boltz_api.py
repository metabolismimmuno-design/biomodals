# /// script
# requires-python = ">=3.9"
# dependencies = ["modal>=1.0"]
# ///
from __future__ import annotations
import modal

app = modal.App("check-boltz-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install("boltz==0.4.1", "pyyaml")
)


@app.function(image=image)
def check_api():
    import subprocess
    r = subprocess.run(["boltz", "--help"], capture_output=True, text=True)
    print("=== boltz --help ===")
    print(r.stdout or r.stderr)

    import boltz.main as bm
    attrs = [x for x in dir(bm) if not x.startswith("_")]
    print("=== boltz.main attrs ===")
    print(attrs)


@app.local_entrypoint()
def main():
    check_api.remote()
