# /// script
# requires-python = ">=3.9"
# dependencies = ["modal>=1.0"]
# ///
from __future__ import annotations
import modal

app = modal.App("check-boltz-dl")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boltz==0.4.1", "pyyaml")
)


@app.function(image=image)
def check_download():
    import inspect
    from boltz.main import download, MODEL_URL
    print("MODEL_URL:", MODEL_URL)
    print("download sig:", inspect.signature(download))
    print("download src snippet:")
    src = inspect.getsource(download)
    print(src[:1000])


@app.local_entrypoint()
def main():
    check_download.remote()
