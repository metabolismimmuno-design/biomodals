# /// script
# requires-python = ">=3.12"
# dependencies = ["modal>=1.0", "pyyaml>=6.0"]
# ///
"""
RFantibody: antibody-finetuned RFdiffusion for VHH/nanobody backbone generation.

Key advantage over modal_rfdiffusion.py (Path D Step D1):
  - RFdiffusion_Ab.pt checkpoint fine-tuned on SAbDab antibody structures
  - CDR-H3 geometry modeled with antibody-specific priors → lower Ibex rejection rate
  - Native nanobody support via HLT format; no manual --contigs needed

HLT preparation:
  HLT files are pre-generated locally by scaffold2hlt.py (~/biomodals/scaffold2hlt.py),
  which uses CDR boundaries from the scaffold YAML design block (validated against ANARCI).
  This replaces chothia2HLT.py, which hardcodes Chothia H3=95-102 and would miss the
  full CDR3 loop in sequentially-numbered VHH scaffolds.

Usage:
    # Step 1: generate HLT files (once per scaffold)
    python ~/biomodals/scaffold2hlt.py \\
        --pdb inputs/scaffolds/7dv4_B_clean.pdb \\
        --yaml inputs/scaffolds/7dv4_B.yaml \\
        --output inputs/rfantibody/7dv4_B.hlt.pdb

    # Step 2: run RFantibody
    modal run modal_rfantibody.py \\
      --target inputs/9JMB_chainA.pdb \\
      --hlt inputs/rfantibody/7dv4_B.hlt.pdb \\
      --target-chain A \\
      --hotspots "298,299,301,303,315,318,319,321" \\
      --design-loops "H1:10,H2:16,H3:14-27" \\
      --num-designs 500 \\
      --out-dir results/gen/rfantibody/7dv4_B

    # for >50 designs add --detach (StreamTerminatedError otherwise)
    modal run --detach modal_rfantibody.py ...

Args:
  --target          antigen PDB (local path)
  --hlt             pre-generated HLT scaffold file (from scaffold2hlt.py)
  --target-chain    antigen chain ID for hotspot prefix, e.g. "A"
  --hotspots        comma-separated residue numbers on target chain, e.g. "298,299,301"
                    → formatted as "A298,A299,A301" before passing to rfdiffusion CLI
  --design-loops    CDR loop length ranges, e.g. "H1:10,H2:16,H3:14-27"
                    lengths are total residues in each CDR region (matching YAML design block)
                    loops NOT listed here are fixed from the scaffold
  --num-designs     number of backbone designs to generate
  --diffuser-t      diffusion timesteps (default 50; reduce for speed, increase for diversity)
  --out-dir         local directory to save output PDB files
  --run-name        optional name prefix for output files
"""

import os
import subprocess
from pathlib import Path

import modal
from modal import Image, App

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 180))

app = App()

vol = modal.Volume.from_name("rfantibody-out", create_if_missing=True)
VOL_PATH = "/vol/rfantibody-out"

image = (
    Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "build-essential")
    # Antibody-finetuned RFdiffusion checkpoint (Baker lab; ~260 MB)
    .run_commands(
        "mkdir -p /weights && "
        "wget -q https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt -P /weights/"
    )
    .run_commands("git clone --depth 1 https://github.com/RosettaCommons/RFantibody.git /rfantibody")
    .pip_install(
        ["torch==2.3.1+cu118", "torchvision==0.18.1+cu118", "torchaudio==2.3.1+cu118"],
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl"
    )
    .pip_install([
        "hydra-core",
        "icecream",
        "opt-einsum",
        "e3nn",
        "pyrsistent",
        "click>=8.0.0",
        "pandas",
        "cuda-python==11.8",
        "pyyaml>=6.0",
    ])
    .run_commands("cd /rfantibody && pip install --no-deps -e .")
)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT * 60, volumes={VOL_PATH: vol})
def run_rfantibody(
    target_content: bytes,      # antigen PDB bytes
    target_chain: str,          # chain ID in target, e.g. "A"
    hlt_content: bytes,         # pre-generated HLT scaffold bytes (from scaffold2hlt.py)
    hotspots: str,              # comma-sep residue numbers on target chain
    design_loops: str,          # e.g. "H1:10,H2:16,H3:14-27"
    num_designs: int,
    diffuser_t: int,
    run_name: str,
) -> int:
    """Run RFantibody backbone generation inside Modal GPU container."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # --- 1. Write target PDB ------------------------------------------
        target_pdb = tmpdir / "target.pdb"
        target_pdb.write_bytes(target_content)

        # --- 2. Write pre-generated HLT scaffold --------------------------
        hlt_pdb = tmpdir / "framework.hlt.pdb"
        hlt_pdb.write_bytes(hlt_content)

        # --- 3. Format hotspots: prepend target chain ---------------------
        hotspot_residues = [r.strip() for r in hotspots.split(",") if r.strip()]
        hotspot_str = ",".join(f"{target_chain}{r}" for r in hotspot_residues)

        # --- 4. Run rfdiffusion CLI ---------------------------------------
        out_dir = tmpdir / "designs"
        out_dir.mkdir()
        out_prefix = out_dir / run_name

        cmd = [
            "rfdiffusion",
            "--target", str(target_pdb),
            "--framework", str(hlt_pdb),
            "--output", str(out_prefix),
            "--num-designs", str(num_designs),
            "--design-loops", design_loops,
            "--weights", "/weights/RFdiffusion_Ab.pt",
            f"--diffuser-t={diffuser_t}",
            "--no-trajectory",
        ]
        if hotspot_str:
            cmd += ["--hotspots", hotspot_str]

        print("[rfdiffusion] command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/rfantibody")
        print("[rfdiffusion] stdout:", result.stdout[-2000:])
        if result.returncode != 0:
            print("[rfdiffusion] stderr:", result.stderr[-2000:])
            raise RuntimeError(f"rfdiffusion failed (exit {result.returncode})")

        # --- 5. Collect output PDBs and write to Modal Volume -------------
        import shutil
        pdbs = list(out_dir.glob("*.pdb"))
        print(f"[output] {len(pdbs)} PDB files generated")
        vol_dir = Path(VOL_PATH) / run_name
        vol_dir.mkdir(parents=True, exist_ok=True)
        for p in pdbs:
            shutil.copy(p, vol_dir / p.name)
        vol.commit()
        print(f"[volume] {len(pdbs)} PDB files written to volume: {vol_dir}")
        return len(pdbs)


@app.local_entrypoint()
def main(
    target: str,
    hlt: str,
    target_chain: str = "A",
    hotspots: str = "",
    design_loops: str = "H1:10,H2:16,H3:14-27",
    num_designs: int = 100,
    diffuser_t: int = 50,
    out_dir: str = "./out/rfantibody",
    run_name: str = "",
):
    """Local entrypoint: read files, dispatch to Modal GPU, save outputs.

    Example:
        modal run --detach modal_rfantibody.py \\
          --target inputs/9JMB_chainA.pdb \\
          --hlt inputs/rfantibody/7dv4_B.hlt.pdb \\
          --target-chain A \\
          --hotspots "298,299,301,303,315,318,319,321" \\
          --design-loops "H1:10,H2:16,H3:14-27" \\
          --num-designs 500 \\
          --out-dir results/gen/rfantibody/7dv4_B
    """
    from datetime import datetime

    # --- Read HLT scaffold file -------------------------------------------
    hlt_path = Path(hlt).resolve()
    if not hlt_path.exists():
        raise FileNotFoundError(
            f"HLT file not found: {hlt_path}\n"
            f"Generate it first with: python ~/biomodals/scaffold2hlt.py "
            f"--pdb <scaffold_clean.pdb> --yaml <scaffold.yaml> --output {hlt_path}"
        )
    hlt_content = hlt_path.read_bytes()
    print(f"[scaffold] HLT: {hlt_path.name}")

    # --- Read target PDB --------------------------------------------------
    target_path = Path(target).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    if target_path.suffix.lower() not in (".pdb",):
        raise ValueError(f"Target must be a PDB file, got: {target_path.suffix}")
    target_content = target_path.read_bytes()
    print(f"[target] {target_path.name}  chain: {target_chain}")

    # --- Resolve run name ------------------------------------------------
    scaffold_tag = hlt_path.stem.replace(".hlt", "")
    ts = datetime.now().strftime("%m%d%H%M")
    run_name = run_name or f"{scaffold_tag}_{ts}"
    print(f"[run] name={run_name}  designs={num_designs}  loops={design_loops}")

    # --- Spawn GPU function and exit immediately (requires --detach) ------
    # Do NOT use .remote() or poll — ephemeral app heartbeat dies after ~10
    # min on a silent subprocess. GPU writes results to Modal Volume instead.
    run_rfantibody.spawn(
        target_content=target_content,
        target_chain=target_chain,
        hlt_content=hlt_content,
        hotspots=hotspots,
        design_loops=design_loops,
        num_designs=num_designs,
        diffuser_t=diffuser_t,
        run_name=run_name,
    )
    print(f"[spawned] GPU job submitted. run_name={run_name}")
    print(f"[wait]    ~10-15 min, then download:")
    print(f"[wait]    modal run ~/biomodals/modal_rfantibody.py::download --run-name {run_name} --out-dir {out_dir}")


@app.local_entrypoint()
def download(
    run_name: str,
    out_dir: str = "./out/rfantibody",
):
    """Download completed run from Modal Volume to local disk.

    Example:
        modal run ~/biomodals/modal_rfantibody.py::download \\
          --run-name 7a6o_B_05010920 \\
          --out-dir results/vhh_max_success/gen/rfantibody/7a6o_B/mvp_VH_hotspots
    """
    entries = list(vol.listdir(run_name))
    if not entries:
        print(f"[error] run_name '{run_name}' not found in volume. Check: modal volume ls rfantibody-out")
        return

    out_path = Path(out_dir) / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    n = 0
    for e in entries:
        fname = Path(e.path).name
        content = b"".join(vol.read_file(e.path))
        (out_path / fname).write_bytes(content)
        print(f"[saved] {fname}  ({len(content):,} bytes)")
        n += 1

    print(f"[done] {n} files → {out_path}")
