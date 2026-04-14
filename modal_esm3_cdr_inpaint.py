# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ESM3 sequence-only CDR inpainting for VHH de novo design.

Path F in `vhh_max_success_design` pipeline (Round 2 R2-6). Uses the
`esm3-sm-open-v1` open-weights model (1.4B params, Cambrian license —
commercial use permitted with attribution).

ONE-TIME SETUP (HuggingFace gated repo):
    The ESM3-sm-open-v1 checkpoint is distributed through a gated HF repo
    even though the Cambrian license itself is permissive. To unblock:

      1. Sign in at https://huggingface.co and visit
         https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
         Click "Agree and access repository" (one-click Cambrian acceptance).
      2. Create a read-only HF token at
         https://huggingface.co/settings/tokens
      3. Register the token as a Modal secret named `huggingface` with
         key `HF_TOKEN`:
             modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
         (If a `huggingface` secret already exists and uses a different
         env-var name, pass it through — the esm SDK honors the standard
         `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` variables.)
      4. Re-run this script. The image rebuild will succeed and cache
         the 1.4B checkpoint into the image layer.

Approach (sequence-only MVP):
    1. LOCAL: parse input FASTA of VHH scaffolds, detect CDR1/2/3
       boundaries via protein-design-utils vhh.cdr_boundaries (motif
       method), build masked sequence with '_' replacing CDR residues.
    2. MODAL: load ESM3-sm, for each masked scaffold call
       `model.generate(protein, GenerationConfig(track="sequence",
       num_steps=n_mask, temperature=T))` N times to produce diverse
       samples (torch seed varies per sample).
    3. LOCAL: extract generated CDR subsequences using the original
       boundaries (length is preserved — inpainting is position-locked),
       write generated FASTA + metadata CSV.

This script does NOT compute sequence log-probabilities. Downstream
scoring of Path F outputs is handled by existing layers:
    - L3 (4-model orthogonal structure validation: chai-1 / boltz-2 /
      ibex / protenix)
    - L4.2 (ESM C PLL, see modal_esmc_pll.py — R2-3 output)
    - L4.7 (pymoo Pareto multi-objective ranking — R2-4 output)

Usage:
    # Real run (motif-based CDR detection via cdr_boundaries)
    modal run modal_esm3_cdr_inpaint.py \\
        --input-faa scaffolds.faa \\
        --num-samples 10 --temperature 0.7 \\
        --out-dir ./out/esm3_inpaint --run-name my_campaign

    # Real run with pre-computed ANARCI boundaries (for natural VHH
    # scaffolds whose non-canonical FR3 breaks motif parsing — e.g.
    # gB-VHH's 1mel/9q9n). JSON must match scaffold_annotations.json
    # schema: { "<fasta_header>": { "annotation": { "cdr1": [..],
    # "cdr2": [..], "cdr3": [..] } } }. CDR positions are 1-indexed
    # sequential; list min/max become the (start, end) tuple.
    modal run modal_esm3_cdr_inpaint.py \\
        --input-faa scaffolds.faa \\
        --boundaries-json configs/scaffold_annotations.json \\
        --num-samples 10 --temperature 0.7

    # Smoke test (2 built-in canonical VHH scaffolds x 3 samples each,
    # ~1 min wall on A10G including model load). No input FASTA needed.
    modal run modal_esm3_cdr_inpaint.py --smoke-test

Output (out_dir/run_name/):
    esm3_generated.faa      FASTA with headers `{scaffold_id}_esm3_s{NNN}`
    esm3_inpaint_meta.csv   scaffold_id, sample_idx, temperature,
                            cdr1_seq, cdr2_seq, cdr3_seq, cdr3_length
"""

import os
import sys
from pathlib import Path
from typing import Optional

import modal
from modal import App, Image, Secret

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))
DEFAULT_MODEL = os.environ.get("ESM3_MODEL", "esm3-sm-open-v1")

# HF token secret (name configurable via env var for flexibility).
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", "huggingface")
hf_secret = Secret.from_name(HF_SECRET_NAME)


def download_esm3_weights():
    """Pre-cache ESM3 weights into the image layer (~3GB).

    Called once at image build time so cold-start avoids HuggingFace
    download round-trip.
    """
    from esm.models.esm3 import ESM3
    try:
        ESM3.from_pretrained(DEFAULT_MODEL)
        print(f"Cached {DEFAULT_MODEL}")
    except Exception as e:
        print(f"Failed to cache {DEFAULT_MODEL}: {e}")


image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install(
        [
            "torch==2.3.1",
            "esm",
            "huggingface_hub",
            "httpx",  # esm.sdk.forge transitive dep not declared in esm pkg
        ]
    )
    # HF_TOKEN must be in scope during image build for the gated
    # EvolutionaryScale/esm3-sm-open-v1 download.
    .run_function(download_esm3_weights, secrets=[hf_secret])
)

app = App("esm3_cdr_inpaint", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU, secrets=[hf_secret])
def run_inpaint(
    masked_fasta: str,
    num_samples: int = 10,
    temperature: float = 0.7,
    model_name: str = DEFAULT_MODEL,
) -> list:
    """Run ESM3 sequence inpainting on pre-masked VHH scaffolds.

    Args:
        masked_fasta: FASTA string where each sequence already has its
            CDR residues replaced with '_' (ESM3 mask token). This
            function does NO CDR detection — boundaries come from the
            local side via protein-design-utils.
        num_samples: Number of independent samples per scaffold.
        temperature: ESM3 decoding temperature (0.0 deterministic,
            higher = more diverse). Default 0.7 is a common middle ground.
        model_name: ESM3 checkpoint name.

    Returns:
        list of 3-tuples (scaffold_id, sample_idx, generated_sequence).
        Samples that fail generation (mask residues remain, or length
        changed) are silently dropped — the `main()` reports the
        pass rate.
    """
    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = ESM3.from_pretrained(model_name).to(device).eval()

    # ---- Parse input FASTA ----
    entries = []
    header, seq_lines = None, []
    for line in masked_fasta.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq_lines)))
            header = line[1:].split()[0]
            seq_lines = []
        elif line:
            seq_lines.append(line)
    if header is not None:
        entries.append((header, "".join(seq_lines)))

    print(
        f"Inpainting {len(entries)} scaffolds x {num_samples} samples "
        f"(T={temperature})..."
    )

    results = []
    for sc_idx, (scaffold_id, masked_seq) in enumerate(entries):
        n_mask = masked_seq.count("_")
        if n_mask == 0:
            print(f"  [{sc_idx + 1}] {scaffold_id}: no masks, skipping")
            continue

        sc_pass = 0
        for s in range(num_samples):
            # Per-sample seed → distinct trajectories even at low T.
            torch.manual_seed(1000 * sc_idx + s)

            protein = ESMProtein(sequence=masked_seq)
            config = GenerationConfig(
                track="sequence",
                num_steps=n_mask,  # one decode step per masked residue
                temperature=temperature,
            )
            try:
                out = model.generate(protein, config)
                gen_seq = out.sequence
            except Exception as e:
                print(f"  [{scaffold_id}/s{s}] ERROR: {e}")
                continue

            if gen_seq is None or "_" in gen_seq:
                print(f"  [{scaffold_id}/s{s}] incomplete generation, dropped")
                continue
            if len(gen_seq) != len(masked_seq):
                print(
                    f"  [{scaffold_id}/s{s}] length drift "
                    f"{len(masked_seq)}->{len(gen_seq)}, dropped"
                )
                continue

            results.append((scaffold_id, s, gen_seq))
            sc_pass += 1

        print(
            f"  [{sc_idx + 1}/{len(entries)}] {scaffold_id}: "
            f"{sc_pass}/{num_samples} PASS"
        )

    return results


# ── Smoke-test built-in scaffolds ───────────────────────────────────────────
# Both verified to match `get_cdr_boundaries` motifs:
#   FR2 anchor W[FYV]RQ, FR3 anchor [A-Z]RF[TN]ISRDNAK,
#   FR3-end YYC/YFC/YHC, FR4 anchor WG[QR]GT.
SMOKE_SCAFFOLDS = {
    "smoke_vhh_WFRQ": (
        "QVQLQESGGGSVQAGGSLRLSCAASGYTIGPYCMGWFRQAPGKEREGVAAINMGGGITYYADSVKG"
        "RFTISRDNAKNTVYLQMNSLKPEDTAIYYCAASRYGNSWYIRTPQSHDYWGQGTQVTVSS"
    ),
    "smoke_vhh_WVRQ": (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
        "RFTISRDNAKNTLYLQMNSLRAEDTAVYYCAKDRGYSYYFDYWGQGTLVTVSS"
    ),
}


@app.local_entrypoint()
def main(
    input_faa: Optional[str] = None,
    boundaries_json: Optional[str] = None,
    num_samples: int = 10,
    temperature: float = 0.7,
    out_dir: str = "./out/esm3_inpaint",
    run_name: Optional[str] = None,
    smoke_test: bool = False,
):
    """Local driver: CDR detection + Modal dispatch + output persistence.

    CDR boundaries can come from two sources:
      1. Motif parser via ``protein-design-utils.vhh.cdr_boundaries``
         (default). Fails on natural VHHs with non-canonical FR3.
      2. Pre-computed ANARCI/IMGT annotations via ``--boundaries-json``.
         JSON maps FASTA header → {"annotation": {"cdr1/2/3": [positions]}}.
         Each CDR's list min/max become the 1-indexed (start, end) tuple.
    """
    import json
    from datetime import datetime

    # Locate protein-design-utils for CDR boundary detection.
    pdu_path = Path.home() / "protein-design-utils"
    if pdu_path.exists() and str(pdu_path) not in sys.path:
        sys.path.insert(0, str(pdu_path))
    try:
        from vhh.cdr_boundaries import get_cdr_boundaries, build_masked_seq
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import vhh.cdr_boundaries from {pdu_path}. "
            f"Ensure ~/protein-design-utils exists and has vhh/cdr_boundaries.py. "
            f"Original error: {e}"
        )

    # Load pre-computed boundaries if provided.
    precomputed_bounds: Optional[dict] = None
    if boundaries_json is not None:
        with open(boundaries_json) as f:
            raw = json.load(f)
        precomputed_bounds = {}
        for sc_id, entry in raw.items():
            ann = entry.get("annotation", entry)  # allow flat or nested
            try:
                precomputed_bounds[sc_id] = {
                    "cdr1": (min(ann["cdr1"]), max(ann["cdr1"])),
                    "cdr2": (min(ann["cdr2"]), max(ann["cdr2"])),
                    "cdr3": (min(ann["cdr3"]), max(ann["cdr3"])),
                }
            except (KeyError, ValueError) as e:
                print(f"  [WARN] boundaries_json entry '{sc_id}' malformed: {e}")
        print(
            f"Loaded pre-computed boundaries for {len(precomputed_bounds)} "
            f"scaffolds from {boundaries_json}"
        )

    # ---- Load input sequences ----
    if smoke_test:
        print("SMOKE TEST: 2 built-in VHH scaffolds x 3 samples each")
        entries = list(SMOKE_SCAFFOLDS.items())
        num_samples = 3
    else:
        if input_faa is None:
            raise ValueError(
                "Either --input-faa or --smoke-test is required."
            )
        entries = []
        header, parts = None, []
        with open(input_faa) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if header:
                        entries.append((header, "".join(parts)))
                    header = line[1:].split()[0]
                    parts = []
                elif line:
                    parts.append(line)
        if header:
            entries.append((header, "".join(parts)))

    # ---- CDR detection + masking (local) ----
    masked_entries = []
    scaffold_bounds = {}  # scaffold_id → ((c1_s,c1_e),(c2_s,c2_e),(c3_s,c3_e))
    for sc_id, seq in entries:
        bounds: Optional[dict] = None
        if precomputed_bounds is not None and sc_id in precomputed_bounds:
            bounds = precomputed_bounds[sc_id]
            src = "pre-computed"
        else:
            try:
                bounds = get_cdr_boundaries(seq)
                src = "motif"
            except ValueError as e:
                print(f"  [SKIP] {sc_id}: CDR detection failed ({e})")
                continue
        masked = build_masked_seq(seq, bounds, mask_char="_")
        masked_entries.append((sc_id, masked))
        scaffold_bounds[sc_id] = (bounds["cdr1"], bounds["cdr2"], bounds["cdr3"])
        print(
            f"  [OK] {sc_id} ({src}): L={len(seq)}, "
            f"CDR1={bounds['cdr1']}, CDR2={bounds['cdr2']}, "
            f"CDR3={bounds['cdr3']}, masked={masked.count('_')}"
        )

    if not masked_entries:
        raise RuntimeError("No scaffolds survived CDR detection.")

    masked_fasta = "\n".join(
        f">{sid}\n{seq}" for sid, seq in masked_entries
    )
    print(
        f"\nDispatching {len(masked_entries)} masked scaffolds to Modal "
        f"(num_samples={num_samples}, T={temperature})..."
    )
    results = run_inpaint.remote(
        masked_fasta,
        num_samples=num_samples,
        temperature=temperature,
    )

    # ---- Persist output ----
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    tag = run_name or today
    if smoke_test:
        tag = f"smoke_{tag}"
    out_path = Path(out_dir) / tag
    out_path.mkdir(parents=True, exist_ok=True)

    fasta_path = out_path / "esm3_generated.faa"
    with open(fasta_path, "w") as f:
        for sc_id, s_idx, gen_seq in results:
            f.write(f">{sc_id}_esm3_s{s_idx:03d}\n{gen_seq}\n")

    csv_path = out_path / "esm3_inpaint_meta.csv"
    with open(csv_path, "w") as f:
        f.write(
            "scaffold_id,sample_idx,temperature,"
            "cdr1_seq,cdr2_seq,cdr3_seq,cdr3_length\n"
        )
        for sc_id, s_idx, gen_seq in results:
            c1, c2, c3 = scaffold_bounds[sc_id]
            cdr1 = gen_seq[c1[0] - 1:c1[1]]
            cdr2 = gen_seq[c2[0] - 1:c2[1]]
            cdr3 = gen_seq[c3[0] - 1:c3[1]]
            f.write(
                f"{sc_id},{s_idx},{temperature},"
                f"{cdr1},{cdr2},{cdr3},{len(cdr3)}\n"
            )

    # ---- Summary ----
    n_gen = len(results)
    n_expected = len(masked_entries) * num_samples
    pass_rate = 100.0 * n_gen / max(n_expected, 1)
    print(f"\nSaved FASTA: {fasta_path}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Generated:   {n_gen}/{n_expected} samples ({pass_rate:.1f}%)")

    # ---- Smoke-test assertions ----
    if smoke_test:
        assert n_gen > 0, "SMOKE FAIL: zero samples generated"
        # Diversity check: generated CDR3 sequences should not all collapse
        cdr3s = set()
        for sc_id, s_idx, gen_seq in results:
            _, _, c3 = scaffold_bounds[sc_id]
            cdr3s.add(gen_seq[c3[0] - 1:c3[1]])
        print(f"SMOKE: unique CDR3 sequences = {len(cdr3s)}/{n_gen}")
        min_unique = max(2, n_gen // 2)
        assert len(cdr3s) >= min_unique, (
            f"SMOKE FAIL: diversity too low "
            f"({len(cdr3s)} unique / {n_gen} generated, "
            f"min expected {min_unique})"
        )
        # Per-scaffold: every scaffold should have produced at least 1 sample
        scaffolds_with_output = {sc_id for sc_id, _, _ in results}
        assert scaffolds_with_output == set(scaffold_bounds.keys()), (
            f"SMOKE FAIL: some scaffolds produced zero samples "
            f"({set(scaffold_bounds.keys()) - scaffolds_with_output})"
        )
        print("SMOKE PASS ✓")
