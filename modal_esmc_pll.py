# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ESM C pseudo-log-likelihood (PLL) scoring for protein sequences.

True masked PLL (same algorithm as modal_esm2_pll.py):
for each position, mask it, forward, take log P(true_aa | context),
sum over positions, normalize by length. Higher PLL = more natural.

Key difference from ESM2 version:
- Uses evolutionaryscale `esm` SDK (not `fair-esm`)
- ESMC model (300m / 600m / 6b); default 600m for closest parameter
  match to ESM2-650M used by modal_esm2_pll.py
- Batches L masked variants in a SINGLE forward pass per sequence
  (leveraging ESMC.forward(sequence_tokens=...) raw-tensor path),
  which is significantly faster than ESM2's chunked masking.

Usage:
    modal run modal_esmc_pll.py --input-faa sequences.fasta \\
        --out-dir ./out/esmc --run-name my_run --model-name esmc_600m

Output:
    CSV file with columns: seq_id,pll,length
    Written to out_dir/run_name/esmc_pll.csv
"""

import os
from pathlib import Path
from typing import Optional

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))
DEFAULT_MODEL = os.environ.get("ESMC_MODEL", "esmc_600m")


def download_models():
    """Pre-cache ESMC weights into the image layer."""
    from esm.models.esmc import ESMC
    for name in ("esmc_300m", "esmc_600m"):
        try:
            ESMC.from_pretrained(name)
            print(f"Cached {name}")
        except Exception as e:
            print(f"Failed to cache {name}: {e}")


image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install(
        [
            "torch==2.3.1",
            "esm",
            "huggingface_hub",
            "httpx",
        ]
    )
    .run_function(download_models)
)

app = App("esmc_pll", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def compute_pll(fasta_str: str, model_name: str = DEFAULT_MODEL) -> list[tuple[str, float, int]]:
    """Compute true masked PLL for all sequences in a FASTA string.

    Returns list of (seq_id, pll, length). PLL is length-normalized
    mean log-probability (higher = better).
    """
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = ESMC.from_pretrained(model_name).to(device).eval()

    # ---- Discovery: find mask token id & BOS offset ----
    # Probe: "A_A" → [BOS, A, MASK, A, EOS] (expected 5 tokens)
    probe = ESMProtein(sequence="A_A")
    probe_ids = model.encode(probe).sequence
    print(f"Probe 'A_A' token ids: {probe_ids.tolist()}")

    # Sanity: expect 2 extra tokens (BOS + EOS) beyond sequence length
    test_ids = model.encode(ESMProtein(sequence="ACDEFGHI")).sequence
    n_extra = test_ids.shape[0] - 8
    assert n_extra == 2, (
        f"Unexpected BOS/EOS count: got {test_ids.shape[0]} tokens for 8 AAs "
        f"(n_extra={n_extra}). Token IDs: {test_ids.tolist()}"
    )
    offset = 1  # BOS at index 0, so seq position p → token index p+1
    mask_id = probe_ids[2].item()
    print(f"mask_token_id = {mask_id}, BOS offset = {offset}")

    # ---- Parse FASTA ----
    entries = []
    header, seq_lines = None, []
    for line in fasta_str.strip().split("\n"):
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

    print(f"Computing masked PLL for {len(entries)} sequences...")
    results = []

    # Sub-batch for memory control (L masked variants × L_tok × hidden_dim)
    SUB_BATCH = 64

    for idx, (seq_id, seq) in enumerate(entries):
        L = len(seq)
        if L == 0:
            results.append((seq_id, float("-inf"), 0))
            continue

        # Encode original once to get ground-truth token ids
        orig_ids = model.encode(ESMProtein(sequence=seq)).sequence.to(device)
        L_tok = orig_ids.shape[0]
        assert L_tok == L + 2, f"Token length mismatch for {seq_id}: {L_tok} vs {L}+2"

        # Build L masked variants in a single (L, L_tok) tensor
        masked_batch = orig_ids.unsqueeze(0).repeat(L, 1).clone()
        for pos in range(L):
            masked_batch[pos, pos + offset] = mask_id

        log_probs_sum = 0.0
        for start in range(0, L, SUB_BATCH):
            end = min(start + SUB_BATCH, L)
            sub = masked_batch[start:end]

            with torch.no_grad():
                out = model.forward(sequence_tokens=sub)

            logits = out.sequence_logits  # (sub_B, L_tok, vocab)
            for i, pos in enumerate(range(start, end)):
                log_p = torch.nn.functional.log_softmax(
                    logits[i, pos + offset, :].float(), dim=-1
                )
                true_tok = orig_ids[pos + offset].item()
                log_probs_sum += log_p[true_tok].item()

            del out, logits
        del masked_batch

        pll = log_probs_sum / L
        results.append((seq_id, pll, L))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(entries):
            print(f"  [{idx + 1}/{len(entries)}] {seq_id}: PLL={pll:.4f} (L={L})")

        if torch.cuda.is_available() and (idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    return results


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/esmc_pll",
    run_name: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
):
    from datetime import datetime

    fasta_str = open(input_faa).read()

    print(f"Sending {input_faa} to Modal for ESM C PLL scoring ({model_name})...")
    results = compute_pll.remote(fasta_str, model_name)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_path = Path(out_dir) / (run_name or today)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "esmc_pll.csv"
    with open(csv_path, "w") as f:
        f.write("seq_id,pll,length\n")
        for seq_id, pll, length in results:
            f.write(f"{seq_id},{pll:.6f},{length}\n")

    print(f"Saved: {csv_path} ({len(results)} sequences, model={model_name})")
