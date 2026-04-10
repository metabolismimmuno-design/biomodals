# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ESM2 pseudo-log-likelihood (PLL) scoring for protein sequences.

For each sequence, computes PLL by masking each position one at a time
and summing log P(aa_i | context). Normalized by sequence length.

Higher PLL = better sequence quality (more natural/designable).

Usage:
    modal run modal_esm2_pll.py --input-faa sequences.fasta \\
        --out-dir ./out/esm2 --run-name my_run

Output:
    CSV file with columns: seq_id, pll, length
    Written to out_dir/run_name/esm2_pll.csv
"""

import os
from pathlib import Path

from typing import Optional

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))
MODEL_NAME = "esm2_t33_650M_UR50D"


def download_model():
    import esm
    _model, _alphabet = esm.pretrained.esm2_t33_650M_UR50D()


image = (
    Image.micromamba(python_version="3.10")
    .apt_install(["git", "wget", "gcc", "g++"])
    .pip_install(
        ["torch==2.0.1+cu117"],
        index_url="https://download.pytorch.org/whl/cu117",
    )
    .pip_install(["fair-esm"])
    .run_function(download_model, gpu=GPU)
)

app = App("esm2_pll", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def compute_pll(fasta_str: str) -> list[tuple[str, float, int]]:
    """Compute PLL for all sequences in a FASTA string.

    Returns list of (seq_id, pll, length).
    PLL is mean log-probability across all positions (higher = better).
    """
    import torch
    import esm
    import math

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Parse FASTA
    entries = []
    header, seq_lines = None, []
    for line in fasta_str.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq_lines)))
            header = line[1:].split()[0]  # use first word as ID
            seq_lines = []
        elif line:
            seq_lines.append(line)
    if header is not None:
        entries.append((header, "".join(seq_lines)))

    print(f"Computing PLL for {len(entries)} sequences...")
    results = []

    for seq_id, seq in entries:
        L = len(seq)
        log_probs_sum = 0.0

        # Process in chunks of 32 masked positions for efficiency
        chunk_size = 32
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            # Build masked variants for this chunk
            masked_seqs = []
            positions = []
            for pos in range(chunk_start, chunk_end):
                masked = list(seq)
                masked[pos] = "<mask>"
                masked_seqs.append((f"{seq_id}_pos{pos}", "".join(masked)))
                positions.append(pos)

            _, _, tokens = batch_converter(masked_seqs)
            if torch.cuda.is_available():
                tokens = tokens.cuda()

            with torch.no_grad():
                results_out = model(tokens, repr_layers=[])

            logits = results_out["logits"]  # (batch, L+2, vocab)

            for i, pos in enumerate(positions):
                # Token index: +1 for BOS token
                token_idx = pos + 1
                log_prob = torch.nn.functional.log_softmax(
                    logits[i, token_idx, :], dim=-1
                )
                # Get the true amino acid token
                true_token = alphabet.get_idx(seq[pos])
                if true_token == alphabet.unk_idx:
                    continue
                log_probs_sum += log_prob[true_token].item()

        pll = log_probs_sum / L if L > 0 else float("-inf")
        results.append((seq_id, pll, L))
        print(f"  {seq_id}: PLL={pll:.4f} (L={L})")

    return results


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/esm2_pll",
    run_name: Optional[str] = None,
):
    from datetime import datetime

    fasta_str = open(input_faa).read()

    print(f"Sending {input_faa} to Modal for ESM2 PLL scoring...")
    results = compute_pll.remote(fasta_str)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_path = Path(out_dir) / (run_name or today)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "esm2_pll.csv"
    with open(csv_path, "w") as f:
        f.write("seq_id,pll,length\n")
        for seq_id, pll, length in results:
            f.write(f"{seq_id},{pll:.6f},{length}\n")

    print(f"Saved: {csv_path} ({len(results)} sequences)")
