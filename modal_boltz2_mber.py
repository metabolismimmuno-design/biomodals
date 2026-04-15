# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Boltz-2 oracle MBER — VHH 定向进化（Boltz-2 作为评分 oracle）

替代 modal_mber.py 中的 AF2-Multimer oracle，消除 optimize（AF2）vs
validate（Boltz-2）的信号错位（domain_protein_design_gotchas #2）。

## 设计原则
- **Oracle**：Boltz-2 structure prediction ipTM（protein-protein 用 ipTM，
  affinity predicted_ddg 仅支持 protein-ligand，不适用于 VHH-抗原）
- **进化回路**：对 CDR 位置做全氨基酸单点扫描，按 Boltz-2 ipTM 排名保留 top-N
- **输入格式**：YAML（Boltz-2 原生格式），target 和 VHH 均传序列字符串

## 用法
```bash
modal run modal_boltz2_mber.py \\
  --target-seq SARMLGDVMAVSTICVPVAADNVIVQNSMRISSRPGAC... \\
  --input-fasta candidates.fasta \\
  --cdr-positions "26-33,50-57,97-115" \\
  --hotspot-residues "92,95,96,97,99" \\
  --n-rounds 3 \\
  --top-n 10 \\
  --output-dir ./out/boltz2_mber
```

## CDR 位置说明
--cdr-positions 格式：逗号分隔的"start-end"区间（1-indexed，闭区间）。
典型 VHH IMGT: "27-38,56-65,105-117"。
若不传，从序列中自动检测（基于 WG[QR]GT FR4 motif 估算）。

## 注意
- L5 MBER 完成后仍强制重走 L3 四模型验证（gotchas #2 要求不变）
- 每轮 Boltz-2 批量评分耗时约 1-3 小时（取决于候选数和 GPU 型号）
"""

from __future__ import annotations

import os
import json
import random
import re
from pathlib import Path
from typing import Optional

from modal import App, Image, Volume

# ─── Modal 环境配置（与 modal_boltz2.py 保持一致）────────────────────────────

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 120))  # minutes

BOLTZ_VOLUME_NAME = "boltz2-models"
BOLTZ_MODEL_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=False)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

BOLTZ_VERSION = "2.0.3"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential", "libffi-dev")
    .pip_install(
        "torch==2.5.1",
        f"boltz=={BOLTZ_VERSION}",
        "pyyaml",
        "pandas",
        "biopython",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"BOLTZ_CACHE": CACHE_DIR})
)

app = App("boltz2-mber", image=image)

# ─── 氨基酸常量 ───────────────────────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ─── CDR 位置检测（Modal 内嵌版，不依赖本地 cdr_boundaries.py）────────────────

def _parse_cdr_positions(cdr_str: str) -> list[int]:
    """解析 CDR 位置字符串，返回 1-indexed 位置列表。

    Args:
        cdr_str: 格式如 "26-33,50-57,97-115"（逗号分隔的区间）
    """
    positions = []
    for part in cdr_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            positions.extend(range(int(start), int(end) + 1))
        else:
            positions.append(int(part))
    return sorted(set(positions))


def _detect_cdr_positions(seq: str) -> list[int]:
    """从 VHH 序列自动检测 CDR 位置（基于 WG[QR]GT FR4 motif 估算）。

    使用简化的 IMGT 近似，适用于典型 VHH scaffold（约 120aa）。
    不如 cdr_boundaries.py 精确，建议优先用 --cdr-positions 传入已知边界。
    """
    # 找 FR4 motif (WG[QR]GT)
    m = re.search(r"WG[QR]GT", seq)
    if m:
        fr4_start = m.start() + 1  # 1-indexed
    else:
        # Fallback: 假设序列约 120aa，FR4 在 110-117 附近
        fr4_start = max(len(seq) - 10, 105)

    # 基于 FR4 位置反推 CDR3 结束（FR4 前一位）
    cdr3_end = fr4_start - 1

    # IMGT 近似（相对 FR4 位置推算）
    # CDR3: 通常从 ~位置 (fr4_start - 20) 到 cdr3_end
    cdr3_start = max(fr4_start - 20, 90)
    # CDR2: 约 56-65（IMGT）
    cdr2_start, cdr2_end = 56, 65
    # CDR1: 约 27-38（IMGT）
    cdr1_start, cdr1_end = 27, 38

    positions = []
    for s, e in [(cdr1_start, cdr1_end), (cdr2_start, cdr2_end), (cdr3_start, cdr3_end)]:
        if s <= len(seq):
            positions.extend(range(s, min(e + 1, len(seq) + 1)))

    return sorted(set(positions))


# ─── 突变生成 ─────────────────────────────────────────────────────────────────

def _generate_single_mutants(
    seq: str,
    cdr_positions: list[int],
    max_per_seq: int = 30,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """在 CDR 位置生成单点突变体（随机采样）。

    Args:
        seq: VHH 氨基酸序列
        cdr_positions: 1-indexed CDR 位置列表
        max_per_seq: 每条序列最多保留的突变体数（避免组合爆炸）
        seed: 随机种子（可重复性）

    Returns:
        list of (mutant_seq, mutation_label)，如 ("ACDE...", "K97R")
    """
    rng = random.Random(seed)
    all_mutants = []
    for pos in cdr_positions:
        idx = pos - 1  # 0-indexed
        if idx < 0 or idx >= len(seq):
            continue
        original_aa = seq[idx]
        for aa in AMINO_ACIDS:
            if aa == original_aa:
                continue
            mutant = seq[:idx] + aa + seq[idx + 1:]
            label = f"{original_aa}{pos}{aa}"
            all_mutants.append((mutant, label))
    rng.shuffle(all_mutants)
    return all_mutants[:max_per_seq]


# ─── Boltz-2 评分函数（Modal remote）────────────────────────────────────────

@app.function(
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={CACHE_DIR: BOLTZ_MODEL_VOLUME},
)
def score_batch(
    target_seq: str,
    vhh_seqs: list[str],
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    seed: int = 42,
) -> list[dict]:
    """用 Boltz-2 对一批 VHH 序列评分（复合物结构预测，返回 ipTM）。

    Args:
        target_seq: 目标抗原序列（单字母氨基酸）
        vhh_seqs: VHH 序列列表
        recycling_steps: Boltz-2 recycling 步数
        diffusion_samples: Boltz-2 扩散采样数（建议 1，节省时间）
        seed: 随机种子

    Returns:
        list of dicts，每条含 {seq, iptm, ptm, error}
        注：protein-protein 不提供 predicted_ddg（仅 protein-ligand affinity 支持）
    """
    import subprocess
    import yaml
    from tempfile import TemporaryDirectory

    os.environ["BOLTZ_CACHE"] = CACHE_DIR
    results = []

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i, vhh_seq in enumerate(vhh_seqs):
            out_dir = tmpdir / f"out_{i}"
            out_dir.mkdir()

            # Boltz-2 YAML 输入：目标蛋白 + VHH
            input_yaml = {
                "sequences": [
                    {"protein": {"id": "A", "sequence": target_seq}},
                    {"protein": {"id": "B", "sequence": vhh_seq}},
                ]
            }
            input_path = tmpdir / f"input_{i}.yaml"
            input_path.write_text(yaml.dump(input_yaml))

            cmd = [
                "boltz", "predict", str(input_path),
                "--out_dir", str(out_dir),
                "--seed", str(seed + i),
                "--recycling_steps", str(recycling_steps),
                "--diffusion_samples", str(diffusion_samples),
                "--accelerator", "gpu",
                "--devices", "1",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                print(f"  [WARN] Boltz-2 失败 seq_{i}: {result.stderr[-300:]}")
                results.append({"seq": vhh_seq, "iptm": 0.0, "ptm": 0.0, "error": True})
                continue

            # 解析 confidence JSON
            conf_files = list(out_dir.rglob("confidence_*.json"))
            if not conf_files:
                print(f"  [WARN] 未找到 confidence JSON: seq_{i}")
                results.append({"seq": vhh_seq, "iptm": 0.0, "ptm": 0.0, "error": True})
                continue

            with open(conf_files[0]) as f:
                conf = json.load(f)

            iptm = conf.get("iptm", conf.get("complex_plddt", 0.0))
            ptm = conf.get("ptm", 0.0)

            results.append({
                "seq": vhh_seq,
                "iptm": float(iptm),
                "ptm": float(ptm),
                "error": False,
            })
            print(f"  seq_{i}: iptm={iptm:.4f}")

    return results


# ─── 本地入口（进化主循环）───────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    target_seq: str,
    input_fasta: str,
    cdr_positions: str = "",
    hotspot_residues: str = "92,95,96,97,99",
    n_rounds: int = 3,
    top_n: int = 10,
    max_mutants_per_seq: int = 30,
    output_dir: str = "./out/boltz2_mber",
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    seed: int = 42,
):
    """Boltz-2 oracle MBER 进化优化主入口。

    Args:
        target_seq: 目标抗原氨基酸序列（单字母）
        input_fasta: 初始 VHH 候选 FASTA（来自 L4 developability 过滤后）
        cdr_positions: CDR 位置字符串，如 "26-33,50-57,97-115"（不传则自动检测）
        hotspot_residues: hotspot 残基 1-indexed 编号，逗号分隔（仅记录用）
        n_rounds: 进化轮数（默认 3）
        top_n: 每轮保留的 top 候选数（默认 10）
        max_mutants_per_seq: 每条序列最多生成的突变体数（默认 30）
        output_dir: 输出目录
        recycling_steps: Boltz-2 recycling 步数（默认 3）
        diffusion_samples: Boltz-2 扩散采样数（默认 1）
        seed: 随机种子（默认 42）
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取初始序列 ──
    seqs = []
    current_id = None
    for line in Path(input_fasta).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            current_id = line[1:]
        elif current_id and line:
            seqs.append(line)
            current_id = None

    if not seqs:
        print("[ERROR] input_fasta 中未读到任何序列，请检查 FASTA 格式")
        return

    # ── CDR 位置 ──
    if cdr_positions:
        cdr_pos = _parse_cdr_positions(cdr_positions)
        print(f"CDR 位置（传入）: {cdr_pos[:5]}...（共 {len(cdr_pos)} 个位置）")
    else:
        cdr_pos = _detect_cdr_positions(seqs[0])
        print(f"CDR 位置（自动检测）: {cdr_pos[:5]}...（共 {len(cdr_pos)} 个位置）")

    print(f"\n初始候选数: {len(seqs)}")
    print(f"进化轮数: {n_rounds}，top_n: {top_n}，max_mutants/seq: {max_mutants_per_seq}")
    print(f"目标序列长度: {len(target_seq)} aa")
    print(f"Hotspot 残基（记录用）: {hotspot_residues}")

    # 初始池：取 top_n
    current_pool = seqs[:top_n]

    for round_idx in range(n_rounds):
        print(f"\n{'=' * 60}")
        print(f"第 {round_idx + 1}/{n_rounds} 轮进化")
        print(f"{'=' * 60}")
        print(f"当前候选池: {len(current_pool)} 条")

        # ── 生成突变体 ──
        mutants_set: set[str] = set()
        for seq in current_pool:
            muts = _generate_single_mutants(seq, cdr_pos, max_mutants_per_seq, seed + round_idx)
            for m_seq, _ in muts:
                mutants_set.add(m_seq)

        # 合并原始 + 突变（去重）
        all_candidates = list(set(current_pool) | mutants_set)
        print(f"评分候选总数（原始 + 突变，去重）: {len(all_candidates)}")

        # ── Boltz-2 批量评分 ──
        print(f"提交 Boltz-2 评分...")
        results = score_batch.remote(
            target_seq=target_seq,
            vhh_seqs=all_candidates,
            recycling_steps=recycling_steps,
            diffusion_samples=diffusion_samples,
            seed=seed + round_idx * 100,
        )

        # ── 过滤 + 排序 ──
        valid = [r for r in results if not r.get("error", False)]
        valid.sort(key=lambda r: r["iptm"], reverse=True)

        # ── 保存本轮结果 ──
        round_out = out_dir / f"round_{round_idx + 1:02d}_scores.json"
        with open(round_out, "w") as f:
            json.dump(valid, f, indent=2)
        print(f"本轮结果写入: {round_out}（共 {len(valid)} 条有效）")
        if valid:
            top = valid[0]
            print(f"本轮 top1: iptm={top['iptm']:.4f}, ptm={top['ptm']:.4f}")

        # ── 更新候选池 ──
        current_pool = [r["seq"] for r in valid[:top_n]]
        if not current_pool:
            print("[WARN] 本轮无有效候选，终止进化")
            break

    # ── 输出最终 FASTA ──
    final_fasta = out_dir / "final_candidates.fasta"
    with open(final_fasta, "w") as f:
        for i, seq in enumerate(current_pool):
            f.write(f">boltz2_mber_{i + 1:03d}\n{seq}\n")
    print(f"\n最终候选写入: {final_fasta}（{len(current_pool)} 条）")

    # ── 汇总报告 ──
    summary = {
        "n_rounds_completed": round_idx + 1,
        "final_pool_size": len(current_pool),
        "top_iptm": valid[0]["iptm"] if valid else None,
        "hotspot_residues": hotspot_residues,
        "cdr_positions_used": cdr_pos,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"运行摘要: {out_dir / 'run_summary.json'}")
