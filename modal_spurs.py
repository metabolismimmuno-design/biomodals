# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
SPURS：ESM2 + ProteinMPNN "rewired" 生成式模型的 ΔΔG 预测，运行于 Modal GPU。

论文/代码：https://github.com/luo-group/SPURS （Nature Commun 2025）
原理：把 ProteinMPNN 的结构表征注入 ESM2 attention 层，在 Megascale (272k 单点突变)
      上监督微调做 ΔΔG 预测；论文声称对 stabilizing mutations 识别优于 SO / ThermoMPNN。

⚠️ 训练集与 ThermoMPNN 同为 Megascale，在 IgG Fc 这类 OOD 结构上仍可能 domain shift。
   使用前请用独立的 Hale 2024 benchmark 验证（见 stability_benchmark/）。

支持三种输入方式：
  1. 指定单点突变：--mutations F243L,S267E  （调用 single-mutation model）
  2. 全扫描模式：--scan-all  （单点 model 对所有位点扫 20 AA）
  3. 多点组合：--multi-mutations "S239D+I332E,G236A+S239D+I332E"  （调用 multi-mutation model）

用法：
  # 单点（与 stability_oracle / thermompnn 对齐 benchmark 集）
  modal run modal_spurs.py --pdb 3AVE_cleaned.pdb --chain A \
      --mutations F243L,S267E,T393A,E430G

  # 全扫描（生成与 SO `hale2024_clean_dimer/predictions.csv` 同 schema 的全 18pos×20AA 表）
  modal run modal_spurs.py --pdb 3AVE_cleaned.pdb --chain A --scan-all

  # 多点（Hale Level 3 加性测试）
  modal run modal_spurs.py --pdb 3AVE_cleaned.pdb --chain A \
      --multi-mutations "S239D+I332E,S239D+A330L+I332E,G236A+S239D+A330L+I332E"

输出：
  {out_dir}/{run_name}/predictions.csv
  列：pdb_code, chain_id, mutation, exp_ddG, pred_ddG
  （pred_ddG < 0 = 稳定化；> 0 = 去稳定化；exp_ddG 恒为 1.0 占位，与 SO 输出对齐）
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from subprocess import run as sp_run

from modal import App, Image, Volume

REPO_DIR = "/app/SPURS"
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))  # 分钟；scan-all 对大蛋白可能较慢

# Volume 用于缓存 HF 自动下载的 SPURS 权重（单次下载约数百 MB，warm 启动受益）
HF_CACHE = "/cache/huggingface"

app = App("spurs")
weights_vol = Volume.from_name("spurs-models", create_if_missing=True)

# ── Image 构建 ──────────────────────────────────────────────────────────────
# SPURS README 官方栈是 python 3.7 + torch 1.12.0+cu113，但 Modal runner 要求
# 容器 Python ≥3.9，必须弃用 3.7 路径。改用 PyTorch 2.0.1 + Python 3.10 基础镜像：
# SPURS 核心是 ESM2 + ProteinMPNN 的 rewiring（纯 PyTorch，无深度框架 API 依赖），
# 经验上 torch 2.x 向后兼容 1.12 代码。若遇到 API 漂移，fallback 方案见下方注释。
image = (
    Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime")
    .apt_install("git", "build-essential", "wget")
    # 1) Clone SPURS 仓库
    .run_commands(
        f"git clone --depth 1 https://github.com/luo-group/SPURS.git {REPO_DIR}"
    )
    # 2) 安装 SPURS —— 用 --no-deps 避免它拉回 torch 1.12 覆盖基础镜像的 torch 2.0
    .run_commands(f"cd {REPO_DIR} && pip install --no-deps -e .")
    # 3) 补齐 SPURS 运行时必需的依赖（从 setup.py 手动抽取，跳过 torch 相关）
    .pip_install(
        [
            "fair-esm",             # facebook ESM (不用 git 版，PyPI 已有稳定 2.0.1)
            "biopython==1.81",
            "numpy<1.25",           # torch 2.0.1 兼容上限
            "pandas",
            "huggingface_hub",      # get_SPURS_from_hub 依赖
            "einops",
            "e3nn",
            "atom3d",               # SPURS parse_pdb 用
            "hydra-core",
            "omegaconf",
            "pytorch-lightning==1.7.3",  # SPURS 硬依赖 _FAIRSCALE_AVAILABLE + LightningLoggerBase，仅 1.7.x 可用
            "torchmetrics<1.0",           # 匹配 pl 1.7.3 era
        ]
    )
    # pl 1.7.3 捎带的 torchtext 0.13.0 为 torch 1.12 编译，与 torch 2.0 二进制冲突
    # 必须换成 torch 2.0.1 匹配的 torchtext 0.15.2（SPURS datamodules 会直接 import torchtext）
    .run_commands(
        "pip uninstall -y torchtext torchdata || true",
        "pip install torchtext==0.15.2 torchdata==0.6.1",
    )
    .env({
        "PYTHONPATH": REPO_DIR,
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": HF_CACHE,
        "TORCH_HOME": HF_CACHE,
    })
)


# ── 工具函数 ───────────────────────────────────────────────────────────────
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"  # 字母表索引，SPURS/ESM 通用顺序

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _parse_single_mutation(s: str) -> tuple[str, int, str]:
    """'F243L' → ('F', 243, 'L')。"""
    s = s.strip()
    if len(s) < 3:
        raise ValueError(f"突变格式错误: {s}，应为 W123M 形式")
    wt, mut = s[0].upper(), s[-1].upper()
    try:
        pos = int(s[1:-1])
    except ValueError:
        raise ValueError(f"突变 {s} 残基编号解析失败")
    if wt not in AA_ORDER or mut not in AA_ORDER:
        raise ValueError(f"突变 {s} 含非标准氨基酸")
    return wt, pos, mut


# ── Modal 远程函数 ──────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={HF_CACHE: weights_vol},
)
def run_single_model(
    pdb_content: bytes,
    pdb_name: str,
    chain: str,
    mutations: list[str] | None,   # None = scan-all
) -> bytes:
    """
    SPURS single-mutation 模型推理。

    返回 CSV 字节：列 pdb_code, chain_id, mutation, exp_ddG, pred_ddG
    - 指定 mutations：只输出这些突变
    - mutations=None：全位点 × 20 AA 扫描
    """
    import torch
    from Bio import PDB as biopdb  # noqa: F401
    from spurs.inference import get_SPURS_from_hub, parse_pdb

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / pdb_name
        pdb_path.write_bytes(pdb_content)
        pdb_code = pdb_path.stem

        # 加载模型（首次会从 HF 下载到 HF_CACHE volume，之后 warm）
        model, cfg = get_SPURS_from_hub()
        model = model.to(device)
        model.eval()

        # 解析 PDB → SPURS 内部格式
        pdb = parse_pdb(str(pdb_path), pdb_code, chain=chain, cfg=cfg)
        # 把 tensor 字段搬到 device（SPURS parse_pdb 产物多为 dict of tensors）
        for k, v in list(pdb.items()):
            if hasattr(v, "to"):
                pdb[k] = v.to(device)

        # 推理：返回 shape [L, 20] 的 ΔΔG 矩阵，WT 位点自身为 0
        with torch.no_grad():
            ddg = model(pdb, return_logist=True)   # SPURS README 原生参数名
        ddg = ddg.detach().cpu().numpy()

        # ── 读 PDB 解析原生残基编号 + WT 序列（用于行号 → EU 编号映射）──
        # SPURS 的 ddg 行序与 parse_pdb 解析的残基顺序一致；为了输出 EU 编号，
        # 再独立解析一次 PDB 构建 (row_idx → (res_seq_num, wt_1letter)) 映射
        parser = biopdb.PDBParser(QUIET=True)
        struct = parser.get_structure("p", str(pdb_path))
        model_bio = list(struct.get_models())[0]
        chain_obj = model_bio[chain]
        row_to_res: list[tuple[int, str]] = []
        for res in chain_obj.get_residues():
            if res.get_id()[0] != " ":
                continue  # 跳过 HETATM
            wt1 = AA3_TO_1.get(res.get_resname().upper())
            if wt1 is None:
                continue  # 非标准 AA
            row_to_res.append((res.get_id()[1], wt1))

        if len(row_to_res) != ddg.shape[0]:
            # SPURS 内部过滤非标准残基后可能与 biopython 计数不一致
            # 此时不抛错，改为只取较短者（对齐首部）并在 stderr 记录
            print(
                f"[WARN] ddg 行数 {ddg.shape[0]} != biopython 残基数 {len(row_to_res)}，"
                "按较短者对齐（可能影响高位点映射准确性）"
            )
            n = min(ddg.shape[0], len(row_to_res))
            row_to_res = row_to_res[:n]
            ddg = ddg[:n]

        # 构建输出表
        rows: list[dict] = []
        if mutations is None:
            # 全扫描：每个位点 × 20 AA
            for i, (res_num, wt1) in enumerate(row_to_res):
                for j, mut1 in enumerate(AA_ORDER):
                    if mut1 == wt1:
                        continue
                    rows.append({
                        "pdb_code": pdb_code,
                        "chain_id": chain,
                        "mutation": f"{wt1}{res_num}{mut1}",
                        "exp_ddG": 1.0,
                        "pred_ddG": round(float(ddg[i, j]), 4),
                    })
        else:
            # 指定突变
            res_to_row = {res_num: i for i, (res_num, _) in enumerate(row_to_res)}
            row_wt = {i: wt for i, (_, wt) in enumerate(row_to_res)}
            for mut_str in mutations:
                wt, pos, mut = _parse_single_mutation(mut_str)
                if pos not in res_to_row:
                    print(f"[WARN] {mut_str} 位点 {pos} 不在 PDB 链 {chain}，跳过")
                    continue
                i = res_to_row[pos]
                if row_wt[i] != wt:
                    print(
                        f"[WARN] {mut_str} WT 不匹配：PDB 链 {chain} 位 {pos} 实际为 "
                        f"{row_wt[i]}，跳过"
                    )
                    continue
                j = AA_ORDER.index(mut)
                rows.append({
                    "pdb_code": pdb_code,
                    "chain_id": chain,
                    "mutation": mut_str,
                    "exp_ddG": 1.0,
                    "pred_ddG": round(float(ddg[i, j]), 4),
                })

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["pdb_code", "chain_id", "mutation", "exp_ddG", "pred_ddG"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={HF_CACHE: weights_vol},
)
def run_multi_model(
    pdb_content: bytes,
    pdb_name: str,
    chain: str,
    combos: list[list[str]],   # 每个 combo = ['S239D', 'I332E']
) -> bytes:
    """SPURS multi-mutation 模型推理。

    返回 CSV 字节：每个组合一行，mutation 列为 'S239D+I332E' 格式。
    """
    import torch
    from spurs.inference import (
        get_SPURS_multi_from_hub,
        parse_pdb,
        parse_pdb_for_mutation,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / pdb_name
        pdb_path.write_bytes(pdb_content)
        pdb_code = pdb_path.stem

        model, cfg = get_SPURS_multi_from_hub()
        model = model.to(device)
        model.eval()

        pdb = parse_pdb(str(pdb_path), pdb_code, chain=chain, cfg=cfg)
        for k, v in list(pdb.items()):
            if hasattr(v, "to"):
                pdb[k] = v.to(device)

        # SPURS multi-model 的 parse_pdb_for_mutation 内部用 torch.tensor 堆叠，
        # 要求所有组合等长。按 combo 长度分组，分别推理
        from collections import defaultdict
        by_len: dict[int, list[list[str]]] = defaultdict(list)
        for c in combos:
            by_len[len(c)].append(c)

        for n, group in by_len.items():
            pdb_g = dict(pdb)   # 每组共享 PDB 特征，只替换 mut_ids / append_tensors
            mut_ids, append_tensors = parse_pdb_for_mutation(group)
            pdb_g["mut_ids"] = mut_ids
            pdb_g["append_tensors"] = append_tensors.to(device)
            with torch.no_grad():
                ddg = model(pdb_g)
            ddg = ddg.detach().cpu().numpy().reshape(-1)
            if len(ddg) != len(group):
                print(f"[WARN] {n}-点组输出 {len(ddg)} != 输入 {len(group)}")
            for combo, d in zip(group, ddg):
                rows.append({
                    "pdb_code": pdb_code,
                    "chain_id": chain,
                    "mutation": "+".join(combo),
                    "exp_ddG": 1.0,
                    "pred_ddG": round(float(d), 4),
                })

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["pdb_code", "chain_id", "mutation", "exp_ddG", "pred_ddG"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


# ── 本地入口 ───────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    pdb: str = "",
    mutations: str = "",
    multi_mutations: str = "",
    scan_all: bool = False,
    chain: str = "A",
    out_dir: str = "./out/spurs",
    run_name: str = "",
):
    """在 Modal GPU 上运行 SPURS 预测（单点 / 多点 / 全扫描）。

    Args:
        pdb: 输入 PDB 文件路径（必须）
        mutations: 逗号分隔单点突变，格式 W123M（如 F243L,S267E）
        multi_mutations: 逗号分隔的多点组合，组合内用 + 连接
                        （如 "S239D+I332E,G236A+S239D+I332E"）
        scan_all: 全位点 × 20 AA 扫描（忽略 mutations / multi_mutations）
        chain: 目标链 ID，默认 'A'
        out_dir: 输出目录
        run_name: 运行名称（用于子目录）
    """
    if not pdb:
        raise ValueError("--pdb 参数必须提供")

    pdb_path = Path(pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB 不存在: {pdb}")

    # 输入模式互斥检查
    modes = [bool(scan_all), bool(mutations), bool(multi_mutations)]
    if sum(modes) != 1:
        raise ValueError(
            "必须且只能选一种模式：--scan-all / --mutations / --multi-mutations"
        )

    run_name = run_name or datetime.now().strftime("%y%m%d%H%M")
    out_path = Path(out_dir) / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"输入 PDB: {pdb}")
    print(f"链: {chain}")
    print(f"输出目录: {out_path}")

    if scan_all:
        print("模式: 全扫描")
        csv_bytes = run_single_model.remote(
            pdb_content=pdb_path.read_bytes(),
            pdb_name=pdb_path.name,
            chain=chain,
            mutations=None,
        )
    elif mutations:
        mut_list = [m.strip() for m in mutations.split(",") if m.strip()]
        print(f"模式: 指定单点，共 {len(mut_list)} 个突变")
        csv_bytes = run_single_model.remote(
            pdb_content=pdb_path.read_bytes(),
            pdb_name=pdb_path.name,
            chain=chain,
            mutations=mut_list,
        )
    else:
        combos = []
        for group in multi_mutations.split(","):
            group = group.strip()
            if not group:
                continue
            combos.append([m.strip() for m in group.split("+") if m.strip()])
        print(f"模式: 多点，共 {len(combos)} 个组合")
        csv_bytes = run_multi_model.remote(
            pdb_content=pdb_path.read_bytes(),
            pdb_name=pdb_path.name,
            chain=chain,
            combos=combos,
        )

    out_csv = out_path / "predictions.csv"
    out_csv.write_bytes(csv_bytes)
    print(f"\n完成！结果 → {out_csv}")

    # 终端摘要
    reader = csv.DictReader(io.StringIO(csv_bytes.decode()))
    rows = list(reader)
    if rows and len(rows) <= 30:
        print(f"\n{'mutation':<25} {'pred_ΔΔG':>10}  解读")
        print("-" * 55)
        for row in rows:
            d = float(row["pred_ddG"])
            tag = "稳定化 ✓" if d < -0.5 else ("去稳定化 ✗" if d > 0.5 else "中性 ~")
            print(f"{row['mutation']:<25} {d:>10.3f}  {tag}")
    elif rows:
        print(f"（共 {len(rows)} 行，略 — 请查看 {out_csv}）")
