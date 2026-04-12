# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
Stability Oracle：基于 3D-ResNet 的蛋白质突变稳定性预测（ΔΔG），运行于 Modal GPU。

论文/代码: https://github.com/danny305/StabilityOracle (Nature Comms 2024)
原理：以残基局部原子微环境（半径 12 Å，最多 512 个原子）为输入，预测氨基酸替换的 ΔΔG。

支持两种输入方式：
1. 指定突变列表：--mutations A123G,V45I（只对这些位点做预测）
2. 全扫描模式：--scan-all（对所有残基扫描 20 种 AA 替换）

用法：
  # 预测特定突变
  modal run modal_stability_oracle.py --pdb protein.pdb --mutations A123G,V45I

  # 全序列扫描（输出每个位点所有替换的 ΔΔG）
  modal run modal_stability_oracle.py --pdb protein.pdb --scan-all

  # 指定链
  modal run modal_stability_oracle.py --pdb protein.pdb --mutations A123G --chain A

输出：
  {out_dir}/{run_name}/predictions.csv
  列：chain_id, res_seq_num, wt_aa, mut_aa, pred_ddg
  负值 = 稳定化突变；正值 = 去稳定化突变
"""

import json
import os
import tempfile
from pathlib import Path
from subprocess import run as sp_run

from modal import App, Image

REPO_DIR = "/app/StabilityOracle"
MODEL_CKPT = f"{REPO_DIR}/models/stability_oracle_cdna117k.pt"
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))

app = App("stability_oracle")

image = (
    Image.from_registry("pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime")
    .apt_install("git", "wget", "build-essential")
    # ── 1. Clone StabilityOracle ──────────────────────────────────────────────
    .run_commands(
        f"git clone --depth 1 https://github.com/danny305/StabilityOracle.git {REPO_DIR}"
    )
    # ── 2. 安装依赖 ────────────────────────────────────────────────────────────
    .pip_install(
        [
            "timm==0.9.12",
            "scikit-learn==1.4.0",
            "numpy==1.26.3",
            "pandas==2.2.0",
            "numba==0.59.0",
            "biopython==1.83",
            "freesasa==2.2.1",
            "scipy",
            "matplotlib",
            "umap-learn",
        ]
    )
    # ── 3. StabilityOracle 无 setup.py，直接加入 PYTHONPATH ──────────────────
    .env({"PYTHONPATH": REPO_DIR})
)

# ─────────────────────────────────────────────────────────────────────────────
# PDB → JSONL 转换（在 Modal 容器内执行）
# ─────────────────────────────────────────────────────────────────────────────

def _three_to_one(resname: str) -> str:
    """三字母 → 单字母氨基酸代码。"""
    AA3 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return AA3.get(resname.upper(), "X")


def _one_to_three(aa1: str) -> str:
    """单字母 → 三字母氨基酸代码。"""
    AA1 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    return AA1.get(aa1.upper(), "UNK")


def _build_jsonl(pdb_path: str, chain: str, positions: list[tuple]) -> str:
    """
    将 PDB 文件转换为 Stability Oracle 所需的 JSONL 格式。

    Args:
        pdb_path: PDB 文件路径
        chain: 目标链 ID（如 'A'）
        positions: [(res_seq_num, wt_aa_1letter), ...] 要预测的位点列表

    Returns:
        JSONL 字符串（每行一个 JSON 对象，对应一个残基位点）
    """
    import numpy as np
    from Bio import PDB as biopdb

    try:
        import freesasa
        HAS_FREESASA = True
    except ImportError:
        HAS_FREESASA = False

    parser = biopdb.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = list(structure.get_models())[0]

    # 收集目标链所有重原子
    chain_obj = model[chain]

    # 计算每个原子的 SASA（如果 freesasa 可用）
    sasa_by_atom = {}
    if HAS_FREESASA:
        try:
            fs_struct = freesasa.Structure(pdb_path)
            fs_result = freesasa.calc(fs_struct)
            # 建立 (chain, resnum, atom_name) → SASA 映射
            for i in range(fs_struct.nAtoms()):
                key = (
                    fs_struct.chainLabel(i),
                    fs_struct.residueNumber(i).strip(),
                    fs_struct.atomName(i).strip(),
                )
                sasa_by_atom[key] = fs_result.atomArea(i)
        except Exception:
            pass  # SASA 计算失败时退回 0.0

    # 构建全链原子列表
    all_atoms = []
    for res in chain_obj.get_residues():
        if res.get_id()[0] != " ":
            continue  # 跳过 HETATM
        res_num = res.get_id()[1]
        res_name = res.get_resname()
        for atom in res.get_atoms():
            if atom.element == "H":
                continue  # 跳过氢原子
            coord = atom.get_coord()
            sasa_key = (chain, str(res_num), atom.get_name().strip())
            sasa_val = sasa_by_atom.get(sasa_key, 0.0)
            all_atoms.append({
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
                "element": atom.element or "C",
                "res_seq_num": res_num,
                "chain_id": chain,
                "atom_name": atom.get_name().strip(),
                "res_name": res_name,
                "_atom_site.fw2_charge": "0.0",
                "_atom_site.FreeSASA_value": str(round(sasa_val, 4)),
            })

    all_coords = np.array([[a["x"], a["y"], a["z"]] for a in all_atoms])
    pdb_name = Path(pdb_path).name

    records = []
    for i, (res_num, wt_aa1) in enumerate(positions):
        wt_aa3 = _one_to_three(wt_aa1)

        # 找目标残基的 Cα 原子
        ca_global_idx = None
        for i, a in enumerate(all_atoms):
            if a["res_seq_num"] == res_num and a["chain_id"] == chain and a["atom_name"] == "CA":
                ca_global_idx = i
                break

        if ca_global_idx is None:
            continue  # 找不到 Cα，跳过

        ca_coord = all_coords[ca_global_idx]

        # 选择局部环境：距离 Cα 12 Å 以内的所有原子
        dists = np.linalg.norm(all_coords - ca_coord, axis=1)
        env_mask = dists <= 12.0
        env_indices_sorted = np.where(env_mask)[0][np.argsort(dists[env_mask])]

        # 目标残基自身的原子索引（将被 loader 掩码）
        target_indices_global = [
            i for i, a in enumerate(all_atoms)
            if a["res_seq_num"] == res_num and a["chain_id"] == chain
        ]
        target_indices_global_set = set(target_indices_global)

        # 构建局部环境原子列表（保留目标残基原子，loader 会自行移除）
        env_atoms = [all_atoms[i] for i in env_indices_sorted[:512]]

        # 在 env_atoms 中找 Cα 的新索引
        ca_local_idx = None
        for j, a in enumerate(env_atoms):
            if (a["res_seq_num"] == res_num and a["chain_id"] == chain
                    and a["atom_name"] == "CA"):
                ca_local_idx = j
                break

        if ca_local_idx is None:
            continue

        # 目标残基在局部列表中的原子索引
        target_local = [
            j for j, a in enumerate(env_atoms)
            if a["res_seq_num"] == res_num and a["chain_id"] == chain
        ]

        atomic_collection = {
            "x": [a["x"] for a in env_atoms],
            "y": [a["y"] for a in env_atoms],
            "z": [a["z"] for a in env_atoms],
            "element": [a["element"] for a in env_atoms],
            "res_seq_num": [a["res_seq_num"] for a in env_atoms],
            "chain_id": [a["chain_id"] for a in env_atoms],
            "_atom_site.fw2_charge": [a["_atom_site.fw2_charge"] for a in env_atoms],
            "_atom_site.FreeSASA_value": [a["_atom_site.FreeSASA_value"] for a in env_atoms],
        }

        record = {
            "snapshot": {
                "filename": pdb_name,
                "model_id": 1,
                "label": wt_aa3,
                "wt_aa": wt_aa3,
                "res_seq_num": res_num,
                "chain_id": chain,
                "ddg": 1.0 if i % 2 == 0 else -1.0,  # 交替假标签，保证 ROC AUC 可算
            },
            "atomic_collection": atomic_collection,
            "target_alpha_carbon": ca_local_idx,
            "target": target_local,
        }
        records.append(json.dumps(record))

    return "\n".join(records)


# ─────────────────────────────────────────────────────────────────────────────
# Modal 远程函数
# ─────────────────────────────────────────────────────────────────────────────

@app.function(image=image, gpu=GPU, timeout=TIMEOUT * 60)
def run_predictions(
    pdb_content: bytes,
    pdb_name: str,
    chain: str,
    positions: list,
) -> bytes:
    """在 Modal GPU 上运行 Stability Oracle 预测。

    Args:
        pdb_content: PDB 文件字节内容
        pdb_name: PDB 文件名
        chain: 目标链 ID
        positions: [(res_seq_num, wt_aa_1letter), ...] 待预测位点

    Returns:
        predictions CSV 文件的字节内容
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / pdb_name
        pdb_path.write_bytes(pdb_content)

        # 构建 JSONL
        jsonl_str = _build_jsonl(str(pdb_path), chain, positions)
        if not jsonl_str.strip():
            raise ValueError("未找到任何可预测的残基位点，请检查链 ID 和残基编号")

        jsonl_path = Path(tmpdir) / "input.jsonl"
        jsonl_path.write_text(jsonl_str)

        out_dir = Path(tmpdir) / "output"
        out_dir.mkdir()

        # 运行 Stability Oracle
        cmd = [
            "python",
            f"{REPO_DIR}/scripts/run_stability_oracle.py",
            "--model-ckpt", MODEL_CKPT,
            "--batch-size", "64",
            "--use-clstoken",
            "--outdir", str(out_dir),
            "--dataset", str(jsonl_path),
        ]
        result = sp_run(cmd, cwd=REPO_DIR, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:])
            print("STDERR:", result.stderr[-2000:])
            raise RuntimeError(
                f"Stability Oracle failed (code {result.returncode})\n"
                f"STDERR: {result.stderr[-500:]}"
            )

        # 找输出 CSV（flat 格式包含所有 AA 替换）
        flat_csvs = list(out_dir.glob("*_flat.csv"))
        if not flat_csvs:
            # fallback：找任意 csv
            flat_csvs = list(out_dir.glob("*.csv"))
        if not flat_csvs:
            raise FileNotFoundError(
                f"输出 CSV 未找到，目录内容：{list(out_dir.iterdir())}"
            )

        return flat_csvs[0].read_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# 本地入口
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    pdb: str = "",
    mutations: str = "",
    scan_all: bool = False,
    chain: str = "A",
    out_dir: str = "./out/stability_oracle",
    run_name: str = None,
):
    """本地入口：在 Modal GPU 上运行 Stability Oracle 突变稳定性预测。

    Args:
        pdb: 输入 PDB 文件路径（必须）
        mutations: 逗号分隔的突变列表，格式 W{pos}M（如 A123G,V45I）
                  wt 氨基酸 + 残基编号 + mut 氨基酸，单字母代码
        scan_all: 若为 True，对所有残基做全 20 AA 扫描（忽略 mutations）
        chain: 目标链 ID，默认 'A'
        out_dir: 输出目录，默认 ./out/stability_oracle
        run_name: 运行名称（用于子目录命名）
    """
    from datetime import datetime

    from Bio import PDB as biopdb

    if not pdb:
        raise ValueError("--pdb 参数必须提供")

    pdb_path = Path(pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB 文件不存在: {pdb}")

    # 解析位点
    if scan_all:
        # 全扫描：从 PDB 读取所有残基
        parser = biopdb.PDBParser(QUIET=True)
        structure = parser.get_structure("p", str(pdb_path))
        model = list(structure.get_models())[0]
        try:
            chain_obj = model[chain]
        except KeyError:
            raise ValueError(f"PDB 中找不到链 '{chain}'")
        positions = []
        for res in chain_obj.get_residues():
            if res.get_id()[0] != " ":
                continue
            res_num = res.get_id()[1]
            wt1 = _three_to_one(res.get_resname())
            if wt1 != "X":
                positions.append((res_num, wt1))
        print(f"全扫描模式：链 {chain}，共 {len(positions)} 个残基")
    elif mutations:
        # 解析指定突变，格式：W123M（wt + resnum + mut）
        positions = []
        for mut in mutations.split(","):
            mut = mut.strip()
            if len(mut) < 3:
                raise ValueError(f"突变格式错误: {mut}，应为 W123M 形式")
            wt1 = mut[0]
            mut1 = mut[-1]
            try:
                res_num = int(mut[1:-1])
            except ValueError:
                raise ValueError(f"突变格式错误: {mut}，残基编号解析失败")
            positions.append((res_num, wt1))
        print(f"指定突变模式：{mutations}")
    else:
        raise ValueError("请提供 --mutations 或使用 --scan-all")

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_path = Path(out_dir) / (run_name or today)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"输入 PDB: {pdb}")
    print(f"链: {chain}，位点数: {len(positions)}")
    print(f"输出目录: {out_path}")

    # 运行预测
    csv_bytes = run_predictions.remote(
        pdb_content=pdb_path.read_bytes(),
        pdb_name=pdb_path.name,
        chain=chain,
        positions=positions,
    )

    # 保存结果
    out_csv = out_path / "predictions.csv"
    out_csv.write_bytes(csv_bytes)
    print(f"\n完成！预测结果 → {out_csv}")

    # 如果是指定突变，打印摘要
    if mutations and not scan_all:
        import csv
        import io
        reader = csv.DictReader(io.StringIO(csv_bytes.decode()))
        rows = list(reader)
        if rows:
            print(f"\n{'突变':<12} {'pred_ΔΔG':>10}  {'解读':>20}")
            print("-" * 46)
            for row in rows:
                try:
                    ddg = float(row.get("pred_ddg") or row.get("pred_ddG") or 0)
                    interp = "稳定化 ✓" if ddg < -0.5 else ("去稳定化 ✗" if ddg > 0.5 else "中性 ~")
                    mut_label = row.get("mutation", "?")
                    print(f"{mut_label:<12} {ddg:>10.3f}  {interp:>20}")
                except (ValueError, KeyError):
                    pass
