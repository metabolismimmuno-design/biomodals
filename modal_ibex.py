# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Ibex + ABodyBuilder3  VHH / 抗体单体结构预测
GitHub: https://github.com/prescient-design/ibex
许可证: Genentech Apache 2.0 Non-Commercial（仅限非商业研究用途）

Ibex 是 Prescient Design (Genentech) 的抗体结构预测工具：
- 支持 VHH（纳米抗体）、传统抗体（VH+VL）、TCR
- Apo / holo 构象感知：--apo 预测未结合态，默认预测结合态
- CDR-H3 RMSD 2.28 Å，内置 ABodyBuilder3 权重（--use-abodybuilder3）
- Python ≥ 3.10 硬性要求

## 首次使用：下载权重到 Modal Volume（约 556 MB，仅需一次）
推荐方式 A（从 Zenodo 直接下载，美国机房速度快）：
  ~/Library/Python/3.9/bin/modal run ~/biomodals/modal_ibex.py --seed-weights

备用方式 B（本机下载后上传，适合 Zenodo 访问受限时）：
  wget https://zenodo.org/records/15866556/files/ibex_v1.ckpt
  wget https://zenodo.org/records/15866556/files/abb3.ckpt
  modal volume put ibex-weights ibex_v1.ckpt /ibex_v1.ckpt
  modal volume put ibex-weights abb3.ckpt /abb3.ckpt

## 单序列预测
VHH（只传 heavy）：
  ~/Library/Python/3.9/bin/modal run ~/biomodals/modal_ibex.py \\
    --heavy "QVQLVESGGGLVQPGGSLRLSCAASGFTFS..."

抗体（heavy + light）：
  ~/Library/Python/3.9/bin/modal run ~/biomodals/modal_ibex.py \\
    --heavy "QVQLVESG..." --light "DIVMTQSP..."

## 批量预测（FASTA 文件）
VHH FASTA（每条单链序列）：
  ~/Library/Python/3.9/bin/modal run ~/biomodals/modal_ibex.py --input-faa vhh_list.faa

抗体 FASTA（相邻两条序列，heavy 在前）：
  >ab1_heavy
  QVQLVESG...
  >ab1_light
  DIVMTQSP...

## 批量预测（CSV 文件）
CSV 列：id, fv_heavy, fv_light（VHH 的 fv_light 留空）：
  ~/Library/Python/3.9/bin/modal run ~/biomodals/modal_ibex.py --input-csv seqs.csv

## 切换模型
ABodyBuilder3（更快，精度略低）：加 --use-abodybuilder3
Apo 构象（游离态，用于 docking 起始构象）：加 --apo

## 查询权重缓存位置（smoke test 期间使用）
  ~/Library/Python/3.9/bin/modal run ~/biomodals/modal_ibex.py --probe-cache
"""

import io
import os
import zipfile
from pathlib import Path
from typing import Optional

from modal import App, Image, Volume

# ── 运行配置 ──────────────────────────────────────────────────────────────────
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))  # 分钟

# ── Zenodo 权重文件 ────────────────────────────────────────────────────────────
ZENODO_BASE = "https://zenodo.org/records/15866556/files"
IBEX_CKPT = "ibex_v1.ckpt"      # 468.7 MB
ABB3_CKPT = "abb3.ckpt"         # 87.0 MB

# ── Modal Volume 持久化权重 ───────────────────────────────────────────────────
# ibex 默认 cache = ~/.cache/ibex（见 ibex/utils.py get_checkpoint_path）
# 文件名格式：{url_filename}_{md5(url)[:8]}，不能直接放裸文件名
# 挂载到默认 cache 目录，让 ibex 自己的逻辑处理路径
WEIGHTS_MOUNT = "/root/.cache/ibex"
ibex_vol = Volume.from_name("ibex-weights", create_if_missing=True)

# ── 镜像：Python 3.10（Ibex 硬性要求）+ PyTorch CUDA + prescient-ibex ─────────
image = (
    Image.debian_slim(python_version="3.10")
    # 先装 CUDA-enabled PyTorch，避免 prescient-ibex 拉 CPU 版
    .pip_install(
        "torch>=2.0.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # biotite Modal mirror 在 14.7/35.8 MB 处截断（已复现两次），改从 PyPI 直接拉
    .run_commands(
        "pip install biotite "
        "--index-url https://pypi.org/simple "
        "--trusted-host pypi.org --trusted-host files.pythonhosted.org"
    )
    .pip_install("prescient-ibex")
)

app = App("ibex", image=image)


# ── 工具函数（本地运行，不在 Modal 容器内） ───────────────────────────────────


def _parse_fasta(fasta_content: str):
    """解析 FASTA 文件，返回 [(name, heavy_seq, light_seq)] 列表。

    规则：
    - 单条序列 → VHH（light_seq = ""）
    - 连续两条序列且 ID 含 _heavy/_H 后缀 → 配对抗体（第一条 heavy，第二条 light）
    - 连续两条序列无特殊后缀 → 也视为配对抗体（兼容无后缀 FASTA）
    """
    entries = []
    current_id = None
    current_seq = []

    for line in fasta_content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                entries.append((current_id, "".join(current_seq)))
            current_id = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        entries.append((current_id, "".join(current_seq)))

    pairs = []
    i = 0
    while i < len(entries):
        name, seq = entries[i]
        name_lower = name.lower()
        # 检测 heavy 标记
        is_heavy = any(
            name_lower.endswith(s) for s in ("_heavy", "_h", "-heavy", "-h", "_hc", "-hc")
        )
        if is_heavy and i + 1 < len(entries):
            _, light_seq = entries[i + 1]
            # 去掉后缀得到基础名
            base = name
            for suffix in ("_heavy", "_H", "-heavy", "-H", "_hc", "-hc"):
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    break
            pairs.append((base, seq, light_seq))
            i += 2
        else:
            # 单链 → VHH
            pairs.append((name, seq, ""))
            i += 1

    return pairs


# ── Modal 函数 ─────────────────────────────────────────────────────────────────


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={WEIGHTS_MOUNT: ibex_vol},
)
def predict_single(
    fv_heavy: str,
    fv_light: Optional[str] = None,
    name: str = "prediction",
    use_abodybuilder3: bool = False,
    apo: bool = False,
    refine: bool = False,
) -> bytes:
    """预测单个 VHH 或抗体的 3D 结构，返回 PDB 文件内容（bytes）。

    Args:
        fv_heavy: 重链 / VHH 氨基酸序列。
        fv_light: 轻链序列；VHH 传 None 或 ""。
        name: 输出文件名前缀。
        use_abodybuilder3: True → 使用 ABodyBuilder3（更快，精度略低）。
        apo: True → 预测 apo（未结合）构象；默认 holo（结合构象）。
        refine: True → OpenMM 精细化（更准，慢约 2x，需额外安装 openmm）。

    Returns:
        PDB 文件内容 bytes。
    """
    import subprocess
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        out_pdb = Path(tmp) / f"{name}.pdb"

        # 优先尝试 Python API（避免 subprocess overhead）
        model_name = "abodybuilder3" if use_abodybuilder3 else "ibex"
        light = fv_light or ""

        # 探测 Volume 内是否已有缓存（ibex 自管文件名，只看目录非空）
        cache_dir = Path(WEIGHTS_MOUNT)
        cached = any(cache_dir.iterdir()) if cache_dir.exists() else False
        print(
            f"模型={model_name}  VHH={'是' if not light else '否'}  "
            f"apo={apo}  refine={refine}  "
            f"cache={'存在' if cached else '不存在，将自动下载'}"
        )

        try:
            import torch
            from ibex import Ibex, inference  # type: ignore

            # ABB3 checkpoint 使用 ml_collections.ConfigDict，PyTorch 2.x 的
            # weights_only=True 默认会阻止加载，需提前加入 safe globals
            try:
                from ml_collections.config_dict.config_dict import ConfigDict
                torch.serialization.add_safe_globals([ConfigDict])
            except Exception:
                pass  # ibex_v1.ckpt 不需要此 patch，忽略失败

            # cache 挂载到 ~/.cache/ibex，from_pretrained 自动命中
            model = Ibex.from_pretrained(model_name)
            inference(model, fv_heavy, light, str(out_pdb), apo=apo, refine=refine)

        except Exception as e:
            print(f"Python API 失败: {e}\n退回 CLI 方式...")
            cmd = ["ibex", "--fv-heavy", fv_heavy, "--output", str(out_pdb)]
            if light:
                cmd += ["--fv-light", light]
            if use_abodybuilder3:
                cmd += ["--abodybuilder3"]
            if apo:
                cmd += ["--apo"]
            if refine:
                cmd += ["--refine"]
            subprocess.run(cmd, check=True)

        if not out_pdb.exists():
            raise RuntimeError(f"预测失败：未生成 PDB 文件 {out_pdb}")

        size_kb = out_pdb.stat().st_size / 1024
        print(f"输出 PDB: {out_pdb.name}  ({size_kb:.1f} KB)")
        return out_pdb.read_bytes()


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={WEIGHTS_MOUNT: ibex_vol},
)
def predict_batch_csv(
    csv_content: str,
    use_abodybuilder3: bool = False,
    apo: bool = False,
) -> bytes:
    """批量预测（CSV 输入），返回 ZIP（含所有 PDB 文件）。

    Args:
        csv_content: CSV 文件内容字符串；必须包含列 id, fv_heavy, fv_light。
            VHH 的 fv_light 列留空即可。
        use_abodybuilder3: 使用 ABodyBuilder3 模型。
        apo: 预测 apo 构象。

    Returns:
        ZIP bytes，内含各序列的 PDB 文件。
    """
    import csv
    from tempfile import TemporaryDirectory

    model_name = "abodybuilder3" if use_abodybuilder3 else "ibex"

    # 加载模型（一次性，共用于所有序列）；cache 挂到 ~/.cache/ibex
    from ibex import Ibex, inference  # type: ignore

    model = Ibex.from_pretrained(model_name)

    reader = csv.DictReader(csv_content.splitlines())
    rows = list(reader)
    print(f"批量预测 {len(rows)} 条序列  模型={model_name}  apo={apo}")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        with TemporaryDirectory() as tmp:
            for row in rows:
                seq_id = row.get("id", "pred").strip()
                fv_heavy = row.get("fv_heavy", "").strip()
                fv_light = row.get("fv_light", "").strip()

                if not fv_heavy:
                    print(f"  跳过 {seq_id}：fv_heavy 为空")
                    continue

                out_pdb = Path(tmp) / f"{seq_id}.pdb"
                try:
                    inference(model, fv_heavy, fv_light, str(out_pdb), apo=apo)
                    zf.write(out_pdb, f"{seq_id}.pdb")
                    print(f"  ✓ {seq_id}")
                except Exception as e:
                    print(f"  ✗ {seq_id}: {e}")

    return zip_buf.getvalue()


@app.function(
    timeout=7200,  # 2 小时，允许慢速下载
    volumes={WEIGHTS_MOUNT: ibex_vol},
)
def download_weights_fn():
    """下载 Ibex + ABodyBuilder3 权重到 Modal Volume（仅需运行一次）。

    使用 ibex 自带的 get_checkpoint_path / download_from_url，
    文件名格式：{url_basename}_{md5(url)[:8]}，与 from_pretrained 的查找路径一致。

    Ibex:  ibex_v1.ckpt  ~447 MB  (来自 Zenodo #15866556)
    ABB3:  abb3.ckpt      ~83 MB
    """
    # ibex 内部工具函数，处理 cache_dir + 带 hash 的文件名
    from ibex.utils import MODEL_CHECKPOINTS, download_from_url, get_checkpoint_path  # type: ignore

    cache_dir = WEIGHTS_MOUNT
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for model_name, url in MODEL_CHECKPOINTS.items():
        local_path = get_checkpoint_path(url, cache_dir=cache_dir)
        if Path(local_path).exists():
            size_mb = Path(local_path).stat().st_size / 1024 / 1024
            print(f"已存在: {Path(local_path).name}  ({size_mb:.1f} MB)  跳过")
            continue

        print(f"下载 {model_name} → {local_path} ...")
        download_from_url(url, local_path)
        ibex_vol.commit()
        size_mb = Path(local_path).stat().st_size / 1024 / 1024
        print(f"完成: {model_name}  ({size_mb:.1f} MB)  → {local_path}")

    # 列出 volume 内容
    print("\nVolume 内容：")
    for f in sorted(Path(cache_dir).glob("*")):
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    print("✓ 权重已保存至 Modal Volume 'ibex-weights'")


@app.function(
    timeout=600,
    volumes={WEIGHTS_MOUNT: ibex_vol},
)
def probe_cache():
    """探测 ibex 实际使用的权重缓存目录（smoke test 辅助工具）。

    输出：环境变量 IBEX_WEIGHTS_DIR、home/.cache 结构、prescient-ibex 版本。
    """
    import glob
    import subprocess

    print("=== ibex 环境探测 ===")
    print(f"IBEX_WEIGHTS_DIR = {os.environ.get('IBEX_WEIGHTS_DIR', '未设置')}")
    print(f"HOME = {os.environ.get('HOME', '?')}")
    print(f"Volume mount ({WEIGHTS_MOUNT}) 内容:")
    for f in sorted(Path(WEIGHTS_MOUNT).glob("**/*")):
        size_mb = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
        print(f"  {f}  ({size_mb:.1f} MB)" if f.is_file() else f"  {f}/")

    # 打印 ibex 包版本和源代码路径
    result = subprocess.run(["pip", "show", "prescient-ibex"], capture_output=True, text=True)
    print("\n=== prescient-ibex 包信息 ===")
    print(result.stdout)

    # 尝试导入并打印内部 cache 路径（如果有的话）
    try:
        import ibex  # type: ignore

        ibex_dir = Path(ibex.__file__).parent
        print(f"\nibex 模块路径: {ibex_dir}")
        # 搜索含 "cache" 或 "checkpoint" 字样的源文件
        for py_file in ibex_dir.glob("**/*.py"):
            content = py_file.read_text(errors="ignore")
            if "cache" in content.lower() and ("download" in content.lower() or "zenodo" in content.lower()):
                print(f"\n关键文件: {py_file}")
                # 打印含 cache/download 的行
                for line in content.splitlines():
                    if any(kw in line.lower() for kw in ("cache", "zenodo", "checkpoint", "download")):
                        print(f"  {line.rstrip()}")
    except ImportError as e:
        print(f"ibex 导入失败: {e}")

    print("\n=== ~/.cache 结构 ===")
    cache_root = Path.home() / ".cache"
    if cache_root.exists():
        for d in sorted(cache_root.iterdir()):
            print(f"  {d}")
    else:
        print("  ~/.cache 不存在")


# ── 本地入口 ───────────────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    heavy: Optional[str] = None,
    light: Optional[str] = None,
    input_faa: Optional[str] = None,
    input_csv: Optional[str] = None,
    name: str = "prediction",
    use_abodybuilder3: bool = False,
    apo: bool = False,
    refine: bool = False,
    out_dir: Optional[str] = None,
    seed_weights: bool = False,
    probe_cache_flag: bool = False,
):
    """Ibex / ABodyBuilder3 VHH 和抗体结构预测。

    Args:
        heavy: 重链 / VHH 序列（直接传字符串，单序列模式）。
        light: 轻链序列（可选，VHH 省略）。
        input_faa: FASTA 文件路径，支持多条序列批量预测。
        input_csv: CSV 文件路径（列：id, fv_heavy, fv_light）批量预测。
        name: 单序列预测时的输出名称前缀。
        use_abodybuilder3: 使用 ABodyBuilder3 模型（默认 Ibex）。
        apo: 预测 apo（未结合）构象（默认 holo）。
        refine: OpenMM 精细化（更准，更慢）。
        out_dir: 输出目录（默认 ./out/ibex/）。
        seed_weights: True → 下载权重到 Volume 后退出。
        probe_cache_flag: True → 探测 ibex 缓存路径后退出（smoke test 用）。
    """
    from datetime import datetime

    # 下载权重
    if seed_weights:
        print("开始下载权重到 Modal Volume 'ibex-weights'...")
        download_weights_fn.remote()
        return

    # 探测缓存路径
    if probe_cache_flag:
        probe_cache.remote()
        return

    model_label = "abb3" if use_abodybuilder3 else "ibex"
    out_base = Path(out_dir or "./out/ibex")
    run_tag = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_path = out_base / run_tag
    out_path.mkdir(parents=True, exist_ok=True)

    # 模式 1：单序列（--heavy）
    if heavy:
        pdb_bytes = predict_single.remote(
            fv_heavy=heavy,
            fv_light=light,
            name=name,
            use_abodybuilder3=use_abodybuilder3,
            apo=apo,
            refine=refine,
        )
        pdb_file = out_path / f"{name}_{model_label}.pdb"
        pdb_file.write_bytes(pdb_bytes)
        print(f"输出: {pdb_file}")
        return

    # 模式 2：FASTA 批量（--input-faa）
    if input_faa:
        fasta_content = Path(input_faa).read_text()
        pairs = _parse_fasta(fasta_content)
        print(f"解析 {len(pairs)} 条序列 from {input_faa}")

        for seq_name, fv_heavy, fv_light in pairs:
            pdb_bytes = predict_single.remote(
                fv_heavy=fv_heavy,
                fv_light=fv_light or None,
                name=seq_name,
                use_abodybuilder3=use_abodybuilder3,
                apo=apo,
                refine=refine,
            )
            pdb_file = out_path / f"{seq_name}_{model_label}.pdb"
            pdb_file.write_bytes(pdb_bytes)
            print(f"  ✓ {pdb_file}")
        return

    # 模式 3：CSV 批量（--input-csv）
    if input_csv:
        csv_content = Path(input_csv).read_text()
        zip_bytes = predict_batch_csv.remote(
            csv_content=csv_content,
            use_abodybuilder3=use_abodybuilder3,
            apo=apo,
        )
        zip_file = out_path / f"ibex_batch_{model_label}.zip"
        zip_file.write_bytes(zip_bytes)
        print(f"批量结果 ZIP: {zip_file}")
        return

    print(
        "请指定输入：\n"
        "  --heavy SEQ             单 VHH / 重链序列\n"
        "  --input-faa FILE        FASTA 文件\n"
        "  --input-csv FILE        CSV 文件\n"
        "  --seed-weights          下载权重到 Volume（首次使用）\n"
        "  --probe-cache-flag      探测权重缓存路径（smoke test）"
    )
