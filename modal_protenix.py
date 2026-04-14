# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Protenix v2 https://github.com/bytedance/Protenix

Protenix is an open-source PyTorch reproduction of AlphaFold 3.
v2 新增 Training-Free Guidance (TFG)，Ab-Ag 预测 DockQ 提升 +9-13 pp。

## 首次使用（下载模型权重到 Modal Volume）
模型权重 (~1GB) 存在 Modal Volume "protenix-v2-weights" 中，需手动初始化一次：

  # 方式 A：从本机下载后上传（推荐，适合中国网络环境）
  wget https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt
  modal volume put protenix-v2-weights protenix-v2.pt /checkpoint/protenix-v2.pt

  # 方式 B：在 Modal 容器内直接下载（需 CDN 可达）
  modal run modal_protenix.py --seed-weights

## Example FASTA input (test_protenix.faa):
```
>protein|A
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS
```

## Usage:
```
modal run modal_protenix.py --input-faa test_protenix.faa
```

With custom parameters:
```
modal run modal_protenix.py --input-faa test_protenix.faa --seeds "42,43" --use-msa
```

## Advanced: JSON input (native Protenix format)
```json
[
    {
        "name": "test_protein",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MAWTPLLLLLLSHCTGSLSQPVLTQPTSL",
                    "count": 1
                }
            }
        ]
    }
]
```

Usage with JSON:
```
modal run modal_protenix.py --input-json test_protenix.json
```
"""

import os
from pathlib import Path
from typing import Optional

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

# 持久化存储预测结果（CIF + JSON），供按需下载
protenix_results_vol = Volume.from_name("protenix-results", create_if_missing=True)
RESULTS_MOUNT = "/root/protenix_results"

ENTITY_TYPES = {"protein", "dna", "rna", "ligand", "ion"}

ENTITY_TYPE_MAP = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
    "ligand": "ligand",
    "ion": "ion",
}

DEFAULT_MODEL = "protenix-v2"
DEFAULT_SEEDS = "42"

# Modal Volume 持久化模型权重，避免每次从中国 CDN 重新下载
# 初始化：见文件头注释"首次使用"部分
protenix_weights = Volume.from_name("protenix-v2-weights", create_if_missing=True)

# CHECKPOINT_DIR = PROTENIX_ROOT_DIR/checkpoint/ （Protenix 默认）
CHECKPOINT_MOUNT = "/root/checkpoint"


image = (
    Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "wget", "g++", "gcc", "make", "clang", "libxrender1")
    .pip_install("protenix==2.0.0")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    # Step 1: 从 wwPDB 下载 components.cif（美国服务器可快速访问，秒级完成）
    .run_commands(
        "mkdir -p /root/common && "
        "wget -q --timeout=300 -O /root/common/components.cif "
        "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif"
    )
    # Step 2: 用 Protenix 自带脚本本地生成 components.cif.rdkit_mol.pkl
    # --disable_download: 跳过重新下载 cif（已在上一步完成）
    # --n_cpu 8: 并行处理 ~40,000 个 CCD 组分，加速生成
    # 注意：此步骤纯 CPU，不需要 GPU，约需 10-30 分钟
    .run_commands(
        "python3 /usr/local/lib/python3.12/site-packages/scripts/gen_ccd_cache.py "
        "--ccd_cache_dir /root/common --disable_download --n_cpu 8"
    )
)

app = App("protenix", image=image)


def _fasta_to_json(input_faa: str, name: str = "input") -> str:
    """Convert FASTA to Protenix JSON format.

    Expected FASTA format:
    >protein|A|description
    SEQUENCE
    >dna|B|description
    ATCG
    >ligand|D|CCD_NAME
    """
    import json
    import re

    rx = re.compile(r"^([^|]+)\|([^|]*)\|?(.*)$")
    sequences = []

    current_id = None
    current_seq: list[str] = []
    for line in input_faa.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                sequences.append((current_id, "".join(current_seq)))
            current_id = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        sequences.append((current_id, "".join(current_seq)))

    json_sequences = []
    for seq_id, seq in sequences:
        match = rx.search(seq_id)
        entity_type = match.group(1).lower() if match else "protein"

        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {ENTITY_TYPES}")

        protenix_type = ENTITY_TYPE_MAP[entity_type]

        if entity_type == "ligand":
            entity = {protenix_type: {"ligand": seq, "count": 1}}
        else:
            entity = {protenix_type: {"sequence": seq, "count": 1}}

        json_sequences.append(entity)

    return json.dumps([{"name": name, "sequences": json_sequences}], indent=2)


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CHECKPOINT_MOUNT: protenix_weights},
)
def protenix(
    input_str: str,
    input_name: str = "input",
    seeds: str = DEFAULT_SEEDS,
    use_msa: bool = True,
    model_name: str = DEFAULT_MODEL,
    use_tfg_guidance: bool = True,
) -> list:
    """Run Protenix on FASTA or JSON input and return output files.

    Args:
        input_str: FASTA or JSON content.
        input_name: Name for prediction job.
        seeds: Comma-separated seeds.
        use_msa: Use MSA.
        model_name: Model name.
        use_tfg_guidance: Enable v2 Training-Free Guidance (TFG) for +9-13 pp Ab-Ag DockQ.
            Default True when using protenix-v2. Set False for single-chain monomer prediction.

    Returns:
        list[tuple[Path, bytes]]: Output files as (relative_path, bytes) tuples.
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        if input_str.strip().startswith(">"):
            json_str = _fasta_to_json(input_str, name=input_name)
            print("Converted FASTA to JSON:")
            print(json_str)
        else:
            json_str = input_str

        json_path = Path(in_dir) / "input.json"
        json_path.write_text(json_str)

        print(f"Running Protenix with model={model_name}, seeds={seeds}, use_msa={use_msa}, use_tfg_guidance={use_tfg_guidance}")

        run(
            f'protenix pred --input "{json_path}" '
            f'--out_dir "{out_dir}" '
            f"--seeds {seeds} "
            f"--use_msa {str(use_msa).lower()} "
            f"--model_name {model_name} "
            f"--use_tfg_guidance {str(use_tfg_guidance).lower()} "
            f"--enable_cache true --enable_fusion true",
            shell=True,
            check=True,
        )

        return [
            (out_file.relative_to(out_dir), out_file.read_bytes())
            for out_file in Path(out_dir).glob("**/*")
            if out_file.is_file()
        ]


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CHECKPOINT_MOUNT: protenix_weights},
)
def run_prediction_zip(
    input_str: str,
    input_name: str = "input",
    seeds: str = DEFAULT_SEEDS,
    use_msa: bool = True,
    model_name: str = DEFAULT_MODEL,
    use_tfg_guidance: bool = True,
) -> bytes:
    """Run Protenix and return all output files as a zip archive (bytes).

    Designed for MCP async dispatch: can be spawned with .spawn() and
    retrieved with call.get(timeout=0). Returns a single zip blob so that
    MCP tools can handle results without list serialisation issues.

    Args:
        input_str: FASTA or JSON content (same as protenix()).
        input_name: Name prefix for the prediction job.
        seeds: Comma-separated seed values (e.g. "42" or "42,43").
        use_msa: Whether to run MSA lookup (slower but more accurate).
        model_name: Protenix model checkpoint name.
        use_tfg_guidance: Enable v2 Training-Free Guidance (TFG) for +9-13 pp Ab-Ag DockQ.
            Default True when using protenix-v2. Set False for single-chain monomer prediction.

    Returns:
        bytes: ZIP archive containing all Protenix output files
            (*.cif structures, *_summary_confidence.json scores, etc.).
    """
    import io
    import zipfile
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        if input_str.strip().startswith(">"):
            json_str = _fasta_to_json(input_str, name=input_name)
            print("Converted FASTA to JSON:")
            print(json_str)
        else:
            json_str = input_str

        json_path = Path(in_dir) / "input.json"
        json_path.write_text(json_str)

        print(f"Running Protenix with model={model_name}, seeds={seeds}, use_msa={use_msa}, use_tfg_guidance={use_tfg_guidance}")

        run(
            f'protenix pred --input "{json_path}" '
            f'--out_dir "{out_dir}" '
            f"--seeds {seeds} "
            f"--use_msa {str(use_msa).lower()} "
            f"--model_name {model_name} "
            f"--use_tfg_guidance {str(use_tfg_guidance).lower()} "
            f"--enable_cache true --enable_fusion true",
            shell=True,
            check=True,
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for out_file in sorted(Path(out_dir).glob("**/*")):
                if out_file.is_file():
                    zf.write(out_file, out_file.relative_to(out_dir))

        return buf.getvalue()


@app.function(
    timeout=14400,  # 4 小时，允许从中国 CDN 慢速下载模型权重
    volumes={CHECKPOINT_MOUNT: protenix_weights},
)
def seed_weights(model_name: str = DEFAULT_MODEL):
    """下载模型权重到 Modal Volume（仅需运行一次）。

    如果本机网络到中国 CDN 更快，建议用命令行方式：
        wget https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt
        modal volume put protenix-v2-weights protenix-v2.pt /checkpoint/protenix-v2.pt

    Args:
        model_name: 要下载的模型名称，默认 protenix-v2。
    """
    import urllib.request

    checkpoint_path = Path(CHECKPOINT_MOUNT) / f"{model_name}.pt"
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / 1024 / 1024
        print(f"权重已存在：{checkpoint_path}（{size_mb:.1f} MB）")
        return

    from protenix.web_service.dependency_url import URL

    url = URL.get(model_name)
    if not url:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(URL.keys())}")

    print(f"下载 {model_name} 权重 from {url} ...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, checkpoint_path)
    protenix_weights.commit()
    size_mb = checkpoint_path.stat().st_size / 1024 / 1024
    print(f"完成！权重已保存至 Volume：{checkpoint_path}（{size_mb:.1f} MB）")


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CHECKPOINT_MOUNT: protenix_weights, RESULTS_MOUNT: protenix_results_vol},
)
def run_prediction_fast(
    input_str: str,
    input_name: str = "input",
    seeds: str = DEFAULT_SEEDS,
    use_msa: bool = True,
    model_name: str = DEFAULT_MODEL,
    use_tfg_guidance: bool = True,
) -> bytes:
    """运行 Protenix，将所有输出存入 Volume，只返回 confidence JSON zip（几 KB）。

    与 run_prediction_zip 相比：
    - 返回值从几十 MB 降至 <100 KB（只含 JSON 分数，无 CIF）
    - 所有 CIF 文件保存在 protenix-results Volume，可通过 get_cif_zip() 按需获取

    Args:
        input_str: FASTA 或 JSON 内容。
        input_name: 任务名称，同时作为 Volume 内的存储路径前缀。
        seeds: 逗号分隔的随机种子。
        use_msa: 是否使用 MSA。
        model_name: 模型名称。
        use_tfg_guidance: 启用 TFG 引导（Ab-Ag 预测推荐开启）。

    Returns:
        bytes: 只含 *_summary_confidence.json 文件的 ZIP。
    """
    import io
    import shutil
    import zipfile
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        if input_str.strip().startswith(">"):
            json_str = _fasta_to_json(input_str, name=input_name)
        else:
            json_str = input_str

        import json as _json
        json_path = Path(in_dir) / "input.json"
        json_path.write_text(json_str)

        print(f"Running Protenix fast: model={model_name}, seeds={seeds}, use_msa={use_msa}")

        run(
            f'protenix pred --input "{json_path}" '
            f'--out_dir "{out_dir}" '
            f"--seeds {seeds} "
            f"--use_msa {str(use_msa).lower()} "
            f"--model_name {model_name} "
            f"--use_tfg_guidance {str(use_tfg_guidance).lower()} "
            f"--enable_cache true --enable_fusion true",
            shell=True,
            check=True,
        )

        # 将所有文件存入 Volume（CIF + JSON 都保留）
        vol_path = Path(RESULTS_MOUNT) / input_name
        if vol_path.exists():
            shutil.rmtree(vol_path)
        shutil.copytree(Path(out_dir), vol_path)
        protenix_results_vol.commit()
        print(f"Saved all outputs to Volume: protenix-results/{input_name}")

        # 只打包 confidence JSON 返回（几 KB）
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for jf in sorted(Path(out_dir).glob("**/*summary_confidence*.json")):
                zf.write(jf, jf.relative_to(out_dir))
        payload = buf.getvalue()
        print(f"Returning scores-only ZIP: {len(payload)/1024:.1f} KB")
        return payload


@app.function(
    timeout=300,
    volumes={RESULTS_MOUNT: protenix_results_vol},
)
def get_cif_zip(input_name: str) -> bytes:
    """从 Volume 按需获取指定任务的 CIF 结构文件（无需重新预测）。

    Args:
        input_name: 任务名称（与 run_prediction_fast 提交时相同）。

    Returns:
        bytes: 含所有 *.cif 文件的 ZIP，若无结果则返回空 ZIP。
    """
    import io
    import zipfile

    vol_path = Path(RESULTS_MOUNT) / input_name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        cif_files = sorted(vol_path.glob("**/*.cif")) if vol_path.exists() else []
        for cif in cif_files:
            zf.write(cif, cif.relative_to(vol_path))
        print(f"Packed {len(cif_files)} CIF files for {input_name}")
    return buf.getvalue()


@app.local_entrypoint()
def main(
    input_faa: Optional[str] = None,
    input_json: Optional[str] = None,
    seeds: str = DEFAULT_SEEDS,
    use_msa: bool = True,
    model_name: str = DEFAULT_MODEL,
    use_mini: bool = False,
    use_tfg_guidance: bool = True,
    out_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    seed_weights_flag: bool = False,
):
    """Run Protenix predictions.

    Args:
        input_faa: Path to FASTA file.
        input_json: Path to JSON file (native Protenix format).
        seeds: Comma-separated seed values.
        use_msa: Whether to use MSA.
        model_name: Model name.
        use_mini: Use mini model variant.
        use_tfg_guidance: Enable v2 Training-Free Guidance for Ab-Ag (+9-13 pp DockQ).
        out_dir: Output directory.
        run_name: Run name subdirectory (timestamp if None).
        seed_weights_flag: If True, download model weights into Volume instead of running inference.
    """
    from datetime import datetime

    if seed_weights_flag:
        seed_weights.remote(model_name=model_name)
        return

    assert input_faa or input_json, "Either input_faa or input_json required"

    if use_mini:
        model_name = "protenix_mini_default_v0.5.0"
        out_dir = out_dir or "./out/protenix_mini"
    else:
        out_dir = out_dir or "./out/protenix"

    input_path = input_faa or input_json
    input_name = Path(input_path).stem

    outputs = protenix.remote(
        input_str=open(input_path).read(),
        input_name=input_name,
        seeds=seeds,
        use_msa=use_msa,
        model_name=model_name,
        use_tfg_guidance=use_tfg_guidance,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        out_path = Path(out_dir_full) / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(out_content or b"")

    print(f"Output saved to: {out_dir_full}")
