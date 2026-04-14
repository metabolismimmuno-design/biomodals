# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
RFdiffusion2：全原子蛋白质设计扩散模型，运行于 Modal GPU。

论文/代码: https://github.com/RosettaCommons/RFdiffusion2

支持两种设计任务：
1. 小分子结合蛋白设计 (small_molecule_binder)：针对配体（ATP、NAD 等）设计结合蛋白
2. 酶活性位点支架设计 (enzyme_scaffolding)：在酶活性位点周围设计蛋白支架

注意：这是独立的全原子扩散模型，与 RFdiffusion v1 功能不同（v1 专注蛋白-蛋白 binder）。

用法：
  # 小分子结合蛋白设计
  modal run modal_rfdiffusion2.py --task binder --pdb protein_with_atp.pdb --ligand ATP

  # 酶活性位点支架设计（使用自定义 contigs）
  modal run modal_rfdiffusion2.py --task scaffold --pdb enzyme.pdb --ligands NAD,OXM \\
      --contigs "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46"

  # 配体列表
  modal run modal_rfdiffusion2.py --task list-ligands
"""

import glob
import os
from pathlib import Path
from subprocess import run as sp_run

from modal import App, Image

RFDIFFUSION2_DIR = "/app/RFdiffusion2"
OUTPUT_ROOT = "rfdiffusion2_out"

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

app = App("rfdiffusion2")

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("git", "wget", "build-essential")
    # ── 1. Clone RFdiffusion2 ──────────────────────────────────────────────────
    .run_commands(
        f"git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion2.git {RFDIFFUSION2_DIR}"
    )
    # ── 2. PyTorch Geometric（CUDA 12.1 wheels）────────────────────────────────
    .pip_install(
        [
            "pyg-lib",
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
        ],
        find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    )
    # ── 3. DGL（CUDA 12.1 wheels）──────────────────────────────────────────────
    .pip_install(
        ["dgl==2.4.0"],
        find_links="https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html",
    )
    # ── 4. 科学计算 + 生物信息依赖 ───────────────────────────────────────────────
    .pip_install(
        [
            "hydra-core==1.3.1",
            "omegaconf==2.3.0",
            "ml-collections==0.1.1",
            "addict==2.4.0",
            "assertpy==1.1.0",
            "biopython==1.83",
            "colorlog",
            "compact-json",
            "cytoolz==0.12.3",
            "deepdiff==6.3.0",
            "dm-tree==0.1.8",
            "e3nn==0.5.1",
            "einops==0.7.0",
            "fire==0.6.0",
            "GPUtil==1.4.0",
            "icecream==2.1.3",
            "mdtraj==1.10.0",
            "numba",
            "opt_einsum==3.3.0",
            "rdkit==2024.3.5",
            "scipy==1.13.1",
            "seaborn==0.13.2",
            "submitit",
            "sympy==1.13.2",
            "tqdm==4.65.0",
            "typer==0.12.5",
            "biotite",
            "torchdata==0.9.0",
            "numpy",
            "pandas",
        ]
    )
    # ── 5. 安装 RFdiffusion2 Python 包 ────────────────────────────────────────
    .run_commands(f"cd {RFDIFFUSION2_DIR} && pip install -e . --no-deps")
    # ── 6. 下载模型权重（烘焙进镜像，Modal 会缓存）─────────────────────────────
    .run_commands(
        f"mkdir -p {RFDIFFUSION2_DIR}/rf_diffusion/model_weights;"
        f"wget -q https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_173.pt "
        f"  -O {RFDIFFUSION2_DIR}/rf_diffusion/model_weights/RFD_173.pt;"
        f"wget -q https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_140.pt "
        f"  -O {RFDIFFUSION2_DIR}/rf_diffusion/model_weights/RFD_140.pt;"
        f"mkdir -p {RFDIFFUSION2_DIR}/rf_diffusion/third_party_model_weights/ligand_mpnn;"
        f"wget -q https://files.ipd.uw.edu/pub/rfdiffusion2/third_party_model_weights/ligand_mpnn/s25_r010_t300_p.pt "
        f"  -O {RFDIFFUSION2_DIR}/rf_diffusion/third_party_model_weights/ligand_mpnn/s25_r010_t300_p.pt;"
        f"wget -q https://files.ipd.uw.edu/pub/rfdiffusion2/third_party_model_weights/ligand_mpnn/s_300756.pt "
        f"  -O {RFDIFFUSION2_DIR}/rf_diffusion/third_party_model_weights/ligand_mpnn/s_300756.pt"
    )
    .env({"PYTHONPATH": RFDIFFUSION2_DIR})
)


def _run_inference(cmd: list[str], cwd: str) -> dict:
    """运行 run_inference.py 子进程并捕获输出。

    Args:
        cmd: 命令及参数列表
        cwd: 工作目录

    Returns:
        包含 stdout/stderr/returncode 的字典
    """
    result = sp_run(cmd, cwd=cwd, capture_output=True, text=True, timeout=3600)
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(image=image, gpu=GPU, timeout=TIMEOUT * 60)
def small_molecule_binder(
    pdb_content: bytes,
    pdb_name: str,
    ligand: str,
    num_designs: int = 5,
    min_length: int = 50,
    max_length: int = 150,
    inference_steps: int = 50,
    noise_scale: float = 1.0,
    hotspot_res: str = "auto",
    binding_radius: float = 8.0,
) -> list:
    """在 Modal GPU 上运行 RFdiffusion2 小分子结合蛋白设计。

    Args:
        pdb_content: 输入 PDB 文件的字节内容（需含目标配体）
        pdb_name: PDB 文件名（用于输出命名）
        ligand: 配体三字母代码（如 ATP、NAD、PH2）
        num_designs: 生成设计数量，默认 5
        min_length: 最小结合蛋白长度（残基数），默认 50
        max_length: 最大结合蛋白长度（残基数），默认 150
        inference_steps: 扩散步数，默认 50
        noise_scale: 噪声强度，默认 1.0
        hotspot_res: 热点残基（'auto' 表示自动检测），默认 'auto'
        binding_radius: 结合位点半径（埃），默认 8.0

    Returns:
        list[tuple[str, bytes]]: (相对路径, 文件内容) 元组列表
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td_in, tempfile.TemporaryDirectory() as td_out:
        # 写入输入 PDB
        input_pdb = Path(td_in) / pdb_name
        input_pdb.write_bytes(pdb_content)

        output_prefix = str(Path(td_out) / "binder")

        cmd = [
            "python",
            f"{RFDIFFUSION2_DIR}/rf_diffusion/run_inference.py",
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={num_designs}",
            "diffuser.binder_design=True",
            f"inference.ligand={ligand}",
            f"inference.hotspot_res={hotspot_res}",
            f"inference.binding_radius={binding_radius}",
            f"contigmap.length=[{min_length}-{max_length}]",
            f"diffuser.T={inference_steps}",
            f"diffuser.noise_scale_ca={noise_scale}",
        ]

        result = _run_inference(cmd, cwd=RFDIFFUSION2_DIR)

        if not result["success"]:
            print(f"RFdiffusion2 错误 (code {result['returncode']}):")
            print(result["stderr"][-2000:])
            print(result["stdout"][-1000:])
            raise RuntimeError(
                f"RFdiffusion2 failed (code {result['returncode']})\n"
                f"STDERR: {result['stderr'][-500:]}"
            )

        return [
            (out_file, open(out_file, "rb").read())
            for out_file in glob.glob(f"{td_out}/**/*.*", recursive=True)
            if Path(out_file).is_file()
        ]


@app.function(image=image, gpu=GPU, timeout=TIMEOUT * 60)
def enzyme_scaffolding(
    pdb_content: bytes,
    pdb_name: str,
    ligands: str = "NAD,OXM",
    contigs: str = "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
    contig_atoms: str = "{'A106':'NE,CD,CZ','A166':'OD1,CG','A169':'NH2,CZ','A193':'NE2,CD2,CE1'}",
    num_designs: int = 5,
    inference_steps: int = 50,
    noise_scale: float = 1.0,
) -> list:
    """在 Modal GPU 上运行 RFdiffusion2 酶活性位点支架设计。

    Args:
        pdb_content: 输入酶结构 PDB 的字节内容
        pdb_name: PDB 文件名
        ligands: 配体代码（逗号分隔），如 "NAD,OXM"，默认 "NAD,OXM"
        contigs: Contig 规格，定义固定/可变区域，默认使用示例值
        contig_atoms: 活性位点原子规格（字典格式字符串），默认使用示例值
        num_designs: 生成设计数量，默认 5
        inference_steps: 扩散步数，默认 50
        noise_scale: 噪声强度，默认 1.0

    Returns:
        list[tuple[str, bytes]]: (相对路径, 文件内容) 元组列表
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td_in, tempfile.TemporaryDirectory() as td_out:
        # 写入输入 PDB
        input_pdb = Path(td_in) / pdb_name
        input_pdb.write_bytes(pdb_content)

        output_prefix = str(Path(td_out) / "scaffold")

        cmd = [
            "python",
            f"{RFDIFFUSION2_DIR}/rf_diffusion/run_inference.py",
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={num_designs}",
            "diffuser.scaffolding=True",
            f"contigmap.contigs=[{contigs}]",
            f"contigmap.inpaint_seq={contig_atoms}",
            f"inference.ligands={ligands}",
            f"diffuser.T={inference_steps}",
            f"diffuser.noise_scale_ca={noise_scale}",
        ]

        result = _run_inference(cmd, cwd=RFDIFFUSION2_DIR)

        if not result["success"]:
            print(f"RFdiffusion2 错误 (code {result['returncode']}):")
            print(result["stderr"][-2000:])
            print(result["stdout"][-1000:])
            raise RuntimeError(
                f"RFdiffusion2 failed (code {result['returncode']})\n"
                f"STDERR: {result['stderr'][-500:]}"
            )

        return [
            (out_file, open(out_file, "rb").read())
            for out_file in glob.glob(f"{td_out}/**/*.*", recursive=True)
            if Path(out_file).is_file()
        ]


# 常见配体代码参考表
COMMON_LIGANDS = {
    "PH2": "邻苯二甲酸 (Phthalic acid)",
    "NAD": "烟酰胺腺嘌呤二核苷酸 (NAD)",
    "ATP": "腺苷三磷酸 (ATP)",
    "GTP": "鸟苷三磷酸 (GTP)",
    "FAD": "黄素腺嘌呤二核苷酸 (FAD)",
    "FMN": "黄素单核苷酸 (FMN)",
    "HEM": "血红素 (Heme)",
    "ZN": "锌离子",
    "MG": "镁离子",
    "CA": "钙离子",
    "FE": "铁离子",
    "OXM": "草酰乙酸 (Oxaloacetate)",
    "COA": "辅酶A (CoA)",
    "SAM": "S-腺苷甲硫氨酸 (SAM)",
}


@app.local_entrypoint()
def main(
    task: str = "binder",
    pdb: str = "",
    ligand: str = "ATP",
    ligands: str = "NAD,OXM",
    contigs: str = "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
    contig_atoms: str = "{'A106':'NE,CD,CZ','A166':'OD1,CG','A169':'NH2,CZ','A193':'NE2,CD2,CE1'}",
    num_designs: int = 5,
    min_length: int = 50,
    max_length: int = 150,
    inference_steps: int = 50,
    noise_scale: float = 1.0,
    out_dir: str = "./out/rfdiffusion2",
    run_name: str = None,
):
    """本地入口：在 Modal GPU 上运行 RFdiffusion2 设计任务。

    Args:
        task: 任务类型，'binder'（小分子结合蛋白）或 'scaffold'（酶活性位点支架）
              或 'list-ligands'（列出常见配体代码）
        pdb: 输入 PDB 文件路径（binder/scaffold 任务必须提供）
        ligand: 目标配体三字母代码，用于 binder 任务（如 ATP、NAD、PH2）
        ligands: 配体代码（逗号分隔），用于 scaffold 任务（如 NAD,OXM）
        contigs: Contig 规格，用于 scaffold 任务
        contig_atoms: 活性位点原子规格，用于 scaffold 任务
        num_designs: 生成设计数量，默认 5
        min_length: 最小长度（binder 任务），默认 50
        max_length: 最大长度（binder 任务），默认 150
        inference_steps: 扩散步数，默认 50
        noise_scale: 噪声强度，默认 1.0
        out_dir: 输出目录，默认 ./out/rfdiffusion2
        run_name: 运行名称（用于输出子目录命名）
    """
    from datetime import datetime

    # 列出配体代码
    if task == "list-ligands":
        print("RFdiffusion2 常见配体代码：")
        for code, name in COMMON_LIGANDS.items():
            print(f"  {code}: {name}")
        return

    # 验证输入
    if not pdb:
        raise ValueError("--pdb 参数必须提供（输入 PDB 文件路径）")

    pdb_path = Path(pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB 文件不存在: {pdb}")

    pdb_content = pdb_path.read_bytes()
    pdb_name = pdb_path.name

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    print(f"任务: {task}")
    print(f"输入: {pdb}")
    print(f"设计数量: {num_designs}")
    print(f"输出目录: {out_dir_full}")

    if task == "binder":
        print(f"目标配体: {ligand}")
        print(f"长度范围: {min_length}-{max_length} 残基")
        outputs = small_molecule_binder.remote(
            pdb_content=pdb_content,
            pdb_name=pdb_name,
            ligand=ligand,
            num_designs=num_designs,
            min_length=min_length,
            max_length=max_length,
            inference_steps=inference_steps,
            noise_scale=noise_scale,
        )
    elif task == "scaffold":
        print(f"配体: {ligands}")
        print(f"Contigs: {contigs}")
        outputs = enzyme_scaffolding.remote(
            pdb_content=pdb_content,
            pdb_name=pdb_name,
            ligands=ligands,
            contigs=contigs,
            contig_atoms=contig_atoms,
            num_designs=num_designs,
            inference_steps=inference_steps,
            noise_scale=noise_scale,
        )
    else:
        raise ValueError(f"未知任务类型: {task}，支持 'binder'、'scaffold' 或 'list-ligands'")

    # 保存输出文件
    out_dir_full.mkdir(parents=True, exist_ok=True)
    for out_file, out_content in outputs:
        out_path = out_dir_full / Path(out_file).name
        out_path.write_bytes(out_content)
        print(f"  已保存: {out_path}")

    print(f"\n完成！共 {len(outputs)} 个输出文件 → {out_dir_full}")
