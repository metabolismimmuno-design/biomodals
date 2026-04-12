#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
# ]
# ///
"""NetMHCpan-4.2 MHC-I binding prediction via DTU web service.

Calls the DTU Health Tech online server directly — no local binary needed.
Free for academic/non-commercial use.

Usage:
    # Peptide mode: predict binding for specific peptides
    python modal_netmhcpan.py --mode peptide --peptides "SLYNTVATL,GILGFVFTL" --allele "HLA-A02:01"

    # Protein mode: scan FASTA for epitopes
    python modal_netmhcpan.py --mode protein --input-fasta protein.fasta --allele "HLA-A02:01" --length 9

    # Multi-allele screening
    python modal_netmhcpan.py --mode peptide --peptides "SLYNTVATL" --allele "HLA-A02:01,HLA-B07:02"

    # Output to specific directory
    python modal_netmhcpan.py --mode peptide --peptides "SLYNTVATL" --allele "HLA-A02:01" --out-dir results/

Note: DTU server has rate limits. For batch screening (>1000 peptides), split into chunks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

DTU_WEBFACE = "https://services.healthtech.dtu.dk/cgi-bin/webface2.fcgi"
DTU_POLL = "https://services.healthtech.dtu.dk/cgi-bin/webface2.cgi"


def submit_and_poll(
    peptides: Optional[str] = None,
    fasta_str: Optional[str] = None,
    allele: str = "HLA-A02:01",
    length: str = "9",
    rank_threshold_strong: float = 0.5,
    rank_threshold_weak: float = 2.0,
    ba_pred: bool = False,
    max_wait: int = 120,
) -> str:
    """Submit prediction to DTU NetMHCpan-4.2 and poll for results.

    DTU uses async job submission: POST -> 302 redirect with jobid -> poll until done.
    For FASTA input, must use multipart form with file upload (SEQSUB).

    Args:
        peptides: Newline-separated peptide sequences (mode=peptide)
        fasta_str: FASTA format string (mode=protein)
        allele: HLA allele(s), comma-separated
        length: Peptide length(s) for scanning
        rank_threshold_strong: %Rank threshold for strong binders
        rank_threshold_weak: %Rank threshold for weak binders
        ba_pred: Include binding affinity prediction (slower)
        max_wait: Maximum seconds to wait for results

    Returns:
        Raw HTML response containing predictions
    """
    import tempfile

    # 构建 multipart form data
    files = {}
    data = {
        "configfile": "/var/www/html/services/NetMHCpan-4.2/webface.cf",
        "allele": allele,
        "threshold": "-99.9",
        "thrs": str(rank_threshold_strong),
        "thrw": str(rank_threshold_weak),
    }

    if ba_pred:
        data["BApred"] = "1"

    if fasta_str:
        # FASTA 模式：inp=0, 通过 SEQSUB 文件上传
        data["inp"] = "0"
        data["SEQPASTE"] = ""
        data["slave0"] = length
        # 写临时文件
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
        tmp.write(fasta_str)
        tmp.close()
        files["SEQSUB"] = ("input.fasta", open(tmp.name, "rb"), "text/plain")
    elif peptides:
        # Peptide 模式：inp=1, 通过 PEPSUB 文件上传
        data["inp"] = "1"
        data["SEQPASTE"] = ""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pep", delete=False)
        tmp.write(peptides)
        tmp.close()
        files["PEPSUB"] = ("input.pep", open(tmp.name, "rb"), "text/plain")

    # 步骤 1：提交，拿 redirect URL 中的 jobid
    resp = requests.post(DTU_WEBFACE, data=data, files=files, allow_redirects=False, timeout=60)

    # 关闭文件句柄
    for f in files.values():
        f[1].close()

    if resp.status_code not in (301, 302):
        return resp.text  # 可能直接返回结果

    redirect_url = resp.headers.get("Location", "")
    if not redirect_url:
        return resp.text

    # 补全 URL
    if redirect_url.startswith("/"):
        redirect_url = f"https://services.healthtech.dtu.dk{redirect_url}"

    # 从 URL 提取 jobid
    jobid_match = re.search(r"jobid=([A-F0-9]+)", redirect_url)
    if not jobid_match:
        return f"Error: Could not extract jobid from redirect: {redirect_url}"

    jobid = jobid_match.group(1)
    print(f"  Job submitted: {jobid}", file=sys.stderr)

    # 步骤 2：轮询结果
    poll_url = f"{DTU_POLL}?jobid={jobid}&wait=20"
    elapsed = 0
    poll_interval = 5

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        resp = requests.get(poll_url, timeout=60)
        html = resp.text

        # 检查是否有结果（NetMHCpan 输出包含 "---" 分隔符和数据行）
        if "-----" in html and ("Score_EL" in html or "Aff(nM)" in html):
            return html

        # 检查是否有错误
        if "Error" in html and "<pre>" in html:
            return html

        # 检查是否仍在排队/运行
        if "queued" in html.lower() or "running" in html.lower():
            print(f"  Job {jobid}: still running ({elapsed}s)...", file=sys.stderr)
            continue

        # 如果有 <pre> 内容但没有标准标记，也返回
        if "<pre>" in html:
            return html

    return f"Error: Job {jobid} timed out after {max_wait}s"


def parse_html_results(html: str) -> list[dict]:
    """Parse NetMHCpan results from DTU HTML response.

    The results are embedded as plain text within <pre> tags or directly in the HTML.
    Format: Pos MHC Peptide Core Of Gp Gl Ip Il Icore Identity Score_EL %Rank_EL [Score_BA %Rank_BA Aff(nM)] [BindLevel]
    """
    predictions = []
    in_data = False

    for line in html.split("\n"):
        stripped = line.strip()

        # 去除 HTML 标签
        clean = re.sub(r"<[^>]+>", "", stripped).strip()
        if not clean:
            continue

        # 检测数据段分隔符
        if clean.startswith("---"):
            in_data = True
            continue

        # 数据段结束（遇到新的注释或 HTML 块）
        if in_data and (clean.startswith("#") and "Rank Threshold" not in clean):
            # 可能是新数据段的注释头，继续
            pass

        if in_data and clean:
            parts = clean.split()
            if len(parts) >= 13:
                try:
                    pos = int(parts[0])
                except (ValueError, IndexError):
                    in_data = False  # 数据段结束
                    continue

                try:
                    pred = {
                        "pos": pos,
                        "allele": parts[1],
                        "peptide": parts[2],
                        "core": parts[3],
                        "identity": parts[10] if len(parts) > 10 else "",
                        "score_el": float(parts[11]) if len(parts) > 11 else 0.0,
                        "rank_el": float(parts[12]) if len(parts) > 12 else 99.0,
                    }

                    # BA 模式额外列
                    if len(parts) >= 16:
                        try:
                            pred["score_ba"] = float(parts[13])
                            pred["rank_ba"] = float(parts[14])
                            pred["affinity_nm"] = float(parts[15])
                        except ValueError:
                            pass

                    # Bind level
                    if parts[-1] in ("SB", "WB"):
                        pred["bind_level"] = parts[-1]
                    elif len(parts) > 13 and parts[-2] in ("<=",):
                        if parts[-1] in ("SB", "WB"):
                            pred["bind_level"] = parts[-1]

                    predictions.append(pred)
                except (ValueError, IndexError):
                    continue

    return predictions


def predict(
    peptides: Optional[str] = None,
    fasta_str: Optional[str] = None,
    allele: str = "HLA-A02:01",
    length: str = "9",
    rank_threshold: float = 2.0,
    ba_pred: bool = False,
) -> dict:
    """Run NetMHCpan prediction and return structured results.

    Args:
        peptides: Comma or newline-separated peptide sequences
        fasta_str: FASTA format protein sequence(s)
        allele: HLA allele(s), comma-separated
        length: Peptide length(s) for protein scanning
        rank_threshold: Filter predictions below this %Rank
        ba_pred: Include binding affinity prediction

    Returns:
        Dict with summary, predictions, and metadata
    """
    # 格式化 peptides
    pep_input = None
    if peptides:
        pep_input = "\n".join(p.strip() for p in peptides.replace(",", "\n").split("\n") if p.strip())

    try:
        html = submit_and_poll(
            peptides=pep_input,
            fasta_str=fasta_str,
            allele=allele,
            length=length,
            ba_pred=ba_pred,
        )
    except requests.RequestException as e:
        return {"error": f"DTU server request failed: {e}"}

    # 检查是否返回了错误
    if "Configuration error" in html or "Exception" in html:
        error_match = re.search(r"Message:</td><td>(.*?)</td>", html)
        msg = error_match.group(1) if error_match else "Unknown server error"
        return {"error": f"DTU server error: {msg}"}

    predictions = parse_html_results(html)

    if not predictions:
        # 可能是异步 job，检查是否有 job ID
        job_match = re.search(r"tmp/netMHCpan_(\w+)", html)
        if job_match:
            return {
                "error": "Job submitted but results not immediately available",
                "job_id": job_match.group(1),
                "hint": "Large jobs may need polling. Try again in a few seconds.",
            }
        return {"error": "No predictions found in response. Check allele format (e.g., HLA-A02:01)."}

    # 统计
    strong = [p for p in predictions if p["rank_el"] <= 0.5]
    weak = [p for p in predictions if 0.5 < p["rank_el"] <= 2.0]
    filtered = [p for p in predictions if p["rank_el"] <= rank_threshold]
    filtered.sort(key=lambda x: x["rank_el"])

    # 按 allele 分组统计
    allele_summary = {}
    for p in predictions:
        al = p["allele"]
        if al not in allele_summary:
            allele_summary[al] = {"total": 0, "strong_binders": 0, "weak_binders": 0}
        allele_summary[al]["total"] += 1
        if p["rank_el"] <= 0.5:
            allele_summary[al]["strong_binders"] += 1
        elif p["rank_el"] <= 2.0:
            allele_summary[al]["weak_binders"] += 1

    return {
        "summary": allele_summary,
        "predictions": filtered,
        "total_predictions": len(predictions),
        "strong_binders": len(strong),
        "weak_binders": len(weak),
        "filtered_count": len(filtered),
        "rank_threshold": rank_threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NetMHCpan-4.2 MHC-I binding prediction (DTU web service)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["peptide", "protein"], default="peptide",
                        help="Prediction mode")
    parser.add_argument("--peptides", default="",
                        help="Comma-separated peptide sequences (mode=peptide)")
    parser.add_argument("--input-fasta", default="",
                        help="Path to FASTA file (mode=protein)")
    parser.add_argument("--allele", default="HLA-A02:01",
                        help="HLA allele(s), comma-separated")
    parser.add_argument("--length", default="9",
                        help="Peptide length(s) for scanning (mode=protein)")
    parser.add_argument("--rank-threshold", type=float, default=2.0,
                        help="%%Rank cutoff for reporting (default: 2.0)")
    parser.add_argument("--ba-pred", action="store_true",
                        help="Include binding affinity prediction")
    parser.add_argument("--out-dir", default="",
                        help="Output directory (default: current dir)")
    parser.add_argument("--json-only", action="store_true",
                        help="Only output JSON, no table")

    args = parser.parse_args()

    # 输入验证
    if args.mode == "peptide" and not args.peptides:
        parser.error("--peptides required for peptide mode")
    if args.mode == "protein" and not args.input_fasta:
        parser.error("--input-fasta required for protein mode")

    # 读取输入
    fasta_str = None
    if args.mode == "protein":
        fasta_path = Path(args.input_fasta)
        if not fasta_path.exists():
            print(f"Error: File not found: {args.input_fasta}", file=sys.stderr)
            sys.exit(1)
        fasta_str = fasta_path.read_text()

    print(f"NetMHCpan-4.2 — mode={args.mode}, allele={args.allele}", file=sys.stderr)

    result = predict(
        peptides=args.peptides if args.mode == "peptide" else None,
        fasta_str=fasta_str,
        allele=args.allele,
        length=args.length,
        rank_threshold=args.rank_threshold,
        ba_pred=args.ba_pred,
    )

    # 错误处理
    if "error" in result:
        print(f"\nError: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # 输出表格
    if not args.json_only:
        print(f"\n{'='*70}")
        print(f"NetMHCpan-4.2 Results")
        print(f"{'='*70}")

        for al, s in result["summary"].items():
            print(f"\n  {al}:")
            print(f"    Total peptides:  {s['total']}")
            print(f"    Strong binders:  {s['strong_binders']}  (<=0.5% rank)")
            print(f"    Weak binders:    {s['weak_binders']}  (<=2.0% rank)")

        preds = result["predictions"]
        print(f"\n  Filtered (rank <= {args.rank_threshold}): {len(preds)}")

        if preds:
            print(f"\n  {'Pos':<5} {'Allele':<15} {'Peptide':<15} {'Rank%':<8} {'Score':<8} {'Level':<4}")
            print(f"  {'-'*60}")
            for p in preds[:30]:
                level = p.get("bind_level", "")
                print(f"  {p['pos']:<5} {p['allele']:<15} {p['peptide']:<15} {p['rank_el']:<8.3f} {p['score_el']:<8.5f} {level}")
            if len(preds) > 30:
                print(f"  ... and {len(preds)-30} more (see JSON output)")

    # 保存 JSON
    out_path = Path(args.out_dir) if args.out_dir else Path(".")
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "netmhcpan_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
