"""Download the three Alibaba PAI GPU trace 2020 tables we use, verify SHA-256, extract, and attach column names.

Usage:  python scripts/fetch_data.py            (writes data/raw/*.tar.gz, data/raw/*.csv, data/raw/*.header)

Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020 (Weng et al., NSDI '22).
The data is for research/study use; it is NOT redistributed in this repo.
Checksums are copied from the official README. If the primary OSS host fails we fall back to the GitHub mirror
https://github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data (same files, same checksums).
"""
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TABLES = ["pai_job_table", "pai_task_table", "pai_group_tag_table"]

PRIMARY = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/{name}"
HEADER_URLS = [
    "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/{name}",
    "https://raw.githubusercontent.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data/master/{name}",
]
MIRROR = "https://github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data/raw/master/{name}"

SHA256 = {  # from cluster-trace-gpu-v2020/README.md
    "pai_job_table.tar.gz": "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
    "pai_task_table.tar.gz": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
    "pai_group_tag_table.tar.gz": "722fef30b7fb7aa50dabd79155614b5423a9d65cf45a9b26c590d57725423a14",
    "pai_job_table.csv": "379ecb3becaba347f44a53bf7eb53e54b185221b2a0338a3f828828d269ba96c",
    "pai_task_table.csv": "6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de",
    "pai_group_tag_table.csv": "d23fcb56d88b2976fda4a62708d77b2a8c98635c4b443f100ed5553d4cd07fbe",
    "pai_job_table.header": "3ac33aefab9a4d81338794fa145fe280594a379444961a9c639f00181c508567",
    "pai_task_table.header": "978bbaabfc8695874c605c01c144b2977f611ceca73aeb72189988cdfbfb0a9c",
    "pai_group_tag_table.header": "320470fa8f77182bc6052c95ff359b75542c748e9516b3d8148a6e2bafc1798d",
}
# Column names as documented in the README, used only if no .header file can be downloaded.
FALLBACK_HEADER = {
    "pai_job_table": "job_name,inst_id,user,status,start_time,end_time",
    "pai_task_table": "job_name,task_name,inst_num,status,start_time,end_time,plan_cpu,plan_mem,plan_gpu,gpu_type",
    "pai_group_tag_table": "inst_id,user,gpu_type_spec,group,workload",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(urls, dest: Path) -> bool:
    for url in urls:
        try:
            print(f"  GET {url}")
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:  # network errors: try the next source
            print(f"    failed: {e}")
    return False


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for t in TABLES:
        tgz, csv, hdr = RAW / f"{t}.tar.gz", RAW / f"{t}.csv", RAW / f"{t}.header"
        print(t)
        if not (tgz.exists() and sha256(tgz) == SHA256[tgz.name]):
            if not download([PRIMARY.format(name=tgz.name), MIRROR.format(name=tgz.name)], tgz):
                print("  could not download archive"); ok = False; continue
        got = sha256(tgz)
        print(f"  archive sha256 {'OK' if got == SHA256[tgz.name] else 'MISMATCH ' + got}")
        ok &= got == SHA256[tgz.name]
        if not csv.exists():
            with tarfile.open(tgz) as tf:
                member = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
                member.name = csv.name  # flatten any folder inside the archive
                tf.extract(member, RAW, filter="data")
        got = sha256(csv)
        print(f"  csv     sha256 {'OK' if got == SHA256[csv.name] else 'MISMATCH ' + got}")
        ok &= got == SHA256[csv.name]
        if not hdr.exists():
            urls = [u.format(name=hdr.name) for u in HEADER_URLS] + [PRIMARY.format(name=hdr.name)]
            if not download(urls, hdr) or sha256(hdr) != SHA256[hdr.name]:
                print("  header not verified; writing README column list instead")
                hdr.write_text(FALLBACK_HEADER[t] + "\n")
        print(f"  columns: {hdr.read_text().strip()}")
    print("ALL CHECKSUMS OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
