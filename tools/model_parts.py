"""Split the local GGUF for Git LFS or restore it after cloning.

Uses streaming I/O and verifies the original size and SHA-256. No model is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]
MODEL = ROOT / "models/qwen2.5-3b-instruct-q4_k_m.gguf"
EXPECTED_SIZE = 2_104_932_768
EXPECTED_SHA256 = "626b4a6678b86442240e33df819e00132d3ba7dddfE1cdc4fbb18e0a9615c62d".lower()
PART_BYTES = 1_000_000_000  # safely below the GitHub Free per-file limit
BLOCK = 8 * 1024 * 1024


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def check_original(model: Path) -> None:
    if not model.is_file():
        raise FileNotFoundError(f"缺少模型文件：{model}")
    if model.stat().st_size != EXPECTED_SIZE or sha_file(model) != EXPECTED_SHA256:
        raise ValueError("模型大小或SHA-256与MODEL_PROVENANCE.md不一致，停止操作")


def manifest_path(model: Path) -> Path:
    return model.parent / "model_parts.json"


def part_path(model: Path, index: int) -> Path:
    return model.with_name(f"{model.name}.part{index:02d}")


def verify_parts(model: Path, manifest: dict) -> None:
    if (manifest.get("schema_version") != "hospital-gguf-parts-v1" or
            manifest.get("model") != model.name or
            manifest.get("model_size") != EXPECTED_SIZE or
            manifest.get("model_sha256") != EXPECTED_SHA256 or
            not isinstance(manifest.get("parts"), list) or
            len(manifest["parts"]) < 2):
        raise ValueError("分片清单与预期模型不一致")
    total = 0
    for index, item in enumerate(manifest["parts"], 1):
        expected = part_path(model, index)
        if item.get("file") != expected.name or not expected.is_file():
            raise FileNotFoundError(f"缺少模型分片：{expected.name}；如通过Git克隆，请先运行 git lfs pull")
        if (expected.stat().st_size != item.get("size") or
                sha_file(expected) != item.get("sha256")):
            raise ValueError(f"模型分片大小或SHA-256不符：{expected.name}；请检查Git LFS下载")
        total += item["size"]
    if total != EXPECTED_SIZE:
        raise ValueError("模型分片总大小不符")


def split(model: Path) -> None:
    check_original(model)
    manifest_file = manifest_path(model)
    if manifest_file.exists():
        verify_parts(model, json.loads(manifest_file.read_text(encoding="utf-8")))
        print("模型分片已存在且校验通过")
        return
    parts = []
    with model.open("rb") as source:
        for index in range(1, (EXPECTED_SIZE + PART_BYTES - 1) // PART_BYTES + 1):
            destination = part_path(model, index)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            remaining, size = PART_BYTES, 0
            digest = hashlib.sha256()
            with temporary.open("wb") as output:
                while remaining:
                    block = source.read(min(BLOCK, remaining))
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                    remaining -= len(block)
            temporary.replace(destination)
            parts.append({"file": destination.name, "size": size, "sha256": digest.hexdigest()})
            print(f"分片 {index}: {size} bytes", flush=True)
    manifest = {"schema_version": "hospital-gguf-parts-v1", "model": model.name,
                "model_size": EXPECTED_SIZE, "model_sha256": EXPECTED_SHA256,
                "part_bytes": PART_BYTES, "parts": parts}
    verify_parts(model, manifest)
    temporary_manifest = manifest_file.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_file)
    print(f"分片清单：{manifest_file}")


def restore(model: Path) -> None:
    if model.is_file():
        check_original(model)
        print("完整模型已存在且SHA-256正确")
        return
    manifest_file = manifest_path(model)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"缺少分片清单：{manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    verify_parts(model, manifest)
    temporary = model.with_name(model.name + ".restore-tmp")
    digest, size = hashlib.sha256(), 0
    with temporary.open("wb") as output:
        for index in range(1, len(manifest["parts"]) + 1):
            with part_path(model, index).open("rb") as source:
                for block in iter(lambda: source.read(BLOCK), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
    if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("合并后的模型大小或SHA-256不符；未替换目标文件")
    temporary.replace(model)
    print(f"模型还原并校验通过：{model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("split", "restore"))
    args = parser.parse_args()
    if args.action == "split":
        split(MODEL)
    else:
        restore(MODEL)


if __name__ == "__main__":
    main()
