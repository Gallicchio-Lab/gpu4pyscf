#!/usr/bin/env python3
"""
Automated HIP conversion script for GPU4PySCF.
Uses hipify-perl with custom post-processing regex to handle quantum chemistry
recurrence patterns, warp-level reductions, and headers.
"""

import os
import sys
import subprocess
from pathlib import Path

HIPIFY_PERL = Path("/opt/rocm-7.2.0/bin/hipify-perl")
if not HIPIFY_PERL.exists():
    HIPIFY_PERL = Path("/opt/rocm/bin/hipify-perl")

LIB_DIR = Path(__file__).resolve().parents[1] / "gpu4pyscf" / "lib"

def hipify_file(file_path: Path):
    cmd = [str(HIPIFY_PERL), "-inplace", str(file_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error hipifying {file_path}: {result.stderr}", file=sys.stderr)
        return False

    # Post-process file content for gpu4pyscf specifics
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 1. Ensure hip_compat.h is included if cuda_runtime or hip_runtime is included
    if "hip_compat.h" not in content:
        if "#include <hip/hip_runtime.h>" in content:
            content = content.replace("#include <hip/hip_runtime.h>", '#include "hip_compat.h"')
        elif "#include <cuda_runtime.h>" in content:
            content = content.replace("#include <cuda_runtime.h>", '#include "hip_compat.h"')

    # 2. Map any remaining unmapped cudaHostGetDevicePointer
    content = content.replace("cudaHostGetDevicePointer", "hipHostGetDevicePointer")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return True

def hipify_directory(target_dir: Path):
    count = 0
    for ext in ("*.cu", "*.cuh"):
        for f in target_dir.rglob(ext):
            if hipify_file(f):
                count += 1
    print(f"Hipified {count} files in {target_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.is_dir():
                hipify_directory(p)
            elif p.is_file():
                hipify_file(p)
    else:
        print("Usage: hipify_gpu4pyscf.py <directory or file ...>")
