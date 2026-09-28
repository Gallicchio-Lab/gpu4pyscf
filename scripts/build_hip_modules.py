#!/usr/bin/env python3
import os
import sys
import subprocess
import concurrent.futures
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HIPCC = '/software/emilio/miniforge3/envs/gpu4pyscf-hip/bin/hipcc'
ARCHS = ['gfx1030', 'gfx1031', 'gfx1032']
ARCH_FLAGS = [f'--offload-arch={a}' for a in ARCHS]
COMPAT_HEADER = os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'hip', 'hip_compat.h')

INCLUDE_DIRS = [
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'hip'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'gint'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'gdft'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'gvhf-rys'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'pbc'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'gvhf-md'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'sem'),
    os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', 'multigrid'),
]

INC_FLAGS = [f'-I{d}' for d in INCLUDE_DIRS]

def parse_sources_from_cmake(cmake_path):
    with open(cmake_path, 'r') as f:
        lines = f.readlines()
    
    in_add_library = False
    sources = []
    lib_name = None
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        
        if 'add_library' in stripped:
            in_add_library = True
            parts = stripped.replace('(', ' ').replace(')', ' ').split()
            idx = parts.index('add_library')
            lib_name = parts[idx + 1]
            for p in parts[idx + 3:]:
                if p.endswith('.c') or p.endswith('.cu'):
                    sources.append(p)
            if ')' in line:
                in_add_library = False
            continue
        
        if in_add_library:
            parts = stripped.replace(')', ' ').split()
            for p in parts:
                if (p.endswith('.c') or p.endswith('.cu')) and not p.startswith('#'):
                    sources.append(p)
            if ')' in line:
                in_add_library = False
    
    return lib_name, sources

def compile_source(src_path, obj_path):
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    if os.path.exists(obj_path) and os.path.getmtime(obj_path) > os.path.getmtime(src_path):
        return True, os.path.basename(src_path), 0.0, ""
    
    if src_path.endswith('.c'):
        cmd = ['gcc', '-O3', '-fPIC', '-fopenmp'] + INC_FLAGS + ['-c', src_path, '-o', obj_path]
    else: # .cu
        cmd = [HIPCC] + ARCH_FLAGS + ['-fgpu-rdc', '-fPIC'] + INC_FLAGS + [
            '-include', COMPAT_HEADER, '-c', src_path, '-o', obj_path
        ]
    
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.time() - t0
    
    basename = os.path.basename(src_path)
    if res.returncode != 0:
        return False, basename, dt, res.stderr
    return True, basename, dt, res.stderr

def build_module(module_rel_path, max_workers=12):
    cmake_path = os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', module_rel_path, 'CMakeLists.txt')
    if not os.path.exists(cmake_path):
        print(f"Error: {cmake_path} not found.")
        return False
    
    lib_name, sources = parse_sources_from_cmake(cmake_path)
    print(f"\n{'='*60}")
    print(f"Building {lib_name} from {module_rel_path} ({len(sources)} sources)")
    print(f"{'='*60}")
    
    build_dir = os.path.join(PROJECT_ROOT, 'build', module_rel_path.replace('-', '_'))
    os.makedirs(build_dir, exist_ok=True)
    
    tasks = []
    for src in sources:
        src_path = os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', module_rel_path, src)
        obj_path = os.path.join(build_dir, f"{src}.o")
        tasks.append((src_path, obj_path))
    
    obj_paths = [t[1] for t in tasks]
    
    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compile_source, src_path, obj_path): os.path.basename(src_path) for src_path, obj_path in tasks}
        
        has_error = False
        for future in concurrent.futures.as_completed(futures):
            src_name = futures[future]
            success, name, dt, err = future.result()
            if success:
                print(f"  [OK]   {name:<30} ({dt:.2f}s)")
            else:
                has_error = True
                print(f"  [FAIL] {name:<30} ({dt:.2f}s)")
                print(f"--- Error from {name} ---")
                print(err[:2000])
                print(f"--- End Error ---")
    
    if has_error:
        print(f"Build failed for {lib_name} due to compilation errors.")
        return False
    
    print(f"\nDevice-linking lib{lib_name}.so ...")
    out_so = os.path.join(PROJECT_ROOT, 'gpu4pyscf', 'lib', f"lib{lib_name}.so")
    link_cmd = [HIPCC] + ARCH_FLAGS + ['-fgpu-rdc', '-shared', '-fPIC', '-o', out_so] + obj_paths + [
        f'-L{os.path.join(PROJECT_ROOT, "gpu4pyscf", "lib")}',
        '-lgint', '-fopenmp',
        "-Wl,-rpath,$ORIGIN"
    ]
    
    t0 = time.time()
    link_res = subprocess.run(link_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt_link = time.time() - t0
    
    if link_res.returncode != 0:
        print(f"  [FAIL] Linking {out_so} failed ({dt_link:.2f}s):")
        print(link_res.stderr[:3000])
        return False
    
    size_mb = os.path.getsize(out_so) / (1024 * 1024)
    total_time = time.time() - t_start
    print(f"  [SUCCESS] {out_so} ({size_mb:.1f} MB) built in {total_time:.1f}s!")
    return True

if __name__ == '__main__':
    modules = sys.argv[1:] if len(sys.argv) > 1 else ['gvhf-rys', 'pbc', 'gvhf-md']
    for m in modules:
        ok = build_module(m)
        if not ok:
            print(f"Stopping on error in {m}.")
            sys.exit(1)
    print("\nAll requested modules built successfully!")
