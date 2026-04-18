#!/usr/bin/env python3
"""
HDF5 File Inspector for ED/TPQ_DSSF Output Files

Usage:
    python3 h5inspect.py <file.h5>                    # Show structure
    python3 h5inspect.py <file.h5> --full             # Show structure + sample data
    python3 h5inspect.py <file.h5> --path /eigendata  # Inspect specific path
    python3 h5inspect.py <file.h5> --dump /eigendata/eigenvalues  # Dump dataset

Quick alternatives (no Python needed):
    h5ls -r file.h5              # Show structure
    h5ls -rv file.h5             # Show with data types  
    h5dump -d /path file.h5      # Dump specific dataset
"""

try:
    import h5py
except ImportError:
    print("Error: h5py not installed. Install with: pip install h5py")
    print("\nAlternatives using built-in HDF5 tools:")
    print("  h5ls -r <file.h5>           # Show structure")
    print("  h5ls -rv <file.h5>          # Show with data types")
    print("  h5dump -H <file.h5>         # Show header info")
    print("  h5dump -d /path <file.h5>   # Dump specific dataset")
    import sys
    sys.exit(1)

import numpy as np
import argparse
import sys
from pathlib import Path


def format_value(val, max_items=8):
    """Format a value for display, truncating if needed."""
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    if isinstance(val, np.ndarray):
        if val.size <= max_items:
            return np.array2string(val, precision=6, suppress_small=True)
        else:
            flat = val.flatten()
            return f"[{flat[0]:.6g}, {flat[1]:.6g}, ... , {flat[-2]:.6g}, {flat[-1]:.6g}]"
    if isinstance(val, (np.floating, float)):
        return f"{val:.6g}"
    if isinstance(val, (np.complexfloating, complex)):
        return f"{val.real:.6g}{val.imag:+.6g}j"
    return str(val)


def format_shape(shape):
    """Format shape tuple."""
    return "×".join(str(s) for s in shape) if shape else "scalar"


def format_dtype(dtype):
    """Format dtype for display."""
    if np.issubdtype(dtype, np.complexfloating):
        return "complex"
    if np.issubdtype(dtype, np.floating):
        return "float"
    if np.issubdtype(dtype, np.integer):
        return "int"
    if dtype.names:  # compound type
        return f"compound({', '.join(dtype.names)})"
    return str(dtype)


def print_attrs(obj, indent=""):
    """Print attributes of an HDF5 object."""
    if len(obj.attrs) > 0:
        for key, val in obj.attrs.items():
            val_str = format_value(val)
            print(f"{indent}  @{key} = {val_str}")


def inspect_group(group, path="", indent="", show_data=False, max_depth=10):
    """Recursively inspect an HDF5 group."""
    if max_depth <= 0:
        print(f"{indent}  ... (max depth reached)")
        return
    
    # Print attributes
    print_attrs(group, indent)
    
    # Iterate over items
    for name in sorted(group.keys()):
        item_path = f"{path}/{name}"
        try:
            item = group[name]
        except Exception as e:
            print(f"{indent}├── {name} [ERROR: {e}]")
            continue
        
        if isinstance(item, h5py.Group):
            n_children = len(item.keys())
            n_attrs = len(item.attrs)
            suffix = f" ({n_children} items, {n_attrs} attrs)" if n_attrs else f" ({n_children} items)"
            print(f"{indent}├── 📁 {name}/{suffix}")
            inspect_group(item, item_path, indent + "│   ", show_data, max_depth - 1)
        elif isinstance(item, h5py.Dataset):
            shape_str = format_shape(item.shape)
            dtype_str = format_dtype(item.dtype)
            size_bytes = item.nbytes
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            print(f"{indent}├── 📊 {name} [{shape_str}] {dtype_str} ({size_str})")
            print_attrs(item, indent + "│   ")
            
            if show_data and item.size > 0:
                try:
                    data = item[()]
                    preview = format_value(data, max_items=10)
                    print(f"{indent}│     → {preview}")
                except Exception as e:
                    print(f"{indent}│     → [Error reading: {e}]")


def dump_dataset(f, path):
    """Dump full contents of a dataset."""
    if path not in f:
        print(f"Path not found: {path}")
        return
    
    item = f[path]
    if isinstance(item, h5py.Group):
        print(f"'{path}' is a group, not a dataset. Contents:")
        for name in item.keys():
            print(f"  - {name}")
        return
    
    print(f"Dataset: {path}")
    print(f"Shape: {item.shape}")
    print(f"Dtype: {item.dtype}")
    print("-" * 40)
    
    data = item[()]
    
    # Handle different data types
    if data.dtype.names:  # Compound type (e.g., complex)
        for i, row in enumerate(data.flatten()):
            if i >= 50:
                print(f"... ({len(data.flatten()) - 50} more entries)")
                break
            vals = [f"{name}={row[name]:.10g}" for name in data.dtype.names]
            print(f"[{i}] {', '.join(vals)}")
    elif np.issubdtype(data.dtype, np.complexfloating):
        for i, val in enumerate(data.flatten()):
            if i >= 50:
                print(f"... ({len(data.flatten()) - 50} more entries)")
                break
            print(f"[{i}] {val.real:+.10e} {val.imag:+.10e}j")
    else:
        # Regular numeric data
        if data.ndim == 1:
            for i, val in enumerate(data):
                if i >= 50:
                    print(f"... ({len(data) - 50} more entries)")
                    break
                print(f"[{i}] {val}")
        else:
            print(data)


def summarize_ed_results(f):
    """Summarize typical ED results file."""
    print("\n" + "=" * 60)
    print("ED RESULTS SUMMARY")
    print("=" * 60)
    
    # Eigenvalues
    if "/eigendata/eigenvalues" in f:
        eigs = f["/eigendata/eigenvalues"][()]
        print(f"\n📊 Eigenvalues: {len(eigs)} computed")
        print(f"   Ground state: E₀ = {eigs[0]:.10f}")
        if len(eigs) > 1:
            print(f"   First gap: ΔE = {eigs[1] - eigs[0]:.6f}")
        if len(eigs) > 5:
            print(f"   Range: [{eigs[0]:.4f}, {eigs[-1]:.4f}]")
    
    # Eigenvectors
    n_eigvecs = sum(1 for k in f.get("/eigendata", {}).keys() if k.startswith("eigenvector_"))
    if n_eigvecs > 0:
        print(f"\n📊 Eigenvectors: {n_eigvecs} stored")
    
    # TPQ samples
    if "/tpq/samples" in f:
        samples = list(f["/tpq/samples"].keys())
        print(f"\n📊 TPQ Samples: {len(samples)}")
        if samples:
            sample0 = f[f"/tpq/samples/{samples[0]}"]
            if "thermodynamics" in sample0:
                thermo = sample0["thermodynamics"][()]
                print(f"   β range: [{thermo[0, 0]:.4f}, {thermo[-1, 0]:.4f}]")
    
    # DSSF results
    if "/spectral" in f:
        ops = [k for k in f["/spectral"].keys() if k != "frequencies"]
        if ops:
            print(f"\n📊 Spectral Functions: {len(ops)} operators")
            for op in ops[:5]:
                print(f"   - {op}")
            if len(ops) > 5:
                print(f"   ... and {len(ops) - 5} more")
    
    print()


def summarize_dssf_results(f):
    """Summarize typical DSSF results file."""
    print("\n" + "=" * 60)
    print("DSSF RESULTS SUMMARY")
    print("=" * 60)
    
    # Metadata
    if "/metadata" in f:
        meta = f["/metadata"]
        print("\n📋 Metadata:")
        for attr in meta.attrs:
            print(f"   {attr}: {format_value(meta.attrs[attr])}")
    
    # Momentum points
    if "/momentum_points/q_vectors" in f:
        q = f["/momentum_points/q_vectors"][()]
        print(f"\n📍 Momentum Points: {len(q)}")
        for i, qi in enumerate(q[:3]):
            print(f"   Q{i}: ({qi[0]/np.pi:.3f}π, {qi[1]/np.pi:.3f}π, {qi[2]/np.pi:.3f}π)")
        if len(q) > 3:
            print(f"   ... and {len(q) - 3} more")
    
    # Spectral functions
    if "/spectral" in f:
        if "frequencies" in f["/spectral"]:
            freq = f["/spectral/frequencies"][()]
            print(f"\n📊 Frequency Grid: {len(freq)} points, ω ∈ [{freq[0]:.3f}, {freq[-1]:.3f}]")
        
        ops = [k for k in f["/spectral"].keys() if k != "frequencies"]
        if ops:
            print(f"\n📊 Operators: {len(ops)}")
            for op in ops[:5]:
                op_grp = f[f"/spectral/{op}"]
                temps = list(op_grp.keys())
                print(f"   - {op} ({len(temps)} temperature points)")
            if len(ops) > 5:
                print(f"   ... and {len(ops) - 5} more")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 files from ED/TPQ_DSSF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.h5                    Show file structure
  %(prog)s results.h5 --full             Show structure with data preview
  %(prog)s results.h5 --path /eigendata  Inspect specific group
  %(prog)s results.h5 --dump /eigendata/eigenvalues  Dump full dataset
  %(prog)s results.h5 --summary          Show high-level summary
        """
    )
    parser.add_argument("file", help="HDF5 file to inspect")
    parser.add_argument("--full", "-f", action="store_true", help="Show data previews")
    parser.add_argument("--path", "-p", help="Inspect specific path")
    parser.add_argument("--dump", "-d", help="Dump full dataset at path")
    parser.add_argument("--summary", "-s", action="store_true", help="Show high-level summary")
    parser.add_argument("--depth", type=int, default=10, help="Maximum depth to traverse")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    try:
        with h5py.File(args.file, "r") as f:
            print(f"\n📂 {args.file}")
            print(f"   Size: {Path(args.file).stat().st_size / (1024*1024):.2f} MB")
            print()
            
            if args.dump:
                dump_dataset(f, args.dump)
            elif args.summary:
                # Auto-detect file type and summarize
                if "/eigendata" in f or "/tpq" in f:
                    summarize_ed_results(f)
                if "/spectral" in f or "/metadata" in f:
                    summarize_dssf_results(f)
                if "/eigendata" not in f and "/tpq" not in f and "/spectral" not in f:
                    print("Unknown file format. Use --full for detailed view.")
            elif args.path:
                if args.path in f:
                    item = f[args.path]
                    if isinstance(item, h5py.Group):
                        print(f"📁 {args.path}/")
                        inspect_group(item, args.path, "", args.full, args.depth)
                    else:
                        dump_dataset(f, args.path)
                else:
                    print(f"Path not found: {args.path}")
            else:
                print("📁 /")
                inspect_group(f, "", "", args.full, args.depth)
                
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
