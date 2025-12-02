#!/usr/bin/env python3
"""
Check if an ED calculation is feasible on current hardware
Estimates memory requirements and provides recommendations
"""

import sys
import math
import subprocess
import os
from typing import Tuple, Dict

def binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n,k)"""
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

def get_system_memory() -> int:
    """Get available system RAM in bytes"""
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Convert KB to bytes
                        return int(line.split()[1]) * 1024
        # Fallback for non-Linux systems
        import psutil
        return psutil.virtual_memory().available
    except:
        return 0

def get_gpu_memory() -> Dict[int, int]:
    """Get GPU memory for available devices (returns device_id -> memory_bytes)"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        memories = {}
        for i, line in enumerate(result.stdout.strip().split("\n")):
            # Convert MB to bytes
            memories[i] = int(line) * 1024 * 1024
        return memories
    except:
        return {}

def format_bytes(bytes_val: int) -> str:
    """Format bytes in human-readable form"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def estimate_memory_requirements(
    num_sites: int,
    use_fixed_sz: bool = False,
    n_up: int = None,
    use_symmetries: bool = False,
    symmetry_reduction: float = 10.0,  # Typical reduction factor
    method: str = "LANCZOS",
    krylov_dim: int = 200
) -> Dict[str, any]:
    """
    Estimate memory requirements for ED calculation
    
    Returns:
        Dict with keys: hilbert_dim, vector_size, total_memory, feasible, recommendation
    """
    
    # Complex number: 16 bytes (8 bytes real + 8 bytes imag)
    COMPLEX_SIZE = 16
    
    # Calculate Hilbert space dimension
    if use_fixed_sz:
        if n_up is None:
            n_up = num_sites // 2
        hilbert_dim = binomial(num_sites, n_up)
    else:
        hilbert_dim = 2 ** num_sites
    
    # Apply symmetry reduction
    if use_symmetries:
        hilbert_dim = hilbert_dim // symmetry_reduction
    
    # Memory per vector
    vector_size = hilbert_dim * COMPLEX_SIZE
    
    # Estimate total memory based on method
    if method in ["LANCZOS", "LANCZOS_SELECTIVE", "LANCZOS_NO_ORTHO"]:
        # Lanczos: needs ~3-5 vectors at a time
        total_memory = vector_size * 5
    elif method in ["FTLM", "LTLM"]:
        # FTLM: Optimized memory usage with block reorthogonalization
        # Typically only needs 5-10 active Krylov vectors + working space
        # The full Krylov basis is only needed for diagonalization (done on small matrix)
        total_memory = vector_size * 10  # 10 active vectors is realistic
    elif method in ["FTLM_GPU_FIXED_SZ", "LANCZOS_GPU_FIXED_SZ"]:
        # GPU methods: More memory efficient due to on-device operations
        # Typically only 3-5 vectors on GPU at once
        total_memory = vector_size * 5
    elif method in ["mTPQ", "cTPQ", "mTPQ_GPU"]:
        # TPQ: Multiple samples but sequential, only 2-3 vectors active
        total_memory = vector_size * 3
    elif method == "DAVIDSON":
        # Davidson: larger subspace
        total_memory = vector_size * min(50, krylov_dim)
    elif method == "FULL":
        # Full diagonalization: needs to store matrix
        total_memory = hilbert_dim * hilbert_dim * COMPLEX_SIZE
    else:
        # Conservative estimate
        total_memory = vector_size * 10
    
    # Check feasibility
    system_memory = get_system_memory()
    gpu_memories = get_gpu_memory()
    
    feasible_cpu = system_memory > 0 and total_memory < system_memory * 0.8  # Use 80% threshold
    feasible_gpu = any(mem > total_memory * 0.8 for mem in gpu_memories.values())
    
    # Generate recommendation
    recommendation = []
    
    if hilbert_dim > 1e9:
        recommendation.append("⚠️  VERY LARGE SYSTEM - Use dimension reduction!")
        if not use_fixed_sz:
            recommendation.append("   → Enable Fixed-Sz (--fixed-sz --n-up=N)")
        if not use_symmetries:
            recommendation.append("   → Consider spatial symmetries (--symmetrized)")
    
    if vector_size > 20e9:  # > 20 GB per vector
        recommendation.append("⚠️  Large memory per vector")
        if method not in ["FTLM", "LTLM", "mTPQ", "cTPQ"]:
            recommendation.append("   → Use FTLM/LTLM/TPQ methods (avoid full storage)")
    
    if not feasible_cpu and not feasible_gpu:
        recommendation.append("❌ INSUFFICIENT MEMORY")
        recommendation.append(f"   Required: {format_bytes(total_memory)}")
        recommendation.append(f"   Available CPU: {format_bytes(system_memory)}")
        if gpu_memories:
            max_gpu = max(gpu_memories.values())
            recommendation.append(f"   Available GPU: {format_bytes(max_gpu)}")
        
        # Suggest alternatives
        if not use_fixed_sz:
            reduction_with_fixed_sz = 2 ** num_sites / binomial(num_sites, num_sites // 2)
            recommendation.append(f"\n   Try Fixed-Sz: ~{reduction_with_fixed_sz:.1f}× smaller")
        if not use_symmetries:
            recommendation.append("   Try symmetries: ~10-100× smaller (system dependent)")
        recommendation.append("   Use FTLM method: Only stores Krylov vectors")
    
    elif feasible_gpu and not feasible_cpu:
        recommendation.append("✅ FEASIBLE WITH GPU")
        best_gpu = max(gpu_memories.items(), key=lambda x: x[1])
        recommendation.append(f"   Use GPU {best_gpu[0]} with {format_bytes(best_gpu[1])} memory")
        recommendation.append(f"   Add --method=FTLM_GPU_FIXED_SZ or LANCZOS_GPU_FIXED_SZ")
    
    elif feasible_cpu:
        recommendation.append("✅ FEASIBLE WITH CPU")
        recommendation.append(f"   Estimated memory: {format_bytes(total_memory)}")
        if feasible_gpu:
            recommendation.append("   GPU also available - consider GPU methods for speed")
    
    return {
        "num_sites": num_sites,
        "use_fixed_sz": use_fixed_sz,
        "n_up": n_up,
        "use_symmetries": use_symmetries,
        "hilbert_dim": hilbert_dim,
        "vector_size": vector_size,
        "total_memory": total_memory,
        "system_memory": system_memory,
        "gpu_memories": gpu_memories,
        "feasible_cpu": feasible_cpu,
        "feasible_gpu": feasible_gpu,
        "recommendation": recommendation
    }

def print_analysis(results: Dict):
    """Print formatted analysis"""
    print("\n" + "="*70)
    print("ED SYSTEM FEASIBILITY ANALYSIS")
    print("="*70)
    
    print(f"\nSystem Configuration:")
    print(f"  Sites:              {results['num_sites']}")
    print(f"  Fixed-Sz:           {results['use_fixed_sz']}")
    if results['use_fixed_sz']:
        print(f"  n_up:               {results['n_up']}")
    print(f"  Spatial symmetries: {results['use_symmetries']}")
    
    print(f"\nHilbert Space:")
    print(f"  Dimension:          {results['hilbert_dim']:,}")
    print(f"  Vector size:        {format_bytes(results['vector_size'])}")
    print(f"  Est. total memory:  {format_bytes(results['total_memory'])}")
    
    print(f"\nAvailable Resources:")
    print(f"  System RAM:         {format_bytes(results['system_memory'])}")
    if results['gpu_memories']:
        for gpu_id, mem in results['gpu_memories'].items():
            print(f"  GPU {gpu_id} memory:      {format_bytes(mem)}")
    else:
        print(f"  GPU memory:         Not available")
    
    print(f"\nFeasibility:")
    print(f"  CPU:                {'✅ Yes' if results['feasible_cpu'] else '❌ No'}")
    print(f"  GPU:                {'✅ Yes' if results['feasible_gpu'] else '❌ No'}")
    
    if results['recommendation']:
        print(f"\nRecommendations:")
        for rec in results['recommendation']:
            print(f"  {rec}")
    
    print("\n" + "="*70 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: check_system_feasibility.py <num_sites> [options]")
        print("\nOptions:")
        print("  --fixed-sz              Use Fixed-Sz sector")
        print("  --n-up=N                Number of up spins (default: num_sites/2)")
        print("  --symmetries            Use spatial symmetries")
        print("  --symmetry-factor=X     Expected symmetry reduction factor (default: 10)")
        print("  --method=METHOD         ED method (default: LANCZOS)")
        print("  --krylov-dim=K          Krylov dimension (default: 200)")
        print("\nExamples:")
        print("  # Check 32 sites with fixed-Sz")
        print("  python3 check_system_feasibility.py 32 --fixed-sz")
        print("\n  # Check 32 sites with fixed-Sz and symmetries")
        print("  python3 check_system_feasibility.py 32 --fixed-sz --symmetries")
        print("\n  # Check 32 sites with FTLM method")
        print("  python3 check_system_feasibility.py 32 --fixed-sz --method=FTLM")
        sys.exit(1)
    
    num_sites = int(sys.argv[1])
    
    # Parse options
    use_fixed_sz = "--fixed-sz" in sys.argv
    use_symmetries = "--symmetries" in sys.argv
    
    n_up = None
    symmetry_factor = 10.0
    method = "LANCZOS"
    krylov_dim = 200
    
    for arg in sys.argv[2:]:
        if arg.startswith("--n-up="):
            n_up = int(arg.split("=")[1])
        elif arg.startswith("--symmetry-factor="):
            symmetry_factor = float(arg.split("=")[1])
        elif arg.startswith("--method="):
            method = arg.split("=")[1]
        elif arg.startswith("--krylov-dim="):
            krylov_dim = int(arg.split("=")[1])
    
    # Run analysis
    results = estimate_memory_requirements(
        num_sites=num_sites,
        use_fixed_sz=use_fixed_sz,
        n_up=n_up,
        use_symmetries=use_symmetries,
        symmetry_reduction=symmetry_factor,
        method=method,
        krylov_dim=krylov_dim
    )
    
    print_analysis(results)
    
    # Return exit code based on feasibility
    if results['feasible_cpu'] or results['feasible_gpu']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
