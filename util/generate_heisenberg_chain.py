#!/usr/bin/env python3
"""
Generate Heisenberg chain Hamiltonian files for ED testing
H = J * sum_<i,j> (S_i^x S_j^x + S_i^y S_j^y + S_i^z S_j^z)
"""

import sys
import os

def generate_heisenberg_chain(n_sites, J=1.0, periodic=False, output_dir="./"):
    """
    Generate InterAll.dat and Trans.dat for a Heisenberg chain
    
    Args:
        n_sites: Number of sites
        J: Exchange coupling (default 1.0)
        periodic: Use periodic boundary conditions
        output_dir: Output directory
    """
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate interactions
    interactions = []
    
    # Nearest-neighbor bonds
    n_bonds = n_sites if periodic else n_sites - 1
    
    for i in range(n_bonds):
        site1 = i
        site2 = (i + 1) % n_sites
        
        # Sx * Sx
        interactions.append(f"        0         {site1}           0           {site2}    {J:.6f}    0.000000")
        # Sy * Sy  
        interactions.append(f"        1         {site1}           1           {site2}    {J:.6f}    0.000000")
        # Sz * Sz
        interactions.append(f"        2         {site1}           2           {site2}    {J:.6f}    0.000000")
    
    # Write InterAll.dat
    with open(os.path.join(output_dir, "InterAll.dat"), 'w') as f:
        f.write("===================\n")
        f.write(f"num      {len(interactions)}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for inter in interactions:
            f.write(inter + "\n")
    
    # Write empty Trans.dat (no single-site terms)
    with open(os.path.join(output_dir, "Trans.dat"), 'w') as f:
        f.write("===================\n")
        f.write("num      0\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
    
    print(f"Generated Heisenberg chain with {n_sites} sites")
    print(f"  Bonds: {n_bonds}")
    print(f"  Periodic: {periodic}")
    print(f"  J = {J}")
    print(f"  Output: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_heisenberg_chain.py <n_sites> [J] [periodic] [output_dir]")
        print("Example: python generate_heisenberg_chain.py 20 1.0 false ./heisenberg_20")
        sys.exit(1)
    
    n_sites = int(sys.argv[1])
    J = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    periodic = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    output_dir = sys.argv[4] if len(sys.argv) > 4 else f"./heisenberg_{n_sites}"
    
    generate_heisenberg_chain(n_sites, J, periodic, output_dir)
