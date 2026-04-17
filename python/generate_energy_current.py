#!/usr/bin/env python3
"""
Generate the energy current operator J^E for a spin model on the pyrochlore lattice.

The energy current is derived from the continuity equation for the local energy density.
For H = sum_{<ij>} h_{ij}, the energy current is:

    J^E_mu = (i/2) sum_b R_{b,mu} [H, h_b]

where R_b = (r_i + r_j)/2 is the bond center.  Only pairs of bonds sharing
a site contribute, giving:

    J^E_mu = (i/4) sum_{j} sum_{i,k in nn(j), i!=k} (r_i - r_k)_mu [h_{jk}, h_{ij}]

Each commutator [h_{jk}, h_{ij}] produces three-body terms S^a_i S^b_j S^c_k
through the spin algebra [S^alpha, S^beta] structure constants.

The output is written in ThreeBodyG.dat format for the ED solver.

Usage:
    python generate_energy_current.py <hamiltonian_dir> [--output_dir <dir>] [--field <hx> <hy> <hz>]

The script reads:
    - positions.dat  (site positions in fractional coordinates)
    - InterAll.dat   (bond Hamiltonian terms)
and writes:
    - JEx.dat, JEx.dat.3body   (energy current x-component)
    - JEy.dat, JEy.dat.3body   (energy current y-component)
    - Optional: Trans.dat with magnetic field
"""

import numpy as np
from collections import defaultdict
import argparse
import os
import sys


# Spin operator conventions: 0 = S+, 1 = S-, 2 = Sz
SP, SM, SZ = 0, 1, 2
OP_NAMES = {SP: "S+", SM: "S-", SZ: "Sz"}

# Structure constants: [A, B] = sum_C f_{AB}^C * C
# f[op_A][op_B] = list of (op_C, coefficient)
COMMUTATOR = {}
COMMUTATOR[(SP, SM)] = [(SZ, 2.0)]     # [S+, S-] = 2Sz
COMMUTATOR[(SM, SP)] = [(SZ, -2.0)]    # [S-, S+] = -2Sz
COMMUTATOR[(SZ, SP)] = [(SP, 1.0)]     # [Sz, S+] = S+
COMMUTATOR[(SZ, SM)] = [(SM, -1.0)]    # [Sz, S-] = -S-
COMMUTATOR[(SP, SZ)] = [(SP, -1.0)]    # [S+, Sz] = -S+
COMMUTATOR[(SM, SZ)] = [(SM, 1.0)]     # [S-, Sz] = S-
# All same-type commutators are zero: [S+,S+]=[S-,S-]=[Sz,Sz]=0


def read_positions(filename):
    """Read positions.dat and return dict: site_id -> (sublattice, x, y, z)."""
    positions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            site_id = int(parts[0])
            sublattice = int(parts[2])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            positions[site_id] = (sublattice, np.array([x, y, z]))
    return positions


def read_interall(filename):
    """Read InterAll.dat and return list of two-body terms.
    
    Returns:
        bonds: dict mapping frozenset({i,j}) -> list of (op_i, site_i, op_j, site_j, coeff)
        all_terms: list of all terms as tuples
    """
    terms = []
    with open(filename) as f:
        # Read header
        f.readline()  # "==================="
        line = f.readline()  # "num  240"
        num_terms = int(line.split()[1])
        for _ in range(3):
            f.readline()  # separator lines
        
        for _ in range(num_terms):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            op_i = int(parts[0])
            site_i = int(parts[1])
            op_j = int(parts[2])
            site_j = int(parts[3])
            real = float(parts[4])
            imag = float(parts[5])
            coeff = complex(real, imag)
            if abs(coeff) > 1e-15:
                terms.append((op_i, site_i, op_j, site_j, coeff))
    
    # Group terms by bond (unordered pair of sites)
    bonds = defaultdict(list)
    for (op_i, site_i, op_j, site_j, coeff) in terms:
        key = frozenset({site_i, site_j})
        bonds[key].append((op_i, site_i, op_j, site_j, coeff))
    
    return bonds, terms


def minimum_image_displacement(r_i, r_j):
    """Compute displacement r_j - r_i with PBC (unit cube)."""
    dr = r_j - r_i
    dr -= np.round(dr)
    return dr


def compute_bond_commutator(terms_b1, terms_b2, shared_site):
    """Compute [h_{b2}, h_{b1}] for two bonds sharing `shared_site`.
    
    Each term in h = c * O^a_site_a O^b_site_b.
    The commutator acts only on the shared site.
    
    Args:
        terms_b1: list of (op_i, site_i, op_j, site_j, coeff) for bond b1
        terms_b2: list of (op_i, site_i, op_j, site_j, coeff) for bond b2
        shared_site: the site index shared by b1 and b2
    
    Returns:
        list of (op_p, site_p, op_s, site_s, op_q, site_q, coeff)
        representing three-body terms: coeff * O^{op_p}_{site_p} O^{op_s}_{site_s} O^{op_q}_{site_q}
        where site_p is unshared site of b1, site_s is shared, site_q is unshared of b2.
    """
    three_body = []
    
    for (op_a1, sa1, op_b1, sb1, c1) in terms_b1:
        # Identify which operator in b1 is on the shared site
        if sa1 == shared_site:
            op_shared_b1 = op_a1
            op_unshared_b1 = op_b1
            site_unshared_b1 = sb1
        elif sb1 == shared_site:
            op_shared_b1 = op_b1
            op_unshared_b1 = op_a1
            site_unshared_b1 = sa1
        else:
            continue  # term doesn't involve shared site (shouldn't happen)
        
        for (op_a2, sa2, op_b2, sb2, c2) in terms_b2:
            # Identify which operator in b2 is on the shared site
            if sa2 == shared_site:
                op_shared_b2 = op_a2
                op_unshared_b2 = op_b2
                site_unshared_b2 = sb2
            elif sb2 == shared_site:
                op_shared_b2 = op_b2
                op_unshared_b2 = op_a2
                site_unshared_b2 = sa2
            else:
                continue
            
            # Compute [O^{shared_b2}, O^{shared_b1}] on the shared site
            comm_key = (op_shared_b2, op_shared_b1)
            if comm_key not in COMMUTATOR:
                continue  # zero commutator
            
            for (op_result, f_coeff) in COMMUTATOR[comm_key]:
                total_coeff = c2 * c1 * f_coeff
                if abs(total_coeff) < 1e-15:
                    continue
                # Result: total_coeff * O^{unshared_b1}_{site_p} O^{result}_{shared} O^{unshared_b2}_{site_q}
                three_body.append((
                    op_unshared_b1, site_unshared_b1,
                    op_result, shared_site,
                    op_unshared_b2, site_unshared_b2,
                    total_coeff
                ))
    
    return three_body


def generate_energy_current(positions, bonds, terms, component='x'):
    """Generate the energy current operator J^E_mu.
    
    Uses: J^E_mu = (i/2) sum_b R_{b,mu} [H, h_b]
         = (i/2) sum_{b1} R_{b1,mu} sum_{b2 != b1} [h_{b2}, h_{b1}]
    
    Only pairs sharing a site contribute.
    
    Args:
        positions: dict site_id -> (sublattice, position_array)
        bonds: dict frozenset({i,j}) -> list of terms
        terms: all InterAll terms
        component: 'x', 'y', or 'z' (which component of J^E)
    
    Returns:
        dict: (op1, s1, op2, s2, op3, s3) -> complex coefficient
    """
    comp_idx = {'x': 0, 'y': 1, 'z': 2}[component]
    
    # Build adjacency: site -> set of neighbor sites
    adjacency = defaultdict(set)
    for bond_key in bonds:
        sites = list(bond_key)
        if len(sites) == 2:
            adjacency[sites[0]].add(sites[1])
            adjacency[sites[1]].add(sites[0])
    
    # Accumulate three-body terms
    three_body_terms = defaultdict(complex)
    
    # For each bond b1
    for b1_key in bonds:
        b1_sites = list(b1_key)
        if len(b1_sites) != 2:
            continue
        si, sj = b1_sites
        
        # Bond center R_b1
        _, ri = positions[si]
        _, rj = positions[sj]
        # Use minimum image for the bond midpoint position
        dr = minimum_image_displacement(ri, rj)
        R_b1 = ri + dr / 2.0
        prefactor_pos = R_b1[comp_idx]  # mu-component of bond center
        
        # Find all neighboring bonds b2 sharing a site with b1
        # Bonds sharing site si
        for sk in adjacency[si]:
            if sk == sj:
                continue  # same bond
            b2_key = frozenset({si, sk})
            if b2_key not in bonds:
                continue
            shared = si
            comm_terms = compute_bond_commutator(bonds[b1_key], bonds[b2_key], shared)
            for (op_p, sp, op_s, ss, op_q, sq, coeff) in comm_terms:
                # Prefactor: (i/2) * R_{b1,mu}
                total = 0.5j * prefactor_pos * coeff
                key = (op_p, sp, op_s, ss, op_q, sq)
                three_body_terms[key] += total
        
        # Bonds sharing site sj
        for sk in adjacency[sj]:
            if sk == si:
                continue
            b2_key = frozenset({sj, sk})
            if b2_key not in bonds:
                continue
            shared = sj
            comm_terms = compute_bond_commutator(bonds[b1_key], bonds[b2_key], shared)
            for (op_p, sp, op_s, ss, op_q, sq, coeff) in comm_terms:
                total = 0.5j * prefactor_pos * coeff
                key = (op_p, sp, op_s, ss, op_q, sq)
                three_body_terms[key] += total
    
    # Filter near-zero terms
    filtered = {}
    for key, coeff in three_body_terms.items():
        if abs(coeff) > 1e-15:
            filtered[key] = coeff
    
    return filtered


def write_empty_interall(filename, num_sites):
    """Write an empty InterAll.dat file (no two-body terms)."""
    with open(filename, 'w') as f:
        f.write("===================\n")
        f.write("num      0\n")
        f.write("===================\n")
        f.write("===================\n")


def write_three_body_file(filename, three_body_terms):
    """Write ThreeBodyG.dat format file.
    
    Format: op1 site1 op2 site2 op3 site3 real imag
    """
    n = len(three_body_terms)
    with open(filename, 'w') as f:
        f.write("===================\n")
        f.write(f"num      {n}\n")
        f.write("===================\n")
        f.write("===================\n")
        for (op1, s1, op2, s2, op3, s3), coeff in three_body_terms.items():
            f.write(f"        {op1}         {s1}           {op2}          {s2}           {op3}          {s3}    {coeff.real:.10f}    {coeff.imag:.10f}\n")


def write_trans_field(filename, num_sites, hx=0.0, hy=0.0, hz=0.0):
    """Write Trans.dat with a uniform magnetic field."""
    if abs(hx) < 1e-15 and abs(hy) < 1e-15 and abs(hz) < 1e-15:
        # Write empty Trans.dat
        with open(filename, 'w') as f:
            f.write("===================\n")
            f.write("num      0\n")
            f.write("===================\n")
            f.write("===================\n")
        return
    
    # h_x S_x = h_x/2 (S+ + S-) => (h_x/2) S+ + (h_x/2) S-
    # h_y S_y = h_y/(2i) (S+ - S-) => (-ih_y/2) S+ + (ih_y/2) S-
    # h_z S_z => h_z Sz
    # In Trans.dat convention: op_type site real imag
    # op_type: 0=S+, 1=S-, 2=Sz
    entries = []
    for site in range(num_sites):
        # S+ coefficient: (hx - i*hy) / 2
        sp_coeff = complex(hx, -hy) / 2.0
        if abs(sp_coeff) > 1e-15:
            entries.append((0, site, sp_coeff))
        # S- coefficient: (hx + i*hy) / 2
        sm_coeff = complex(hx, hy) / 2.0
        if abs(sm_coeff) > 1e-15:
            entries.append((1, site, sm_coeff))
        # Sz coefficient: hz
        if abs(hz) > 1e-15:
            entries.append((2, site, complex(hz, 0)))
    
    with open(filename, 'w') as f:
        f.write("===================\n")
        f.write(f"num      {len(entries)}\n")
        f.write("===================\n")
        f.write("===================\n")
        for (op, site, coeff) in entries:
            f.write(f"        {op}         {site}    {coeff.real:.10f}    {coeff.imag:.10f}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate energy current operator for thermal Hall computation")
    parser.add_argument("hamiltonian_dir", help="Directory containing positions.dat and InterAll.dat")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: hamiltonian_dir)")
    parser.add_argument("--field", nargs=3, type=float, default=[0, 0, 0.1],
                        metavar=("hx", "hy", "hz"),
                        help="Magnetic field components (default: 0 0 0.1)")
    args = parser.parse_args()
    
    ham_dir = args.hamiltonian_dir
    out_dir = args.output_dir or ham_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Read lattice data
    pos_file = os.path.join(ham_dir, "positions.dat")
    interall_file = os.path.join(ham_dir, "InterAll.dat")
    
    print(f"Reading positions from {pos_file}")
    positions = read_positions(pos_file)
    num_sites = len(positions)
    print(f"  Found {num_sites} sites")
    
    print(f"Reading Hamiltonian from {interall_file}")
    bonds, terms = read_interall(interall_file)
    print(f"  Found {len(bonds)} bonds, {len(terms)} terms")
    
    # Generate energy current operators
    for comp in ['x', 'y']:
        print(f"\nGenerating J^E_{comp}...")
        three_body = generate_energy_current(positions, bonds, terms, component=comp)
        print(f"  Generated {len(three_body)} three-body terms")
        
        # Diagnostics
        max_coeff = max(abs(c) for c in three_body.values()) if three_body else 0
        n_real = sum(1 for c in three_body.values() if abs(c.imag) < 1e-10 * abs(c))
        n_imag = sum(1 for c in three_body.values() if abs(c.real) < 1e-10 * abs(c))
        n_complex = len(three_body) - n_real - n_imag
        print(f"  Max |coeff| = {max_coeff:.6e}")
        print(f"  Real: {n_real}, Pure imag: {n_imag}, Complex: {n_complex}")
        
        # Write files
        interall_path = os.path.join(out_dir, f"JE{comp}.dat")
        threebody_path = os.path.join(out_dir, f"JE{comp}.dat.3body")
        
        write_empty_interall(interall_path, num_sites)
        write_three_body_file(threebody_path, three_body)
        print(f"  Written: {interall_path}")
        print(f"  Written: {threebody_path}")
    
    # Write magnetic field (Trans.dat) for TRS breaking
    hx, hy, hz = args.field
    if abs(hx) > 1e-15 or abs(hy) > 1e-15 or abs(hz) > 1e-15:
        trans_path = os.path.join(out_dir, "Trans.dat")
        write_trans_field(trans_path, num_sites, hx, hy, hz)
        print(f"\nMagnetic field: h = ({hx}, {hy}, {hz})")
        print(f"  Written: {trans_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Energy current operator generation complete.")
    print("="*60)
    print(f"\nTo compute kappa_xy, run ED with:")
    print(f"  --dyn-operator=JEx.dat --dyn-operator2=JEy.dat")
    print(f"  --hamiltonian-dir={out_dir}")
    print(f"\nThis computes S_{{JEx,JEy}}(omega) = <JEx†(t) JEy>")
    print(f"The thermal Hall conductivity is proportional to")
    print(f"  kappa_xy ~ (1/T) * Im[S(omega->0)] / omega")


if __name__ == "__main__":
    main()
