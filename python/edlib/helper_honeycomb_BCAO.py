import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

# Honeycomb lattice parameters
site_basis = np.array([[0, 0], [0, 1/np.sqrt(3)]])
basis = np.array([[1, 0], [0.5, np.sqrt(3)/2]])
bond_types = ['x', 'y', 'z']

def parse_config_file(config_file):
    """Parse configuration file and return parameter dictionary"""
    params = {}
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found")
        return params

    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key in ['dim1', 'dim2', 'pbc']:
                    params[key] = int(value)
                elif key in ['h', 'J1xy', 'J1z', 'D', 'E', 'F', 'G', 'J3xy', 'J3z']:
                    params[key] = float(value)
                elif key == 'field_dir':
                    coords = [float(x.strip()) for x in value.split(',')]
                    if len(coords) == 3:
                        params['hx'], params['hy'], params['hz'] = coords
                else:
                    params[key] = value
    return params

def get_config_or_args():
    parser = argparse.ArgumentParser(description='Generate honeycomb lattice Kitaev Hamiltonian')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--dim1', type=int, default=4, help='Number of unit cells in the x-direction')
    parser.add_argument('--dim2', type=int, default=4, help='Number of unit cells in the y-direction')
    parser.add_argument('--pbc', type=int, default=1, help='Use periodic boundary conditions (1) or open boundary conditions (0)')
    parser.add_argument('--J1xy', type=float, default=-6.54, help='J1 coupling constant')
    parser.add_argument('--J1z', type=float, default=0.36, help='Delta1 coupling constant')
    parser.add_argument('--J3xy', type=float, default=0.15, help='Jpmpm coupling constant')
    parser.add_argument('--J3z', type=float, default=-3.76, help='Jzpm coupling constant')
    parser.add_argument('--D', type=float, default=-0.21, help='D coupling constant')
    parser.add_argument('--E', type=float, default=1.7, help='E coupling constant')
    parser.add_argument('--F', type=float, default=1.7, help='F coupling constant')
    parser.add_argument('--G', type=float, default=1.7, help='G coupling constant')
    parser.add_argument('--h', type=float, default=0.0, help='Field strength')
    parser.add_argument('--hx', type=float, default=1.0, help='Field strength in x-direction')
    parser.add_argument('--hy', type=float, default=0.0, help='Field strength in y-direction')
    parser.add_argument('--hz', type=float, default=0.0, help='Field strength in z-direction')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    params = vars(args)
    if args.config:
        config_params = parse_config_file(args.config)
        params.update(config_params)
    return argparse.Namespace(**params)

def flattenIndex(indices, dim2):
    flattened = []
    for idx in indices:
        i, j, u = idx
        flattened.append(i*dim2*2 + j*2 + u)
    return flattened

def indices_open_BC(i, j, u, d1, d2):
    neighbors = []
    if u == 0:
        neighbors.append([i, j, 1])
        neighbors.append([i, j-1, 1])
        neighbors.append([i+1, j-1, 1])
    else:
        neighbors.append([i, j, 0])
        neighbors.append([i, j+1, 0])
        neighbors.append([i-1, j+1, 0])
    return neighbors

def indices_open_BC_NNN(i, j, u, d1, d2):
    neighbors = []
    neighbors.append([i+1, j, u])
    neighbors.append([i, j+1, u])
    neighbors.append([i+1, j-1, u])
    neighbors.append([i-1, j, u])
    neighbors.append([i, j-1, u])
    neighbors.append([i-1, j+1, u])
    return neighbors

def indices_open_BC_NNNN(i, j, u, d1, d2):
    neighbors = []
    if u == 0:
        neighbors.append([i+1, j, 1])
        neighbors.append([i-1, j, 1])
        neighbors.append([i+1, j-2, 1])
    else:
        neighbors.append([i-1, j, 0])
        neighbors.append([i+1, j, 0])
        neighbors.append([i-1, j+2, 0])
    return neighbors

def indices_periodic_BC(i, j, u, d1, d2):
    neighbors = []
    if u == 0:
        neighbors.append([i, j, 1])
        neighbors.append([np.mod(i, d1), np.mod(j-1, d2), 1])
        neighbors.append([np.mod(i+1, d1), np.mod(j-1, d2), 1])
    else:
        neighbors.append([i, j, 0])
        neighbors.append([np.mod(i, d1), np.mod(j+1, d2), 0])
        neighbors.append([np.mod(i-1, d1), np.mod(j+1, d2), 0])
    return neighbors

def indices_periodic_BC_NNN(i, j, u, d1, d2):
    neighbors = []
    neighbors.append([np.mod(i+1, d1), j, u])
    neighbors.append([i, np.mod(j+1, d2), u])
    neighbors.append([np.mod(i+1, d1), np.mod(j-1, d2), u])
    neighbors.append([np.mod(i-1, d1), j, u])
    neighbors.append([i, np.mod(j-1, d2), u])
    neighbors.append([np.mod(i-1, d1), np.mod(j+1, d2), u])
    return neighbors

def indices_periodic_BC_NNNN(i, j, u, d1, d2):
    neighbors = []
    if u == 0:
        neighbors.append([np.mod(i+1, d1), j, 1])
        neighbors.append([np.mod(i-1, d1), j, 1])
        neighbors.append([np.mod(i+1, d1), np.mod(j-2, d2), 1])
    else:
        neighbors.append([np.mod(i-1, d1), j, 0])
        neighbors.append([np.mod(i+1, d1), j, 0])
        neighbors.append([np.mod(i-1, d1), np.mod(j+2, d2), 0])
    return neighbors

def KitaevNN(J_values, indx1, indx2, bond_type, max_site):
    J1xy, J1z, J3xy, J3z, D, E, F, G = J_values
    J1z_ = np.array([[J1xy+D, E, F],
                     [E, J1xy-D, G],
                     [F, G, J1z]])
    U_2pi_3 = np.array([[np.cos(2*np.pi/3), -np.sin(2*np.pi/3), 0],
                        [np.sin(2*np.pi/3), np.cos(2*np.pi/3), 0],
                       [0, 0, 1]])
    J1x_ = U_2pi_3 @ J1z_ @ U_2pi_3.T
    J1y_ = U_2pi_3.T @ J1z_ @ U_2pi_3
    if indx1 < max_site and indx2 < max_site and indx1 >= 0 and indx2 >= 0:
        J_matrix = np.zeros((3, 3))
        if bond_type == 'x':
            J_matrix = J1x_
        elif bond_type == 'y':
            J_matrix = J1y_
        elif bond_type == 'z':
            J_matrix = J1z_
        Jxx = J_matrix[0, 0]
        Jxy = J_matrix[0, 1]
        Jxz = J_matrix[0, 2]
        Jyx = J_matrix[1, 0]
        Jyy = J_matrix[1, 1]
        Jyz = J_matrix[1, 2]
        Jzx = J_matrix[2, 0]
        Jzy = J_matrix[2, 1]
        Jzz = J_matrix[2, 2]
        return np.array([
            [2, indx1, 2, indx2, Jzz, 0],
            [0, indx1, 0, indx2, (Jxx-Jyy)/4, -Jxy/2],
            [1, indx1, 1, indx2, (Jxx-Jyy)/4, Jxy/2],
            [0, indx1, 1, indx2, (Jxx+Jyy)/4, Jyx/2],
            [1, indx1, 0, indx2, (Jxx+Jyy)/4, -Jyx/2],
            [0, indx1, 2, indx2, Jxz/2, -Jyz/2],
            [1, indx1, 2, indx2, Jxz/2, Jyz/2],
            [2, indx1, 0, indx2, Jzx/2, -Jzy/2],
            [2, indx1, 1, indx2, Jzx/2, Jzy/2],
        ])
    return np.array([])

def J2NNN(J_values, indx1, indx2, bond_type, max_site):
    J1xy, J1z, J3xy, J3z, D, E, F, G = J_values
    if indx1 < max_site and indx2 < max_site and indx1 >= 0 and indx2 >= 0:
        return np.array([[1, indx1, 0, indx2, D/2, 0],[0, indx1, 1, indx2, D/2, 0]])
    return np.array([])

def J3NNNN(J_values, indx1, indx2, bond_type, max_site):
    J1xy, J1z, J3xy, J3z, D, E, F, G = J_values
    if indx1 < max_site and indx2 < max_site and indx1 >= 0 and indx2 >= 0:
        return np.array([[2, indx1, 2, indx2, J3z, 0],
                    [1, indx1, 0, indx2, J3xy/2, 0],
                    [0, indx1, 1, indx2, J3xy/2, 0]])
    return np.array([])

def Zeeman(h, hx, hy, hz):
    h_dir = np.array([hx, hy, hz])
    if np.linalg.norm(h_dir) > 0:
        h_dir = h_dir / np.linalg.norm(h_dir)
    hx, hy, hz = h * h_dir
    return np.array([
        [0, 0, -hx/2, hy/2],
        [1, 0, -hx/2, -hy/2],
        [2, 0, -hz, 0]
    ])

def write_interALL(interALL, output_dir, file_name):
    num_param = len(interALL)
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(num_param):
            f.write(f" {int(interALL[i,0]):8d} " \
                  + f" {int(interALL[i,1]):8d}   " \
                  + f" {int(interALL[i,2]):8d}   " \
                  + f" {int(interALL[i,3]):8d}   " \
                  + f" {interALL[i,4]:8f}   " \
                  + f" {interALL[i,5]:8f}   " \
                  + "\n")

def write_transfer(transfer, output_dir, file_name):
    num_param = len(transfer)
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(num_param):
            f.write(f" {int(transfer[i,0]):8d} " \
                  + f" {int(transfer[i,1]):8d}   " \
                  + f" {transfer[i,2]:8f}   " \
                  + f" {transfer[i,3]:8f}" \
                  + "\n")

def write_site_positions(output_dir, dim1, dim2):
    with open(output_dir+"site_positions.dat", 'wt') as f:
        f.write("# index, x, y\n")
        for i in range(dim1):
            for j in range(dim2):
                for u in range(2):
                    site_index = i*dim2*2 + j*2 + u
                    unit_cell_pos = i*basis[0] + j*basis[1]
                    position = unit_cell_pos + site_basis[u]
                    f.write(f"{site_index} {position[0]:.6f} {position[1]:.6f}\n")

def write_lattice_parameters(output_dir):
    with open(output_dir+"lattice_parameters.dat", 'wt') as f:
        f.write("# Honeycomb lattice parameters\n\n")
        f.write("# Unit cell site basis (2 sites per unit cell)\n")
        f.write("# site_index, x, y\n")
        for i, site in enumerate(site_basis):
            f.write(f"{i} {site[0]:.6f} {site[1]:.6f}\n")
        f.write("\n")
        f.write("# Lattice vectors\n")
        f.write("# vector_index, x, y\n")
        for i, vector in enumerate(basis):
            f.write(f"{i} {vector[0]:.6f} {vector[1]:.6f}\n")

def lattice_pos(site_indx, dim1, dim2):
    i  = site_indx // (dim2 * 2)
    j = (site_indx // 2) % dim2
    u = site_indx % 2
    position = i * basis[0] + j * basis[1] + site_basis[u]
    return position

def plot_honeycomb_lattice(output_dir, dim1, dim2, use_pbc, NN_list, bond_list):
    positions = []
    sublattice_indices = []
    for i in range(dim1):
        for j in range(dim2):
            for u in range(2):
                unit_cell_pos = i*basis[0] + j*basis[1]
                position = unit_cell_pos + site_basis[u]
                positions.append(position)
                sublattice_indices.append(u)
    positions = np.array(positions)
    sublattice_indices = np.array(sublattice_indices)
    plt.figure(figsize=(10, 8))
    colors = ['r', 'b']
    for u in range(2):
        mask = sublattice_indices == u
        plt.scatter(
            positions[mask, 0], 
            positions[mask, 1],
            s=80,
            c=colors[u],
            marker='o',
            label=f'Sublattice {u}'
        )
    bond_colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    for site_idx in range(len(NN_list)):
        site_pos = positions[site_idx]
        for idx, neighbor_idx in enumerate(NN_list[site_idx]):
            if idx < len(bond_list[site_idx]):
                bond_type = bond_list[site_idx][idx]
                neighbor_pos = positions[neighbor_idx]
                if site_idx < neighbor_idx:
                    plt.plot(
                        [site_pos[0], neighbor_pos[0]],
                        [site_pos[1], neighbor_pos[1]],
                        color=bond_colors[bond_type],
                        alpha=0.7,
                        linewidth=2,
                        label=f'{bond_type}-bond' if (site_idx == 0 and idx == 0) else ""
                    )
    plt.xlabel('X')
    plt.ylabel('Y')
    bc_type = "PBC" if use_pbc else "Open BC"
    plt.title(f'Honeycomb Lattice ({dim1}x{dim2} unit cells, {bc_type})')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.axis('equal')
    plt.savefig(f"{output_dir}honeycomb_lattice.png", dpi=300, bbox_inches='tight')
    plt.close()

def one_body_correlations(file_name, All_N, output_dir):
    num_green_one = All_N
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_one:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(3):
            for all_i in range(0, All_N):
                f.write(f" {i:8d} {all_i:8d}   1.000000   0.000000\n")

def two_body_correlations(file_name, All_N, output_dir):
    num_green_two = All_N*All_N
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_two:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for i in range(3):
            for j in range(3):
                for all_i in range(0, All_N):
                    for all_j in range(0, All_N):
                        f.write(f" {i:8d} {all_i:8d}   {j:8d}   {all_j:8d}   1.000000   0.000000\n")

def spin_operators(Op, Q, file_name, All_N, dim1, dim2, output_dir):
    num_green_one = All_N
    with open(output_dir+file_name, 'wt') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_one:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for all_i in range(0, All_N):
            pos = lattice_pos(all_i, dim1, dim2)
            factor = np.exp(1j*Q[0]*pos[0]+1j*Q[1]*pos[1])
            f.write(f" {Op:8d} {all_i:8d}   {np.real(factor):8f}   {np.imag(factor):8f}\n")

def main():
    args = get_config_or_args()
    global dim1, dim2
    dim1, dim2 = args.dim1, args.dim2
    use_pbc = args.pbc == 1
    J1xy = args.J1xy
    J1z = args.J1z
    J3xy = args.J3xy
    J3z = args.J3z
    D = args.D
    E = args.E
    F = args.F
    G = args.G
    J_values = [J1xy, J1z, J3xy, J3z, D, E, F, G]
    h = args.h
    hx, hy, hz = args.hx, args.hy, args.hz
    field_dir = [hx, hy, hz]
    output_dir = args.outdir + "/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    max_site = dim1*dim2*2
    NN_list = np.zeros((max_site, 3), dtype=int)
    bond_list = np.zeros((max_site, 3), dtype=str)
    for i in range(dim1):
        for j in range(dim2):
            for u in range(2):
                site_idx = i*dim2*2 + j*2 + u
                if use_pbc:
                    neighbor_indices = indices_periodic_BC(i, j, u, dim1, dim2)
                else:
                    neighbor_indices = indices_open_BC(i, j, u, dim1, dim2)
                nn_flat = flattenIndex(neighbor_indices, dim2)
                bond_types_for_site = ['z', 'x', 'y'][:len(nn_flat)]
                NN_list[site_idx] = nn_flat
                bond_list[site_idx] = bond_types_for_site
    NNN_list = np.zeros((max_site, 6), dtype=int)
    for i in range(dim1):
        for j in range(dim2):
            for u in range(2):
                site_idx = i*dim2*2 + j*2 + u
                if use_pbc:
                    neighbor_indices = indices_periodic_BC_NNN(i, j, u, dim1, dim2)
                else:
                    neighbor_indices = indices_open_BC_NNN(i, j, u, dim1, dim2)
                nn_flat = flattenIndex(neighbor_indices, dim2)
                NNN_list[site_idx] = nn_flat
    NNNN_list = np.zeros((max_site, 3), dtype=int)
    for i in range(dim1):
        for j in range(dim2):
            for u in range(2):
                site_idx = i*dim2*2 + j*2 + u
                if use_pbc:
                    neighbor_indices = indices_periodic_BC_NNNN(i, j, u, dim1, dim2)
                else:
                    neighbor_indices = indices_open_BC_NNNN(i, j, u, dim1, dim2)
                nn_flat = flattenIndex(neighbor_indices, dim2)
                NNNN_list[site_idx] = nn_flat
    interALL = []
    transfer = []
    for site_idx in range(max_site):
        # NN interactions
        for j, neighbor_idx in enumerate(NN_list[site_idx]):
            if j < len(bond_list[site_idx]):
                bond_type = bond_list[site_idx][j]
                if site_idx < neighbor_idx:
                    term = KitaevNN(J_values, site_idx, neighbor_idx, bond_type, max_site)
                    if term.size > 0:
                        interALL.append(term)
        # NNN interactions
        for neighbor_idx in NNN_list[site_idx]:
            if site_idx < neighbor_idx:
                term = J2NNN(J_values, site_idx, neighbor_idx, "nnn", max_site)
                if term.size > 0:
                    interALL.append(term)
        # NNNN interactions
        for neighbor_idx in NNNN_list[site_idx]:
            if site_idx < neighbor_idx:
                term = J3NNNN(J_values, site_idx, neighbor_idx, "nnnn", max_site)
                if term.size > 0:
                    interALL.append(term)
        # Zeeman terms
        zeeman_term = Zeeman(h, *field_dir)
        for term in zeeman_term:
            transfer.append(np.array([[term[0], site_idx, term[2], term[3]]]))
    interALL = np.vstack([arr for arr in interALL if arr.size > 0]) if interALL else np.array([])
    transfer = np.vstack([arr for arr in transfer if arr.size > 0]) if transfer else np.array([])
    fstrength = np.zeros((1,1))
    fstrength[0,0] = h
    np.savetxt(output_dir+"field_strength.dat", fstrength)
    with open(output_dir+"boundary_condition.dat", 'w') as f:
        f.write(f"PBC: {'True' if use_pbc else 'False'}\n")
        f.write(f"Dimensions: {dim1}x{dim2}\n")
    write_interALL(interALL, output_dir, "InterAll.dat")
    write_transfer(transfer, output_dir, "Trans.dat")
    write_site_positions(output_dir, dim1, dim2)
    write_lattice_parameters(output_dir)
    plot_honeycomb_lattice(output_dir, dim1, dim2, use_pbc, NN_list, bond_list)
    one_body_correlations("one_body_correlations.dat", max_site, output_dir)
    two_body_correlations("two_body_correlations.dat", max_site, output_dir)
    spin_operators(0, [0, 0], "observables_S+_Gamma.dat", max_site, dim1, dim2, output_dir)
    spin_operators(1, [0, 0], "observables_S-_Gamma.dat", max_site, dim1, dim2, output_dir)
    spin_operators(2, [0, 0], "observables_Sz_Gamma.dat", max_site, dim1, dim2, output_dir)
    spin_operators(0, [np.pi, 0], "observables_S+_M.dat", max_site, dim1, dim2, output_dir)
    spin_operators(1, [np.pi, 0], "observables_S-_M.dat", max_site, dim1, dim2, output_dir)
    spin_operators(2, [np.pi, 0], "observables_Sz_M.dat", max_site, dim1, dim2, output_dir)
    print(f"Generated honeycomb lattice Kitaev Hamiltonian with dimensions {dim1}x{dim2}")
    print(f"Boundary conditions: {'Periodic' if use_pbc else 'Open'}")
    print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    main()
