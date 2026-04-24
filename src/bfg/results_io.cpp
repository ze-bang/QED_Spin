// =============================================================================
// src/bfg/results_io.cpp
//
// HDF5 results-writers for the BFG order-parameter pipeline (P2.1 results-IO
// slice). Lifted byte-for-byte out of `compute_bfg_order_parameters.cpp`:
// every dataset name, attribute name, complex compound member name (`r`/`i`),
// orientation index map, and stdout transcript matches the historical CPU
// driver so existing Python plotting scripts and post-processing notebooks
// continue to read the files identically.
//
// The only intentional structural delta vs. the original code is namespacing:
// the writers and the structs they consume now live in `namespace ed::bfg`.
// =============================================================================

#include "ed/bfg/results_io.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <H5Cpp.h>

namespace ed::bfg {

void save_results(
    const std::string& filename,
    const StructureFactorResult& sf,
    const NematicResult& nem,
    const NematicResult& nem_spsm,
    const NematicResult& nem_szsz,
    const NematicResult& nem_heisenberg,
    const VBSResult& vbs,
    const PlaquetteResult& plaq,
    const Cluster& cluster,
    const Sq2DGridResult& s_q_2d,
    int n_q_grid,
    const std::map<std::pair<int, int>, Complex>& spsm_bonds,
    const std::map<std::pair<int, int>, double>& szsz_bonds,
    const std::map<std::pair<int, int>, double>& heisenberg_bonds
) {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);

        H5::CompType complex_type(sizeof(Complex));
        complex_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);

        // S(q) at k-points (full Heisenberg).
        {
            hsize_t dims[1] = {sf.s_q.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_q", complex_type, dataspace);
            dataset.write(sf.s_q.data(), complex_type);
        }

        if (!sf.s_q_smsp.empty()) {
            hsize_t dims[1] = {sf.s_q_smsp.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_q_smsp", complex_type, dataspace);
            dataset.write(sf.s_q_smsp.data(), complex_type);
        }

        if (!sf.s_q_szsz.empty()) {
            hsize_t dims[1] = {sf.s_q_szsz.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_q_szsz", complex_type, dataspace);
            dataset.write(sf.s_q_szsz.data(), complex_type);
        }

        // 2D q-grid datasets.
        if (!s_q_2d.s_q_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(n_q_grid),
                               static_cast<hsize_t>(n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("S_q_2d", complex_type, dataspace);
            std::vector<Complex> flat(n_q_grid * n_q_grid);
            for (int i = 0; i < n_q_grid; ++i) {
                for (int j = 0; j < n_q_grid; ++j) {
                    flat[i * n_q_grid + j] = s_q_2d.s_q_2d[i][j];
                }
            }
            dataset.write(flat.data(), complex_type);
        }

        if (!s_q_2d.s_q_smsp_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(n_q_grid),
                               static_cast<hsize_t>(n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("S_q_smsp_2d", complex_type, dataspace);
            std::vector<Complex> flat(n_q_grid * n_q_grid);
            for (int i = 0; i < n_q_grid; ++i) {
                for (int j = 0; j < n_q_grid; ++j) {
                    flat[i * n_q_grid + j] = s_q_2d.s_q_smsp_2d[i][j];
                }
            }
            dataset.write(flat.data(), complex_type);
        }

        if (!s_q_2d.s_q_szsz_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(n_q_grid),
                               static_cast<hsize_t>(n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet(
                "S_q_szsz_2d", H5::PredType::NATIVE_DOUBLE, dataspace);
            std::vector<double> flat(n_q_grid * n_q_grid);
            for (int i = 0; i < n_q_grid; ++i) {
                for (int j = 0; j < n_q_grid; ++j) {
                    flat[i * n_q_grid + j] = s_q_2d.s_q_szsz_2d[i][j];
                }
            }
            dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
        }

        // VBS S_D(q) at k-points.
        if (!vbs.S_d_xy.empty()) {
            hsize_t dims[1] = {vbs.S_d_xy.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q_xy", complex_type, dataspace);
            dataset.write(vbs.S_d_xy.data(), complex_type);
        }

        if (!vbs.S_d_heis.empty()) {
            hsize_t dims[1] = {vbs.S_d_heis.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(
                "S_D_q_heis", H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(vbs.S_d_heis.data(), H5::PredType::NATIVE_DOUBLE);
        }

        if (!vbs.S_d_xy_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(vbs.n_q_grid),
                               static_cast<hsize_t>(vbs.n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q_xy_2d", complex_type, dataspace);
            std::vector<Complex> flat(vbs.n_q_grid * vbs.n_q_grid);
            for (int i = 0; i < vbs.n_q_grid; ++i) {
                for (int j = 0; j < vbs.n_q_grid; ++j) {
                    flat[i * vbs.n_q_grid + j] = vbs.S_d_xy_2d[i][j];
                }
            }
            dataset.write(flat.data(), complex_type);
        }

        if (!vbs.S_d_heis_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(vbs.n_q_grid),
                               static_cast<hsize_t>(vbs.n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet(
                "S_D_q_heis_2d", H5::PredType::NATIVE_DOUBLE, dataspace);
            std::vector<double> flat(vbs.n_q_grid * vbs.n_q_grid);
            for (int i = 0; i < vbs.n_q_grid; ++i) {
                for (int j = 0; j < vbs.n_q_grid; ++j) {
                    flat[i * vbs.n_q_grid + j] = vbs.S_d_heis_2d[i][j];
                }
            }
            dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
        }

        // Orientation-resolved S_D^{αβ}(q) (6 unique components per k-point).
        if (!vbs.S_d_xy_oriented.empty()) {
            int n_k = static_cast<int>(vbs.S_d_xy_oriented.size());

            {
                std::vector<Complex> flat_data(n_k * 6);
                for (int ik = 0; ik < n_k; ++ik) {
                    for (int c = 0; c < 6; ++c) {
                        flat_data[ik * 6 + c] = vbs.S_d_xy_oriented[ik][c];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(n_k), 6};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet(
                    "S_D_q_xy_oriented", complex_type, dataspace);
                dataset.write(flat_data.data(), complex_type);
            }

            {
                std::vector<double> flat_data(n_k * 6);
                for (int ik = 0; ik < n_k; ++ik) {
                    for (int c = 0; c < 6; ++c) {
                        flat_data[ik * 6 + c] = vbs.S_d_heis_oriented[ik][c];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(n_k), 6};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet(
                    "S_D_q_heis_oriented", H5::PredType::NATIVE_DOUBLE, dataspace);
                dataset.write(flat_data.data(), H5::PredType::NATIVE_DOUBLE);
            }

            {
                hsize_t dims[1] = {3};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = file.createDataSet(
                    "n_bonds_per_orientation", H5::PredType::NATIVE_INT, dataspace);
                dataset.write(vbs.n_bonds_per_orientation.data(),
                              H5::PredType::NATIVE_INT);
            }

            std::cout << "  Orientation-resolved S_D^{αβ}(q) saved: "
                      << n_k << " x 6" << std::endl;
        }

        // Bond-resolved dimer correlation matrices.
        if (!vbs.dimer_corr_xy.empty()) {
            int nb = vbs.n_bonds;

            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet(
                    "dimer_corr_xy", complex_type, dataspace);
                std::vector<Complex> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.dimer_corr_xy[i][j];
                    }
                }
                dataset.write(flat.data(), complex_type);
            }

            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet(
                    "connected_corr_xy", complex_type, dataspace);
                std::vector<Complex> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.connected_corr_xy[i][j];
                    }
                }
                dataset.write(flat.data(), complex_type);
            }

            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet(
                    "dimer_corr_heis", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.dimer_corr_heis[i][j];
                    }
                }
                dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
            }

            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet(
                    "connected_corr_heis", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.connected_corr_heis[i][j];
                    }
                }
                dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
            }

            std::cout << "Saved bond-resolved dimer correlations: "
                      << nb << " x " << nb << " matrices" << std::endl;
        }

        // k-points used by S(q) / S_D(q).
        {
            hsize_t dims[2] = {cluster.k_points.size(), 2};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet(
                "k_points", H5::PredType::NATIVE_DOUBLE, dataspace);
            std::vector<double> k_flat(cluster.k_points.size() * 2);
            for (size_t i = 0; i < cluster.k_points.size(); ++i) {
                k_flat[2 * i] = cluster.k_points[i][0];
                k_flat[2 * i + 1] = cluster.k_points[i][1];
            }
            dataset.write(k_flat.data(), H5::PredType::NATIVE_DOUBLE);
        }

        // 2D q-grid abscissa.
        {
            std::vector<double> q_vals(n_q_grid);
            for (int i = 0; i < n_q_grid; ++i) {
                q_vals[i] = -1.0 + 2.0 * i / (n_q_grid - 1);
            }
            hsize_t dims[1] = {static_cast<hsize_t>(n_q_grid)};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(
                "q_grid_vals", H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(q_vals.data(), H5::PredType::NATIVE_DOUBLE);
        }

        // Scalar attributes.
        H5::Group root = file.openGroup("/");
        {
            H5::DataSpace scalar(H5S_SCALAR);

            auto write_scalar = [&](const std::string& name, double val) {
                H5::Attribute attr =
                    root.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar);
                attr.write(H5::PredType::NATIVE_DOUBLE, &val);
            };

            auto write_int = [&](const std::string& name, int val) {
                H5::Attribute attr =
                    root.createAttribute(name, H5::PredType::NATIVE_INT, scalar);
                attr.write(H5::PredType::NATIVE_INT, &val);
            };

            write_scalar("m_translation", sf.m_translation);
            write_scalar("s_q_max", std::abs(sf.s_q_max));
            write_int("q_max_idx", sf.q_max_idx);
            write_scalar("q_max_x", sf.q_max[0]);
            write_scalar("q_max_y", sf.q_max[1]);

            write_scalar("m_nematic", nem.m_nem);
            write_scalar("nematic_anisotropy", nem.anisotropy);

            write_scalar("m_nematic_spsm", nem_spsm.m_nem);
            write_scalar("nematic_anisotropy_spsm", nem_spsm.anisotropy);
            write_scalar("m_nematic_szsz", nem_szsz.m_nem);
            write_scalar("nematic_anisotropy_szsz", nem_szsz.anisotropy);
            write_scalar("m_nematic_heisenberg", nem_heisenberg.m_nem);
            write_scalar("nematic_anisotropy_heisenberg", nem_heisenberg.anisotropy);

            write_scalar("m_vbs_xy", vbs.m_vbs_xy);
            write_scalar("D_mean_xy", vbs.D_mean_xy);
            write_scalar("s_d_max_xy", std::abs(vbs.s_d_max_xy));
            write_int("vbs_q_max_idx_xy", vbs.q_max_idx_xy);
            write_scalar("vbs_q_max_x_xy", vbs.q_max_xy[0]);
            write_scalar("vbs_q_max_y_xy", vbs.q_max_xy[1]);

            write_scalar("m_vbs_heis", vbs.m_vbs_heis);
            write_scalar("D_mean_heis", vbs.D_mean_heis);
            write_scalar("s_d_max_heis", std::abs(vbs.s_d_max_heis));
            write_int("vbs_q_max_idx_heis", vbs.q_max_idx_heis);
            write_scalar("vbs_q_max_x_heis", vbs.q_max_heis[0]);
            write_scalar("vbs_q_max_y_heis", vbs.q_max_heis[1]);

            write_scalar("m_vbs", vbs.m_vbs_xy);
            write_scalar("D_mean", vbs.D_mean_xy);

            write_int("n_sites", cluster.n_sites);
            write_int("n_bonds", static_cast<int>(cluster.edges_nn.size()));
            write_int("n_q_grid", n_q_grid);
        }

        // /bonds group: positions, edges, per-bond expectations.
        if (!spsm_bonds.empty()) {
            H5::Group bonds_group = file.createGroup("/bonds");

            {
                hsize_t dims[2] = {static_cast<hsize_t>(cluster.n_sites), 2};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = bonds_group.createDataSet(
                    "positions", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> pos_flat(cluster.n_sites * 2);
                for (int i = 0; i < cluster.n_sites; ++i) {
                    pos_flat[2 * i] = cluster.positions[i][0];
                    pos_flat[2 * i + 1] = cluster.positions[i][1];
                }
                dataset.write(pos_flat.data(), H5::PredType::NATIVE_DOUBLE);
            }

            int n_bonds = static_cast<int>(cluster.edges_nn.size());
            {
                hsize_t dims[2] = {static_cast<hsize_t>(n_bonds), 2};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = bonds_group.createDataSet(
                    "edges", H5::PredType::NATIVE_INT, dataspace);
                std::vector<int> edge_flat(n_bonds * 2);
                int e = 0;
                for (const auto& [i, j] : cluster.edges_nn) {
                    edge_flat[2 * e] = i;
                    edge_flat[2 * e + 1] = j;
                    e++;
                }
                dataset.write(edge_flat.data(), H5::PredType::NATIVE_INT);
            }

            {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet(
                    "spsm", complex_type, dataspace);
                std::vector<Complex> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = spsm_bonds.at(edge);
                }
                dataset.write(vals.data(), complex_type);
            }

            if (!szsz_bonds.empty()) {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet(
                    "szsz", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = szsz_bonds.at(edge);
                }
                dataset.write(vals.data(), H5::PredType::NATIVE_DOUBLE);
            }

            if (!heisenberg_bonds.empty()) {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet(
                    "heisenberg", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = heisenberg_bonds.at(edge);
                }
                dataset.write(vals.data(), H5::PredType::NATIVE_DOUBLE);
            }

            {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet(
                    "orientation", H5::PredType::NATIVE_INT, dataspace);
                std::vector<int> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = cluster.bond_orientation.at(edge);
                }
                dataset.write(vals.data(), H5::PredType::NATIVE_INT);
            }

            // Bond-statistics summary line (preserved from CPU-driver behaviour).
            double spsm_sum = 0, szsz_sum = 0, heis_sum = 0;
            for (const auto& edge : cluster.edges_nn) {
                spsm_sum += std::real(spsm_bonds.at(edge));
                if (!szsz_bonds.empty()) szsz_sum += szsz_bonds.at(edge);
                if (!heisenberg_bonds.empty()) heis_sum += heisenberg_bonds.at(edge);
            }
            std::cout << "Bond expectations saved:" << std::endl;
            std::cout << "  <S+S->_avg = " << spsm_sum / n_bonds << std::endl;
            std::cout << "  <SzSz>_avg = " << szsz_sum / n_bonds << std::endl;
            std::cout << "  <S·S>_avg  = " << heis_sum / n_bonds << std::endl;
        }

        // /plaquette group: bowtie resonance + triangle chiral.
        if (plaq.n_plaquettes > 0) {
            H5::Group plaq_group = file.createGroup("/plaquette");

            {
                hsize_t dims[1] = {1};
                H5::DataSpace dataspace(1, dims);

                H5::DataSet ds = plaq_group.createDataSet(
                    "m_plaquette", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.m_plaquette, H5::PredType::NATIVE_DOUBLE);

                ds = plaq_group.createDataSet(
                    "P_mean", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.P_mean, H5::PredType::NATIVE_DOUBLE);

                ds = plaq_group.createDataSet(
                    "resonance_strength", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.resonance_strength, H5::PredType::NATIVE_DOUBLE);

                ds = plaq_group.createDataSet(
                    "chi_mean", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.chi_mean, H5::PredType::NATIVE_DOUBLE);

                ds = plaq_group.createDataSet(
                    "n_plaquettes", H5::PredType::NATIVE_INT, dataspace);
                ds.write(&plaq.n_plaquettes, H5::PredType::NATIVE_INT);

                ds = plaq_group.createDataSet(
                    "n_triangles", H5::PredType::NATIVE_INT, dataspace);
                ds.write(&plaq.n_triangles, H5::PredType::NATIVE_INT);
            }

            {
                hsize_t dims[1] = {2};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "q_max", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(plaq.q_max.data(), H5::PredType::NATIVE_DOUBLE);
            }

            if (!plaq.S_p.empty()) {
                hsize_t dims[1] = {plaq.S_p.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "S_p", complex_type, dataspace);
                ds.write(plaq.S_p.data(), complex_type);
            }

            if (!plaq.S_p_2d.empty()) {
                int grid_size = static_cast<int>(plaq.S_p_2d.size());
                std::vector<Complex> flat_data(grid_size * grid_size);
                for (int i = 0; i < grid_size; ++i) {
                    for (int j = 0; j < grid_size; ++j) {
                        flat_data[i * grid_size + j] = plaq.S_p_2d[i][j];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(grid_size),
                                   static_cast<hsize_t>(grid_size)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "S_p_2d", complex_type, dataspace);
                ds.write(flat_data.data(), complex_type);
            }

            if (!plaq.S_p_oriented.empty()) {
                int n_k = static_cast<int>(plaq.S_p_oriented.size());
                std::vector<Complex> flat_data(n_k * 6);
                for (int ik = 0; ik < n_k; ++ik) {
                    for (int c = 0; c < 6; ++c) {
                        flat_data[ik * 6 + c] = plaq.S_p_oriented[ik][c];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(n_k), 6};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "S_p_oriented", complex_type, dataspace);
                ds.write(flat_data.data(), complex_type);

                hsize_t dims3[1] = {3};
                H5::DataSpace ds3(1, dims3);
                H5::DataSet orient_ds = plaq_group.createDataSet(
                    "n_plaquettes_per_orientation",
                    H5::PredType::NATIVE_INT, ds3);
                orient_ds.write(plaq.n_plaquettes_per_orientation.data(),
                                H5::PredType::NATIVE_INT);

                std::cout << "  Orientation-resolved S_P^{αβ}(q) saved: "
                          << n_k << " x 6" << std::endl;
            }

            if (!plaq.orientations.empty()) {
                hsize_t dims[1] = {plaq.orientations.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "orientations", H5::PredType::NATIVE_INT, dataspace);
                ds.write(plaq.orientations.data(), H5::PredType::NATIVE_INT);
            }

            if (!plaq.P_r.empty()) {
                hsize_t dims[1] = {plaq.P_r.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "P_r", complex_type, dataspace);
                ds.write(plaq.P_r.data(), complex_type);
            }

            if (!plaq.centers.empty()) {
                std::vector<double> flat_centers(plaq.centers.size() * 2);
                for (size_t i = 0; i < plaq.centers.size(); ++i) {
                    flat_centers[2 * i] = plaq.centers[i][0];
                    flat_centers[2 * i + 1] = plaq.centers[i][1];
                }
                hsize_t dims[2] = {plaq.centers.size(), 2};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "centers", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(flat_centers.data(), H5::PredType::NATIVE_DOUBLE);
            }

            if (!plaq.chi_r.empty()) {
                hsize_t dims[1] = {plaq.chi_r.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet(
                    "chi_r", complex_type, dataspace);
                ds.write(plaq.chi_r.data(), complex_type);
            }

            std::cout << "Plaquette data saved:" << std::endl;
            std::cout << "  n_plaquettes = " << plaq.n_plaquettes << std::endl;
            std::cout << "  n_triangles = " << plaq.n_triangles << std::endl;
            std::cout << "  m_plaquette = " << plaq.m_plaquette << std::endl;
        }

        std::cout << "Results saved to: " << filename << std::endl;

    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 write error: " + std::string(e.getCDetailMsg()));
    }
}

void save_temperature_scan_results(
    const std::vector<OrderParameterResults>& results,
    const std::string& output_file,
    double jpm
) {
    if (results.empty()) return;

    try {
        H5::H5File file(output_file, H5F_ACC_TRUNC);

        const size_t n = results.size();

        std::vector<double> temperatures(n), m_trans(n), m_nem(n), m_nem_spsm(n);
        std::vector<double> m_nem_szsz(n), m_nem_heis(n), aniso(n);
        std::vector<double> m_vbs_vals(n), m_vbs_xy_vals(n), m_vbs_heis_vals(n);
        std::vector<double> D_mean_vals(n), D_mean_xy_vals(n), D_mean_heis_vals(n);
        std::vector<double> m_plaq(n), P_mean_vals(n), res_strength(n), chi_mean_vals(n);

        for (size_t i = 0; i < n; ++i) {
            temperatures[i]     = results[i].temperature;
            m_trans[i]          = results[i].m_translation;
            m_nem[i]            = results[i].m_nematic;
            m_nem_spsm[i]       = results[i].m_nematic_spsm;
            m_nem_szsz[i]       = results[i].m_nematic_szsz;
            m_nem_heis[i]       = results[i].m_nematic_heisenberg;
            aniso[i]            = results[i].anisotropy;
            m_vbs_vals[i]       = results[i].m_vbs;
            m_vbs_xy_vals[i]    = results[i].m_vbs_xy;
            m_vbs_heis_vals[i]  = results[i].m_vbs_heis;
            D_mean_vals[i]      = results[i].D_mean;
            D_mean_xy_vals[i]   = results[i].D_mean_xy;
            D_mean_heis_vals[i] = results[i].D_mean_heis;
            m_plaq[i]           = results[i].m_plaquette;
            P_mean_vals[i]      = results[i].P_mean;
            res_strength[i]     = results[i].resonance_strength;
            chi_mean_vals[i]    = results[i].chi_mean;
        }

        auto write_dataset = [&](const std::string& name,
                                 const std::vector<double>& data) {
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(
                name, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };

        {
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute attr = file.createAttribute(
                "jpm", H5::PredType::NATIVE_DOUBLE, attr_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &jpm);
        }

        write_dataset("temperature",          temperatures);
        write_dataset("m_translation",        m_trans);
        write_dataset("m_nematic",            m_nem);
        write_dataset("m_nematic_spsm",       m_nem_spsm);
        write_dataset("m_nematic_szsz",       m_nem_szsz);
        write_dataset("m_nematic_heisenberg", m_nem_heis);
        write_dataset("anisotropy",           aniso);
        write_dataset("m_vbs",                m_vbs_vals);
        write_dataset("m_vbs_xy",             m_vbs_xy_vals);
        write_dataset("m_vbs_heis",           m_vbs_heis_vals);
        write_dataset("D_mean",               D_mean_vals);
        write_dataset("D_mean_xy",            D_mean_xy_vals);
        write_dataset("D_mean_heis",          D_mean_heis_vals);
        write_dataset("m_plaquette",          m_plaq);
        write_dataset("P_mean",               P_mean_vals);
        write_dataset("resonance_strength",   res_strength);
        write_dataset("chi_mean",             chi_mean_vals);

        std::cout << "  Saved T-scan results to: " << output_file << std::endl;

    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 write error: " + std::string(e.getCDetailMsg()));
    }
}

void save_scan_results(
    const std::vector<OrderParameterResults>& results,
    const std::string& output_file
) {
    try {
        H5::H5File file(output_file, H5F_ACC_TRUNC);

        const size_t n = results.size();
        std::vector<double> jpm_vals(n), temperature_vals(n);
        std::vector<double> m_trans(n);
        std::vector<double> m_nem(n), m_nem_spsm(n), m_nem_szsz(n), m_nem_heis(n), aniso(n);
        std::vector<double> m_vbs_vals(n), m_vbs_xy_vals(n), m_vbs_heis_vals(n);
        std::vector<double> D_mean_vals(n), D_mean_xy_vals(n), D_mean_heis_vals(n);
        std::vector<double> m_plaq(n), P_mean_vals(n), res_strength(n), chi_mean_vals(n);
        std::vector<int> n_plaq(n), n_tri(n);

        for (size_t i = 0; i < n; ++i) {
            jpm_vals[i]         = results[i].jpm;
            temperature_vals[i] = results[i].temperature;
            m_trans[i]          = results[i].m_translation;

            m_nem[i]            = results[i].m_nematic;
            m_nem_spsm[i]       = results[i].m_nematic_spsm;
            m_nem_szsz[i]       = results[i].m_nematic_szsz;
            m_nem_heis[i]       = results[i].m_nematic_heisenberg;
            aniso[i]            = results[i].anisotropy;

            m_vbs_vals[i]       = results[i].m_vbs;
            m_vbs_xy_vals[i]    = results[i].m_vbs_xy;
            m_vbs_heis_vals[i]  = results[i].m_vbs_heis;

            D_mean_vals[i]      = results[i].D_mean;
            D_mean_xy_vals[i]   = results[i].D_mean_xy;
            D_mean_heis_vals[i] = results[i].D_mean_heis;

            m_plaq[i]           = results[i].m_plaquette;
            P_mean_vals[i]      = results[i].P_mean;
            res_strength[i]     = results[i].resonance_strength;
            chi_mean_vals[i]    = results[i].chi_mean;
            n_plaq[i]           = results[i].n_plaquettes;
            n_tri[i]            = results[i].n_triangles;
        }

        auto write_dataset = [&](const std::string& name,
                                 const std::vector<double>& data) {
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(
                name, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };

        auto write_dataset_int = [&](const std::string& name,
                                     const std::vector<int>& data) {
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(
                name, H5::PredType::NATIVE_INT, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_INT);
        };

        write_dataset("jpm_values",           jpm_vals);
        write_dataset("temperature",          temperature_vals);
        write_dataset("m_translation",        m_trans);

        write_dataset("m_nematic",            m_nem);
        write_dataset("m_nematic_spsm",       m_nem_spsm);
        write_dataset("m_nematic_szsz",       m_nem_szsz);
        write_dataset("m_nematic_heisenberg", m_nem_heis);
        write_dataset("anisotropy",           aniso);

        write_dataset("m_vbs",                m_vbs_vals);
        write_dataset("m_vbs_xy",             m_vbs_xy_vals);
        write_dataset("m_vbs_heis",           m_vbs_heis_vals);
        write_dataset("D_mean",               D_mean_vals);
        write_dataset("D_mean_xy",            D_mean_xy_vals);
        write_dataset("D_mean_heis",          D_mean_heis_vals);

        write_dataset("m_plaquette",          m_plaq);
        write_dataset("P_mean",               P_mean_vals);
        write_dataset("resonance_strength",   res_strength);
        write_dataset("chi_mean",             chi_mean_vals);
        write_dataset_int("n_plaquettes",     n_plaq);
        write_dataset_int("n_triangles",      n_tri);

        std::cout << "Scan results saved to: " << output_file << std::endl;

    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 write error: " + std::string(e.getCDetailMsg()));
    }
}

}  // namespace ed::bfg
