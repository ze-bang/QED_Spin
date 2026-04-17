#!/usr/bin/env python3
"""
Comprehensive visualization of the thermal Hall conductivity computation
for quantum spin ice on the 16-site pyrochlore cluster.

Produces a single multi-panel figure with:
  (a) 3D lattice + bond structure
  (b) Full spectral function Re[S_{JEx,JEy}(ω)] at 3 temperatures
  (c) Im[S_{JEx,JEy}(ω)] — antisymmetric part
  (d) Positive-ω excitation spectrum with gap annotation
  (e) Low-ω zoom for Kubo extraction
  (f) κ_xy(T) extracted from the dc limit
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os, sys


# ── Helpers ──────────────────────────────────────────────────────────
def load_positions(path):
    coords = []
    sublat = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            coords.append([float(p[3]), float(p[4]), float(p[5])])
            sublat.append(int(float(p[2])))
    return np.array(coords), np.array(sublat)


def load_hamiltonian_bonds(path):
    """Return list of (site_i, site_j) pairs from InterAll.dat."""
    bonds = set()
    with open(path) as f:
        lines = f.readlines()
    # skip headers
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('=') or line.startswith('num'):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 6:
            try:
                si, sj = int(parts[1]), int(parts[3])
                if si != sj:
                    bonds.add((min(si, sj), max(si, sj)))
            except (ValueError, IndexError):
                pass
        i += 1
    return list(bonds)


def load_hdf5_dynamical(h5_path):
    datasets = []
    with h5py.File(h5_path, 'r') as f:
        dyn = f['dynamical']
        for key in sorted(dyn.keys()):
            if 'frequencies' not in dyn[key]:
                continue
            T = dyn[key].attrs['temperature']
            datasets.append({
                'T': T, 'key': key,
                'freq': np.array(dyn[key]['frequencies']),
                'Re_S': np.array(dyn[key]['spectral_real']),
                'Im_S': np.array(dyn[key]['spectral_imag']),
                'err_Re': np.array(dyn[key]['error_real']),
                'err_Im': np.array(dyn[key]['error_imag']),
            })
    datasets.sort(key=lambda d: d['T'])
    return datasets


# ── Main ─────────────────────────────────────────────────────────────
def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(base, 'test_pyro16_qsi_kappa')
    out_dir = os.path.join(base, 'output')
    h5_path = os.path.join(out_dir, 'ed_results.h5')

    pos, sublat = load_positions(os.path.join(run_dir, 'positions.dat'))
    bonds = load_hamiltonian_bonds(os.path.join(run_dir, 'InterAll.dat'))
    datasets = load_hdf5_dynamical(h5_path)

    # ── Color palette ──
    temp_colors = ['#2166ac', '#999999', '#b2182b']  # blue, grey, red
    sublat_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
    sublat_names = ['A', 'B', 'C', 'D']

    # ── Figure layout ──
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.32,
                  left=0.06, right=0.97, top=0.94, bottom=0.06)

    # (a) 3D lattice ──────────────────────────────────────────────────
    ax_lat = fig.add_subplot(gs[0, 0], projection='3d')
    for sl in range(4):
        mask = sublat == sl
        ax_lat.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2],
                       s=80, c=sublat_colors[sl], label=sublat_names[sl],
                       edgecolors='k', linewidths=0.4, zorder=5, depthshade=False)
    # bonds
    segments = [[pos[i], pos[j]] for i, j in bonds
                if np.linalg.norm(pos[i] - pos[j]) < 0.5]  # skip PBC wraps
    lc = Line3DCollection(segments, colors='gray', linewidths=0.6, alpha=0.5)
    ax_lat.add_collection3d(lc)

    ax_lat.set_xlabel('x', fontsize=8, labelpad=-2)
    ax_lat.set_ylabel('y', fontsize=8, labelpad=-2)
    ax_lat.set_zlabel('z', fontsize=8, labelpad=-2)
    ax_lat.set_title('(a)  16-site pyrochlore cluster', fontsize=11, pad=8)
    ax_lat.legend(fontsize=7, loc='upper left', framealpha=0.7,
                  title='sublattice', title_fontsize=7)
    ax_lat.tick_params(labelsize=6)
    ax_lat.view_init(elev=22, azim=-50)

    # (b) Re[S(ω)] full range ────────────────────────────────────────
    ax_re = fig.add_subplot(gs[0, 1:])
    for d, c in zip(datasets, temp_colors):
        ax_re.plot(d['freq'], d['Re_S'], color=c, lw=1.3,
                   label=f"T = {d['T']:.3f}")
        ax_re.fill_between(d['freq'],
                           d['Re_S'] - d['err_Re'],
                           d['Re_S'] + d['err_Re'],
                           color=c, alpha=0.12)
    ax_re.axhline(0, color='k', lw=0.4)
    ax_re.axvline(0, color='k', lw=0.3, ls='--', alpha=0.4)
    ax_re.set_xlabel(r'$\omega\;/\;J_{zz}$', fontsize=11)
    ax_re.set_ylabel(r'Re[$S_{J^E_x,\,J^E_y}(\omega)$]', fontsize=11)
    ax_re.set_title(r'(b)  Energy-current cross-correlator — real part', fontsize=11)
    ax_re.legend(fontsize=9, framealpha=0.8)
    ax_re.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # (c) Im[S(ω)] ───────────────────────────────────────────────────
    ax_im = fig.add_subplot(gs[1, 0:2])
    for d, c in zip(datasets, temp_colors):
        ax_im.plot(d['freq'], d['Im_S'], color=c, lw=1.3,
                   label=f"T = {d['T']:.3f}")
        ax_im.fill_between(d['freq'],
                           d['Im_S'] - d['err_Im'],
                           d['Im_S'] + d['err_Im'],
                           color=c, alpha=0.12)
    ax_im.axhline(0, color='k', lw=0.4)
    ax_im.axvline(0, color='k', lw=0.3, ls='--', alpha=0.4)
    ax_im.set_xlabel(r'$\omega\;/\;J_{zz}$', fontsize=11)
    ax_im.set_ylabel(r'Im[$S_{J^E_x,\,J^E_y}(\omega)$]', fontsize=11)
    ax_im.set_title(r'(c)  Imaginary part (antisymmetric in $\omega$)', fontsize=11)
    ax_im.legend(fontsize=9, framealpha=0.8)
    ax_im.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # (d) Positive-ω excitation spectrum ──────────────────────────────
    ax_exc = fig.add_subplot(gs[1, 2])
    for d, c in zip(datasets, temp_colors):
        mask = d['freq'] > 0.05
        ax_exc.plot(d['freq'][mask], -d['Re_S'][mask], color=c, lw=1.5,
                    label=f"T = {d['T']:.3f}")
    ax_exc.set_yscale('log')
    ax_exc.set_xlabel(r'$\omega\;/\;J_{zz}$', fontsize=11)
    ax_exc.set_ylabel(r'$-$Re[$S(\omega)$]  (log scale)', fontsize=11)
    ax_exc.set_title('(d)  Excitation spectrum', fontsize=11)

    # Mark spectral features
    ax_exc.axvline(0.65, color='gray', ls=':', lw=1.0, alpha=0.7)
    ax_exc.text(0.67, ax_exc.get_ylim()[1]*0.3 if ax_exc.get_ylim()[1] > 0 else 1e-8,
                r'$\Delta_{\mathrm{gap}}$', fontsize=9, color='gray', va='top')
    ax_exc.legend(fontsize=8, framealpha=0.8, loc='upper right')
    ax_exc.set_xlim(0, 3.0)

    # (e) Low-ω zoom — Kubo extraction ───────────────────────────────
    ax_low = fig.add_subplot(gs[2, 0:2])
    for d, c in zip(datasets, temp_colors):
        mask = (d['freq'] > -0.6) & (d['freq'] < 0.6)
        ax_low.plot(d['freq'][mask], d['Re_S'][mask], color=c, lw=1.5,
                    label=f"T = {d['T']:.3f}")
        ax_low.fill_between(d['freq'][mask],
                            (d['Re_S'] - d['err_Re'])[mask],
                            (d['Re_S'] + d['err_Re'])[mask],
                            color=c, alpha=0.12)

    # Shade the extraction window [0.03, 0.15]
    yl = ax_low.get_ylim()
    ax_low.axvspan(0.03, 0.15, color='gold', alpha=0.15, zorder=0)
    ax_low.text(0.09, yl[0]*0.15, 'extraction\nwindow', fontsize=7,
                ha='center', va='bottom', color='#886600', style='italic')

    ax_low.axhline(0, color='k', lw=0.4)
    ax_low.axvline(0, color='k', lw=0.3, ls='--', alpha=0.4)
    ax_low.set_xlabel(r'$\omega\;/\;J_{zz}$', fontsize=11)
    ax_low.set_ylabel(r'Re[$S_{J^E_x,\,J^E_y}(\omega)$]', fontsize=11)
    ax_low.set_title(r'(e)  Low-$\omega$ region — dc transport extraction', fontsize=11)
    ax_low.legend(fontsize=9, framealpha=0.8)
    ax_low.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # (f) κ_xy(T) ────────────────────────────────────────────────────
    ax_kxy = fig.add_subplot(gs[2, 2])
    T_arr, kxy_arr, kxy_err = [], [], []

    for d in datasets:
        T = d['T']
        mask_low = (d['freq'] > 0.03) & (d['freq'] < 0.15)
        ReS_low = np.mean(d['Re_S'][mask_low])
        ReS_err = np.sqrt(np.mean(d['err_Re'][mask_low]**2)) / np.sqrt(np.sum(mask_low))
        kxy = -np.pi / T**2 * ReS_low
        kerr = np.pi / T**2 * ReS_err
        T_arr.append(T)
        kxy_arr.append(kxy)
        kxy_err.append(kerr)

    ax_kxy.errorbar(T_arr, kxy_arr, yerr=kxy_err, fmt='s-', capsize=5,
                    markersize=7, color='#b2182b', markeredgecolor='k',
                    markeredgewidth=0.5, linewidth=1.5, ecolor='#666666')
    ax_kxy.axhline(0, color='k', lw=0.4)
    ax_kxy.set_xlabel(r'$T\;/\;J_{zz}$', fontsize=11)
    ax_kxy.set_ylabel(r'$\kappa_{xy}$  [arb. units]', fontsize=11)
    ax_kxy.set_title(r'(f)  $\kappa_{xy}(T) = -\pi\,\mathrm{Re}[S(0^+)]/T^2$',
                     fontsize=11)
    ax_kxy.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # ── Global title ──
    fig.suptitle(
        r'Thermal Hall conductivity of quantum spin ice — '
        r'$J_{zz}\!=\!1,\; J_{\pm}\!=\!-0.02,\; J_{\pm\pm}\!=\!0.05,\; h_z\!=\!0.1$'
        '\n'
        r'16-site pyrochlore, FTLM (10 samples, $N_K$=200, $\eta$=0.1)',
        fontsize=13, fontweight='bold', y=0.99
    )

    out_path = os.path.join(out_dir, 'kappa_xy_summary.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f'Saved:  {out_path}')

    # ── Also print numerical table ──
    print('\n' + '='*70)
    print('  T/J_zz   |  max|Re[S]|     |  Re[S(0^+)]     |  κ_xy')
    print('-'*70)
    for d, kxy, kerr in zip(datasets, kxy_arr, kxy_err):
        T = d['T']
        mx = np.max(np.abs(d['Re_S']))
        mask_low = (d['freq'] > 0.03) & (d['freq'] < 0.15)
        rs0 = np.mean(d['Re_S'][mask_low])
        print(f'  {T:.4f}   |  {mx:.3e}     |  {rs0:+.3e}     |  {kxy:+.3e} ± {kerr:.1e}')
    print('='*70)


if __name__ == '__main__':
    main()
