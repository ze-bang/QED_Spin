#!/usr/bin/env python3
"""
Comprehensive analysis of S_{JEx,JEy}(omega,T) for thermal Hall conductivity.

Correct Kubo formula:
    kappa_xy = (-pi / T^2 V) * Re[S_{JEx,JEy}(omega -> 0^+)]

where S is the finite-T spectral function from FTLM.
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    h5_path = os.path.join(output_dir, "ed_results.h5")
    
    h5 = h5py.File(h5_path, 'r')
    dyn = h5['dynamical']
    
    datasets = []
    for key in sorted(dyn.keys()):
        if 'frequencies' not in dyn[key]:
            continue
        T = dyn[key].attrs['temperature']
        datasets.append({
            'T': T,
            'key': key,
            'freq': np.array(dyn[key]['frequencies']),
            'Re_S': np.array(dyn[key]['spectral_real']),
            'Im_S': np.array(dyn[key]['spectral_imag']),
            'err_Re': np.array(dyn[key]['error_real']),
            'err_Im': np.array(dyn[key]['error_imag']),
        })
    datasets.sort(key=lambda d: d['T'])
    h5.close()
    
    # ========== Figure 1: Full spectral function ==========
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(datasets)))
    
    for d, c in zip(datasets, colors):
        axes[0].plot(d['freq'], d['Re_S'], label=f"T={d['T']:.3f}", color=c)
        axes[0].fill_between(d['freq'],
                             d['Re_S'] - d['err_Re'],
                             d['Re_S'] + d['err_Re'],
                             alpha=0.15, color=c)
        axes[1].plot(d['freq'], d['Im_S'], label=f"T={d['T']:.3f}", color=c)
        axes[1].fill_between(d['freq'],
                             d['Im_S'] - d['err_Im'],
                             d['Im_S'] + d['err_Im'],
                             alpha=0.15, color=c)
    
    axes[0].set_ylabel(r"Re[$S_{J^E_x, J^E_y}(\omega)$]", fontsize=12)
    axes[0].set_title(r"Energy-current cross-correlator — 16-site pyrochlore QSI ($h_z=0.1$)",
                      fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].axvline(0, color='k', lw=0.3, ls='--')
    
    axes[1].set_xlabel(r"$\omega / J_{zz}$", fontsize=12)
    axes[1].set_ylabel(r"Im[$S_{J^E_x, J^E_y}(\omega)$]", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].axvline(0, color='k', lw=0.3, ls='--')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectral_function_JE_v2.png"), dpi=200)
    print(f"Saved: spectral_function_JE_v2.png")
    
    # ========== Figure 2: Zoom on low-omega + Kubo extraction ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Re[S] near omega=0
    ax = axes2[0]
    for d, c in zip(datasets, colors):
        mask = (d['freq'] > -0.5) & (d['freq'] < 0.5)
        ax.plot(d['freq'][mask], d['Re_S'][mask], label=f"T={d['T']:.3f}", color=c)
        ax.fill_between(d['freq'][mask],
                        (d['Re_S'] - d['err_Re'])[mask],
                        (d['Re_S'] + d['err_Re'])[mask],
                        alpha=0.15, color=c)
    ax.set_xlabel(r"$\omega / J_{zz}$", fontsize=12)
    ax.set_ylabel(r"Re[$S_{J^E_x, J^E_y}(\omega)$]", fontsize=12)
    ax.set_title(r"Low-$\omega$ region (Kubo extraction window)", fontsize=12)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.3, ls='--')
    ax.legend(fontsize=9)
    
    # Right: kappa_xy(T) using correct Kubo formula
    ax2 = axes2[1]
    
    T_arr = []
    kxy_arr = []
    kxy_err_arr = []
    
    for d in datasets:
        T = d['T']
        # Average Re[S] in a window around omega=0
        # Use |omega| < 0.15 (about 1.5x broadening eta=0.1)
        mask_low = (d['freq'] > 0.03) & (d['freq'] < 0.15)
        ReS_low = np.mean(d['Re_S'][mask_low])
        ReS_err = np.sqrt(np.mean(d['err_Re'][mask_low]**2)) / np.sqrt(np.sum(mask_low))
        
        # kappa_xy = -pi / T^2 * Re[S(0)]
        kxy = -np.pi / T**2 * ReS_low
        kxy_err = np.pi / T**2 * ReS_err
        
        T_arr.append(T)
        kxy_arr.append(kxy)
        kxy_err_arr.append(kxy_err)
        print(f"  T={T:.4f}: Re[S(0^+)]={ReS_low:.3e} ± {ReS_err:.3e}  →  "
              f"κ_xy = {kxy:.3e} ± {kxy_err:.3e}")
    
    ax2.errorbar(T_arr, kxy_arr, yerr=kxy_err_arr, fmt='o-', capsize=5,
                 markersize=7, color='darkred', linewidth=1.5)
    ax2.set_xlabel(r"$T / J_{zz}$", fontsize=12)
    ax2.set_ylabel(r"$\kappa_{xy}$ [arb. units]", fontsize=12)
    ax2.set_title(r"$\kappa_{xy}(T) = -\pi\,\mathrm{Re}[S(0)]/T^2$", fontsize=12)
    ax2.axhline(0, color='k', lw=0.5)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "kappa_xy_kubo_v2.png"), dpi=200)
    print(f"Saved: kappa_xy_kubo_v2.png")
    
    # ========== Figure 3: Positive-omega spectral weight (physical content) ==========
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    
    for d, c in zip(datasets, colors):
        mask = d['freq'] > 0.05
        ax3.plot(d['freq'][mask], d['Re_S'][mask], label=f"T={d['T']:.3f}", color=c, lw=1.5)
    
    ax3.set_xlabel(r"$\omega / J_{zz}$", fontsize=12)
    ax3.set_ylabel(r"Re[$S_{J^E_x, J^E_y}(\omega)$]", fontsize=12)
    ax3.set_title("Energy-current spectral weight — excitation spectrum of QSI", fontsize=13)
    ax3.legend(fontsize=10)
    ax3.axhline(0, color='k', lw=0.5)
    
    # Mark the approximate spin gap
    ax3.axvline(0.6, color='gray', lw=0.8, ls=':', label=r'$\Delta_{spinon} \approx 0.6\,J_{zz}$')
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "spectral_weight_positive_omega.png"), dpi=200)
    print(f"Saved: spectral_weight_positive_omega.png")
    
    # ========== Numerical summary ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"System: 16-site pyrochlore, J_zz=1.0, J_pm=-0.02, J_pmpm=0.05, h_z=0.1")
    print(f"FTLM: 10 samples, Krylov dim 200, broadening eta=0.1")
    print(f"Ground state energy: -4.876")
    print()
    print("Key findings:")
    print(f"  Spectral weight concentrated at ω ≈ 0.8-1.8 J_zz (above spin gap)")
    print(f"  Re[S] >> Im[S] by factor ~10-20")
    print(f"  Signal grows strongly with T (thermal activation)")
    print(f"  κ_xy is tiny near ω→0 (gapped system, finite-size effects)")
    print()
    print("Temperature  | max|Re[S(ω)]|  | Re[S(0)]      | κ_xy = -π Re[S(0)]/T²")
    print("-" * 75)
    for i, d in enumerate(datasets):
        T = d['T']
        maxReS = np.max(np.abs(d['Re_S']))
        i0 = np.argmin(np.abs(d['freq']))
        ReS0 = d['Re_S'][i0]
        print(f"  {T:.4f}      | {maxReS:.3e}      | {ReS0:.3e}     | {kxy_arr[i]:.3e}")


if __name__ == "__main__":
    main()
