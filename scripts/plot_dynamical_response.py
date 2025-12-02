#!/usr/bin/env python3
"""
Plot dynamical response (spectral function) results from Lanczos calculation.

Usage:
    python plot_dynamical_response.py <input_file> [options]
    
Example:
    python plot_dynamical_response.py output/spectral_function.txt --output plot.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def read_dynamical_response(filename):
    """
    Read dynamical response data from file.
    
    Expected format:
    # Comment lines
    # Frequency  Spectral_Function  Error
    omega1  S1  err1
    omega2  S2  err2
    ...
    """
    data = np.loadtxt(filename)
    
    if data.ndim != 2:
        raise ValueError(f"Invalid data format in {filename}")
    
    frequencies = data[:, 0]
    spectral_function = data[:, 1]
    
    if data.shape[1] >= 3:
        errors = data[:, 2]
    else:
        errors = None
    
    return frequencies, spectral_function, errors

def plot_spectral_function(frequencies, spectral_function, errors=None,
                           title="Dynamical Response", 
                           xlabel="Frequency (ω)", 
                           ylabel="Spectral Function S(ω)",
                           output_file=None,
                           show_errors=True,
                           logscale=False):
    """Plot spectral function with optional error bars."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if errors is not None and show_errors and np.any(errors > 0):
        ax.fill_between(frequencies, 
                        spectral_function - errors,
                        spectral_function + errors,
                        alpha=0.3, label='±1 std error')
    
    ax.plot(frequencies, spectral_function, 'b-', linewidth=2, label='S(ω)')
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    if logscale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_comparison(data_list, labels, title="Spectral Function Comparison",
                   output_file=None):
    """Plot multiple spectral functions for comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for i, (freq, spec, err) in enumerate(data_list):
        label = labels[i] if i < len(labels) else f"Data {i+1}"
        ax.plot(freq, spec, '-', linewidth=2, color=colors[i], label=label)
        
        if err is not None and np.any(err > 0):
            ax.fill_between(freq, spec - err, spec + err, 
                           alpha=0.2, color=colors[i])
    
    ax.set_xlabel("Frequency (ω)", fontsize=14)
    ax.set_ylabel("Spectral Function S(ω)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def print_statistics(frequencies, spectral_function):
    """Print statistical information about the spectral function."""
    
    print("\n" + "="*60)
    print("Spectral Function Statistics")
    print("="*60)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(spectral_function, prominence=0.1*np.max(spectral_function))
    
    print(f"\nFrequency range: [{frequencies[0]:.4f}, {frequencies[-1]:.4f}]")
    print(f"Number of points: {len(frequencies)}")
    print(f"\nSpectral function:")
    print(f"  Maximum: {np.max(spectral_function):.6e} at ω = {frequencies[np.argmax(spectral_function)]:.4f}")
    print(f"  Minimum: {np.min(spectral_function):.6e}")
    print(f"  Mean:    {np.mean(spectral_function):.6e}")
    
    # Integral (sum rule)
    integral = np.trapz(spectral_function, frequencies)
    print(f"\nIntegral ∫S(ω)dω = {integral:.6f}")
    
    # Peak information
    if len(peaks) > 0:
        print(f"\nNumber of prominent peaks: {len(peaks)}")
        print("\nPeak locations:")
        for i, peak_idx in enumerate(peaks[:5]):  # Show top 5 peaks
            print(f"  Peak {i+1}: ω = {frequencies[peak_idx]:.4f}, "
                  f"S(ω) = {spectral_function[peak_idx]:.6e}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Plot dynamical response (spectral function) from Lanczos calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', type=str, 
                       help='Input file containing spectral function data')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output plot file (if not specified, show interactively)')
    parser.add_argument('--title', '-t', type=str, default='Dynamical Response',
                       help='Plot title')
    parser.add_argument('--xlabel', type=str, default='Frequency (ω)',
                       help='X-axis label')
    parser.add_argument('--ylabel', type=str, default='Spectral Function S(ω)',
                       help='Y-axis label')
    parser.add_argument('--no-errors', action='store_true',
                       help='Do not plot error bars')
    parser.add_argument('--log', action='store_true',
                       help='Use logarithmic scale for y-axis')
    parser.add_argument('--stats', action='store_true',
                       help='Print statistics about the spectral function')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Additional files to compare')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for comparison plots')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Read main data
    try:
        frequencies, spectral_function, errors = read_dynamical_response(args.input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Print statistics if requested
    if args.stats:
        print_statistics(frequencies, spectral_function)
    
    # Comparison mode
    if args.compare:
        data_list = [(frequencies, spectral_function, errors)]
        labels = [args.labels[0] if args.labels else os.path.basename(args.input_file)]
        
        for i, comp_file in enumerate(args.compare):
            if not os.path.exists(comp_file):
                print(f"Warning: Comparison file '{comp_file}' not found, skipping")
                continue
            
            try:
                freq, spec, err = read_dynamical_response(comp_file)
                data_list.append((freq, spec, err))
                
                label = args.labels[i+1] if args.labels and i+1 < len(args.labels) else os.path.basename(comp_file)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Error reading '{comp_file}': {e}, skipping")
        
        plot_comparison(data_list, labels, title=args.title, output_file=args.output)
    
    # Single plot mode
    else:
        plot_spectral_function(
            frequencies, spectral_function, errors,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            output_file=args.output,
            show_errors=not args.no_errors,
            logscale=args.log
        )

if __name__ == '__main__':
    main()
