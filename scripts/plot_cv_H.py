import os
import numpy as np
import re
from matplotlib.cm import get_cmap

import matplotlib.pyplot as plt

def extract_field_value(path):
    """Extract the numeric field value from a path containing 'h=X.Y'"""
    match = re.search(r'h=(\d+\.?\d*)', path)
    if match:
        return float(match.group(1))
    return None

def main():
    # Find all nlc_specific_heat.txt files
    files = []
    base_dir = './Pr2Zr2O7_field_scan_alternative'
    
    for root, dirs, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename == 'nlc_specific_heat.txt':
                files.append(os.path.join(root, filename))
    
    if not files:
        print(f"No nlc_specific_heat.txt files found in {base_dir}")
        return
    
    # Extract field values and sort files by field value
    files_with_field = [(f, extract_field_value(f)) for f in files]
    files_with_field = [(f, h) for f, h in files_with_field if h is not None]
    files_with_field = sorted(files_with_field, key=lambda x: x[1])
    
    plt.figure(figsize=(10, 6))
    
    # Use a colormap for different field values
    cmap = get_cmap('viridis', max(len(files_with_field), 2))
    
    for i, (file_path, field_value) in enumerate(files_with_field):
        field_str = f"h = {field_value}"
        if (field_value == np.array([4, 6, 8])).any():
            try:
                # Load the data
                data = np.loadtxt(file_path, comments='#')
                temperature = data[:, 0]
                specific_heat = data[:, 1]
                
                # Plot the data
                plt.plot(temperature, specific_heat, '-',
                        label=field_str, linewidth=2)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
    plt.xlabel('Temperature (K)')
    plt.xscale('log')
    plt.xlim(1, 20)
    plt.ylabel('Specific Heat')
    plt.title('Specific Heat vs Temperature at Different Magnetic Field Values')
    plt.legend(title='Magnetic Field')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and display
    plt.savefig('specific_heat_vs_temperature.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()