import numpy as np
import re
import matplotlib.pyplot as plt

# Load the data
dir = "/home/pc_linux/exact_diagonalization_cpp/ED_16_sites_Jpm=0.045/output/"
data_file = dir + "spin_corr_rand0.dat"

# Read header and get column names
with open(data_file, 'r') as f:
    header = f.readline()
    if header.startswith('#'):
        header = header[1:].strip()
    column_names = header.split()

# Load the data
data = np.loadtxt(data_file)

# Get inv_temp column and calculate temperature
inv_temp = data[:, 0]
temperature = 1.0 / inv_temp

# Plot szz and spm separately
plt.figure(figsize=(12, 6))

# Plot szz(real) and szz(imag)
plt.subplot(1, 2, 1)
plt.plot(temperature, data[:, 3], 'r-', label='szz')
plt.xscale('log')
plt.xlabel('Temperature')
plt.ylabel('szz')
plt.legend()
plt.title('szz vs Temperature')

# Plot spm(real) and spm(imag)
plt.subplot(1, 2, 2)
plt.plot(temperature, data[:, 1], 'b-', label='spm')
plt.xscale('log')
plt.xlabel('Temperature')
plt.ylabel('spm')
plt.legend()
plt.title('spm vs Temperature')

plt.tight_layout()
plt.savefig(dir + 'szz_spm_plot.png')

# Create 4x4 subplot for szz# and spm#
fig, axs = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle('Site-specific correlations', fontsize=16)

fig1, axs1 = plt.subplots(4, 4, figsize=(20, 16))
fig1.suptitle('Site-specific correlations', fontsize=16)

# Organize columns
site_columns = {}
for i, col_name in enumerate(column_names):
    # Extract site indices from column names
    match = re.search(r'(szz|spm)(\d+)', col_name)
    if match:
        corr_type, site = match.groups()
        site = int(site)
        if site not in site_columns:
            site_columns[site] = {}
        
        # Store column index for each correlation type and site
        if '(real)' in col_name:
            site_columns[site][f"{corr_type}_real"] = i
        elif '(imag)' in col_name:
            site_columns[site][f"{corr_type}_imag"] = i

plt.savefig(dir + 'site_correlations_subplot_szz.png')
# Plot each site in the 4x4 grid
for site in range(16):
    row = site // 4
    col = site % 4
    
    if site in site_columns:
        # Plot szz for this site
        if 'szz_real' in site_columns[site]:
            axs[row, col].plot(temperature, data[:, site_columns[site]['szz_real']], 'r-', label='szz(real)')
        
        # Plot spm for this site
        if 'spm_real' in site_columns[site]:
            axs1[row, col].plot(temperature, data[:, site_columns[site]['spm_real']], 'b-', label='spm(real)')
        
        axs[row, col].set_title(f'Site {site}')
        axs[row, col].set_xscale('log')
        
        axs1[row, col].set_title(f'Site {site}')
        axs1[row, col].set_xscale('log')

# Add labels to the figure
for ax in axs.flat:
    ax.set(xlabel='Temperature', ylabel='Correlation')

for ax in axs1.flat:
    ax.set(xlabel='Temperature', ylabel='Correlation')



plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'site_correlations_subplot_spm.png')

plt.show()