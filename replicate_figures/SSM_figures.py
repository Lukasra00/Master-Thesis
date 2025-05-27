import pandas as pd
import numpy as np
import ipdb
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from glob import glob
import re
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



from utils import palette_light, palette_saturated, amino_acids, hydro_scale



# This script can be used to re-create the panel concerning itself with the SSM
out_dir = '/disk2/lukas/Master-Thesis/replicate_figures/figure_output/'

# Subfigure for ZS site prediction
corr_df = '/disk2/lukas/Master-Thesis/data/Correlation_df.csv'
out_SSM = os.path.join(out_dir,'SSM_panel', 'ZS_SSM_site_prediction.png')
corr_df = pd.read_csv(corr_df)
ranking_dict = {}
columns_to_rank = [
    'AF3', 'EVmutation', 'EVmutation + AF3_$w$', 'GT', 'mmseqs2', 'noMSA',
    'jackhmmr_notemplate', 'jackhmmr_unpaired', 'jackhmmr_d484',
    'mmseqs2_idx2idx0', 'noMSA_idx2idx0', 'jackhmmr_notemplate_idx2idx0',
    'jackhmmr_unpaired_idx2idx0', 'jackhmmr_d484_idx2idx0'
]
for col in columns_to_rank:
    site_avg = corr_df.groupby('site')[col].mean()
    ranked_sites = site_avg.sort_values(ascending=False).index.tolist()
    ranking_dict[col] = ranked_sites

sites = ['56X', 
         '59X', 
         '60X', 
         '61X', 
         '63X', 
         '69X', 
         '73X', 
         '85X', 
         '86X', 
         '89X', 
         '90X', 
         '93X', 
         '145X'
        ] 
corr_df['AF3'] = -corr_df['AF3'] # reinvert, because FZL inverted before
grouped_af3 = [corr_df[corr_df['site'] == site]['AF3'].dropna().values 
               for site in sites
            ]

plt.figure(figsize=(8, 5))
box = plt.boxplot(grouped_af3, 
            labels=[site[:-1] for site in sites],
            patch_artist=True,
            medianprops=dict(color='black', 
                             linewidth=1
                             ),
            showfliers=False,
            showmeans=True,
            meanprops=dict(marker='o', 
                markerfacecolor=palette_saturated['orange'], 
                markeredgecolor='black',
                markersize=7)
            )
for patch in box['boxes']:
    patch.set_facecolor('silver')
for i, data in enumerate(grouped_af3, start=1):
    x = np.random.normal(loc=i, 
                         scale=0.025, 
                         size=len(data))  
    plt.plot(x, 
             data, 
             '.', 
             alpha=0.6, 
             markersize=5,
             color='dimgrey')
plt.title('AF3 ZS prediction for Selected Sites')
plt.xlabel('Site')
plt.ylabel('ZS (AF3 pAE Cofactor-Protein)')
plt.tight_layout()
plt.savefig(out_SSM)


# Subfigure for assessing the spearman-rho between GT and ZS
corr_df_filtered = corr_df[corr_df['GT'].notna()]
#corr_df_filtered = corr_df[corr_df['GT'] > 0.1]
spearman_corrs = {}
for col in columns_to_rank:
    if col != 'GT':
        valid = corr_df_filtered[[col, 'GT']].dropna()
        if not valid.empty:
            corr, pval = spearmanr(valid[col], valid['GT'])
            spearman_corrs[col] = (corr.round(2), pval.round(2))
        else:
            spearman_corrs[col] = None  # not enough data
spearman_df = pd.DataFrame(list(spearman_corrs.items()), columns=['Column', 'Spearman Correlation with GT'])


# Subfigure generating a bar-plot for each individual SSM site

# also keep fitness matrix for following figure
parent_seq = "MTPSDIPGYDYGRVEKSPITDLEFDLLKKTVMLGEKDVMYLKKAHDVLKDQVDEILDLTGGWAASNEHLIYYVSNPDTGEPIKEYLERAGARFGAWILDTTCRDYNREWLDYQYEVGLRHHRSKKGVTDGVRTAPHIPLRYLIAWIYPQTATIKPFLAKKGGSPEDIEGMYNAWFKSVVLQVAIWSHPYTKEND"
heatmap_matrix = np.zeros((len(parent_seq), len(amino_acids)))

plate_paths = glob('/disk2/lukas/Master-Thesis/data/PgA93_*_plate.csv')
for plate_path in plate_paths:
    plate = pd.read_csv(plate_path, header=None)
    fit = plate.iloc[:8]
    seq = plate.iloc[8:]
    seqfit_dict = {aa: [] for aa in amino_acids}
    for row in range(fit.shape[0]):
        for col in range(fit.shape[1]):
            fit_val = fit.iat[row, col]
            mut_str = seq.iat[row, col]
            if len(mut_str) <= 5 and len(mut_str) > 3 and '#' not in mut_str and '*' not in mut_str: 
                parent = mut_str[0]
                mutant = mut_str[-1]
                plate_nr = mut_str
                seqfit_dict[mutant].append(float(fit_val))
            if mut_str == '#PARENT#':
                seqfit_dict[parent].append(float(fit_val))


    plate_nr = re.findall(r'\d+', plate_nr)[0]
    means = [np.mean(seqfit_dict[aa]) if seqfit_dict[aa] else np.nan for aa in amino_acids]
    errors = [np.std(seqfit_dict[aa]) if len(seqfit_dict[aa]) > 1 else 0 for aa in amino_acids]
    plt.figure(figsize=(10, 5))
    bar_colors = ['darkgray' if aa != parent else palette_light['green'] for aa in amino_acids]
    plt.bar(x=seqfit_dict.keys(), 
            height=[np.mean(seqfit_dict[key]) for key in seqfit_dict.keys()],
            color=bar_colors,
            yerr=errors,
            capsize=4)
    for i, aa in enumerate(amino_acids):
        y_vals = seqfit_dict[aa]
        x_vals = np.random.normal(loc=i, scale=0.08, size=len(y_vals))  
        plt.scatter(x_vals, 
                    y_vals, 
                    color='dimgrey', 
                    s=20, 
                    zorder=3
                    )    
    parent_avg = np.mean(seqfit_dict[parent])
    plt.axhline(y=parent_avg, linestyle='--', color=palette_saturated['orange'], linewidth=1.5, alpha=0.8)
    plt.ylabel('Fitness')
    plt.xlabel('Amino Acid')
    plt.title(plate_nr+'X')
    plt.ylim(0, 2)
    plt.tight_layout()
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(f'/disk2/lukas/Master-Thesis/replicate_figures/figure_output/SSM_panel/barplot_{plate_nr}X.png')

    heatmap_matrix[int(plate_nr)-1] = means


# Subfigure, matrix plot of PgA9.3 SSM positions and previously targeted positions
parent_seq = "MTPSDIPGYDYGRVEKSPITDLEFDLLKKTVMLGEKDVMYLKKAHDVLKDQVDEILDLTGGWAASNEHLIYYVSNPDTGEPIKEYLERAGARFGAWILDTTCRDYNREWLDYQYEVGLRHHRSKKGVTDGVRTAPHIPLRYLIAWIYPQTATIKPFLAKKGGSPEDIEGMYNAWFKSVVLQVAIWSHPYTKEND"
previously_targeted= [61, 62, 45, 113, 149, 185, 102, 142, 148, 19, 181]

# generate green mask for parent residues
mask_parent =  np.zeros_like(heatmap_matrix)
for seq_pos in range(len(parent_seq)):
    aa = parent_seq[seq_pos]
    aa_index = amino_acids.index(aa)
    mask_parent[seq_pos, aa_index] = 1

# generate darkgrey mask for previously assessed positions
darkgrey_mask = np.zeros_like(heatmap_matrix, dtype=bool)
for row in previously_targeted:
    row = row-1
    non_parent_cols = [i for i in range(len(amino_acids)) if mask_parent[row, i] == 0]
    non_parent_values = heatmap_matrix[row, non_parent_cols]
    if np.all(non_parent_values == 0):
        darkgrey_mask[row, non_parent_cols] = True

# colorgradient for experimental fitness data
masked_matrix = np.ma.masked_where(mask_parent, heatmap_matrix)
fig, ax = plt.subplots(figsize=(9, 6))
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", 
                                                ['white', 
                                                 palette_saturated['pink']])
im1 = ax.imshow(masked_matrix, cmap=custom_cmap, norm=Normalize(vmin=0, vmax=2))
overlay_data = np.full_like(heatmap_matrix, np.nan)
overlay_data[mask_parent.astype(bool)] = 1  
cmap_overlay = ListedColormap([palette_light["green"]])
im2 = ax.imshow(overlay_data, cmap=cmap_overlay, alpha=1.0)
overlay_darkgrey = np.full_like(heatmap_matrix, np.nan)
overlay_darkgrey[darkgrey_mask] = 1
cmap_darkgrey = ListedColormap(["darkgrey"])
ax.set_xticks(np.arange(len(amino_acids)))
ax.set_yticks(np.arange(len(parent_seq)))
ax.set_xticklabels(amino_acids)
ax.tick_params(axis='x', labelsize=2)
ax.tick_params(axis='y', labelsize=2)
ax.set_yticklabels(parent_seq)
plt.setp(ax.get_xticklabels(), ha="center")
plt.setp(ax.get_yticklabels(), ha="center")
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.tick_params(axis='x', length=1, width=0.5)
for i, tick in enumerate(ax.yaxis.get_major_ticks()):
    if i % 10 == 9 or i == 0:
        tick.tick1line.set_markersize(3)  
        tick.tick2line.set_markersize(3)
    else:
        tick.tick1line.set_markersize(0)  
        tick.tick2line.set_markersize(0)
zero_mask = (heatmap_matrix == 0) & ~mask_parent.astype(bool)
overlay_zero = np.full_like(heatmap_matrix, np.nan)
overlay_zero[zero_mask] = 1
cmap_zero = ListedColormap(["lightgray"])
im0 = ax.imshow(overlay_zero, cmap=cmap_zero, alpha=1.0)
im_dark = ax.imshow(overlay_darkgrey, cmap=cmap_darkgrey, alpha=1.0)
ax.xaxis.set_label_position('top')
ax.set_xlabel("Protoglobin Map", fontsize=3, weight='bold')   
tick_vals = np.arange(9, len(parent_seq), 10)       
tick_labels = [str(i + 1) for i in tick_vals]
ax2 = ax.secondary_yaxis('right')
ax2.set_yticks(tick_vals)
ax2.set_yticklabels(tick_labels)
ax2.yaxis.set_minor_locator(MultipleLocator(1))
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(5)  
    label.set_horizontalalignment('center')  
ax2.tick_params(axis='y', which='major', length=2, labelsize=3, pad=6)
ax2.tick_params(axis='y', which='minor', length=1, width=0.5, labelsize=0, direction='out')
ax.set_xticks(np.arange(-0.5, heatmap_matrix.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, heatmap_matrix.shape[0], 1), minor=True)
ax.grid(which='minor', color='lightgrey', linewidth=0.2)
ax.tick_params(axis='both', which='minor', length=0, labelsize=0)
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.6, 0.065, 0.05, 0.01])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Variant Fitness", fontsize=4, fontweight='bold')
cbar.ax.tick_params(labelsize=3)
for label in cbar.ax.get_xticklabels():  
    label.set_fontweight('bold')
cbar.set_ticks([0, 2])
plt.tight_layout()

# add a hydrophobicity plot
hydro_values = [hydro_scale.get(res, 0.0) for res in parent_seq]
box = ax.get_position()
hydro_ax = fig.add_axes([box.x1 + 0.025, box.y0, 0.022, box.height])
hydro_ax.barh(np.arange(len(parent_seq)), hydro_values, height=1.0, color='darkgrey')
hydro_ax.set_ylim(ax.get_ylim())  # Match matrix y-axis
hydro_ax.set_yticks(ax2.get_yticks())
hydro_ax.set_yticklabels([])
hydro_ax.tick_params(axis='y', which='major', length=2)
hydro_ax.yaxis.set_minor_locator(MultipleLocator(1))
hydro_ax.tick_params(axis='y', which='minor', length=1, width=0.5) 
hydro_ax.xaxis.set_label_position('top')
hydro_ax.set_xlabel("Kyte–Doolittle\n Hydropathy", fontsize=3, weight='bold')       
hydro_ax.set_xticks([])     
hydro_ax.set_xticks([-4.5, 0, 4.5])
hydro_ax.set_xticklabels(['-4.5', '0', '4.5'])
hydro_ax.tick_params(axis='x', labelsize=3, width=0.5, length=1, direction='out')
for label in hydro_ax.get_xticklabels():
    label.set_fontweight('bold')     

heatmap_path = os.path.join(out_dir, 'SSM_heatmap.png')
plt.savefig(heatmap_path, dpi=900)


# Subfigure: Correlation analysis of ZS metrics
plt.figure()
plt.plot(corr_df['GT'], 
         corr_df['AF3'], 
         'o', 
         markersize=5,
         color='gray')
plt.title('ZS AF3 Jackhmmr')
plt.xlabel('GT Fitness')
plt.ylabel('ZS')
for spine in plt.gca().spines.values():
    spine.set_visible(False)

corr_dotplot_path = os.path.join(out_dir, 'ZS_corr_dotplot_jackhmmr.png')
plt.savefig(corr_dotplot_path, dpi=900)



 