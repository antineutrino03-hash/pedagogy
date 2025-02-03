import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib.markers as markers
from matplotlib.markers import MarkerStyle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cms
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, LogLocator, MultipleLocator
from matplotlib.ticker import FuncFormatter

from scipy.interpolate import interp1d
sys.path.append("/Users/asm18/Documents/python_repo/pyReSpect-freq/")
import contSpec as cs
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use("QtAgg")

rheology_labels = {
    'Time': 'Time, $t$ [s]',
    'Elongational modulus (dyn/cm2)':'Elongational modulus, $E$ [dyn/cm\^2]',
    'master Time (hrs)': 'Reduced Time, $t_r$ [hrs]',
    'master Elongational modulus (dyn/cm2)':'Reduced modulus, $E_r$ [dyn/cm\^2]',
    'Temperature': 'Temperature, $T$ [\si{\degreeCelsius}]',
    'Angular frequency': 'Angular frequency, $\\omega$ [rad/s]',
    'Viscosity': 'Viscosity, $\\eta$ [Pa.s]',
    'Shear stress': 'Shear stress, $\\tau$ [Pa]',
    'Shear rate': 'Shear rate, $\\dot{\\gamma}$ [s^-1]',
    'Relaxation modulus': 'Stress Relaxation modulus, $G(t)$ [Pa]',
    'Storage modulus': 'Viscoelastic Moduli, $G\', G\'\'$ [Pa]',
    'Loss modulus': 'Viscoelastic Moduli, $G\', G\'\'$ [Pa]',
    'Complex modulus': 'Complex Modulus, $G^*$ [Pa]',
    'Complex viscosity': 'Complex viscosity, $\\eta^*$ [Pa.s]',
    'Tan(delta)': 'Phase angle, tan $\\delta$ [ ]',
    'master Storage modulus': 'Reduced Moduli, $G\'_r, G\'\'_r$ [Pa]',
    'master Loss modulus': 'Reduced Moduli, $G\'_r, G\'\'_r$ [Pa]',
    'master Angular frequency': 'Reduced Angular frequency, $\\omega_r$ [rad/s]',
    'master Tan(delta)': 'Phase angle, tan $\\delta$ [ ]',
    'log Angular frequency': 'Angular frequency, $\\log{\\omega}$ [rad/s]',
    'log Complex modulus': 'Complex Modulus, $\\log{G^*}$ [Pa]',
    'log Storage modulus': 'Viscoelastic Moduli, $\\log{G\'}, \\log{G\'\'}$ [Pa]',
    'log Loss modulus': 'Viscoelastic Moduli, $\\log{G\'}, \\log{G\'\'}$ [Pa]',
    'None': '',
    'master None': '',
    'Strain':'Strain, $\\gamma_0$ [\%]',
}

criteria_labels = {
    'Temperature': 'OA$_k$',
    'overlap_area': 'OA$_k$',
    'linearity': r'$R^2$',
    'overlap_length': '$\\text{OR}_{\\text{len}}$',
    'overlap_num': '$\\text{OR}_{\\text{num}}$',
    'cos_measure': 'Cos',
    'overlap_index': 'OI$_k$',
    'frechet': '$d_{\\text{Fr}}$',
    'pearson': r'$\text{PCC}_k$',
    'shift_variation': '$\\text{nStd}(a_T)$',
    'phase_area': '$A_{t-T}$'
}

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = '/Users/asm18/Documents/python_repo/time_temperature/'

def set_plot_params(temp_arr=np.ones((50,)), model_fit=0, param_plot=0, combined_plot=0, lean_plot=0, model_fit2=0, single_plot=0, model_fit3=0, spectrum_fit=0, num_plots=10):
    plt.style.use('default')
    
    # Set main font to serif, but math text (like labels) to sans-serif
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx} \usepackage{bm} \usepackage[cm]{sfmath}'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.default'] = 'sf'  # Sans-serif for math text
    plt.rcParams['mathtext.fontset'] = 'stixsans'

    line_thickness = 1.5

    # Set linewidths and font sizes for better visibility
    plt.rcParams['axes.linewidth'] = line_thickness
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['lines.markersize'] = 7
    plt.rcParams['errorbar.capsize'] = 4
    plt.rcParams['lines.linewidth'] = 1.5 * line_thickness
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['blue'])

    # Font sizes for labels, ticks, and legends
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 12

    plt.rcParams['legend.title_fontsize'] = 11
    plt.rcParams['axes.labelweight'] = 'bold'

    # Tick and padding adjustments
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['xtick.major.width'] = line_thickness
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['ytick.major.width'] = line_thickness
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.major.pad'] = 5
    plt.rcParams['ytick.major.pad'] = 5

    # Prepare marker and color arrays
    symbol_array = ['o', 's', '^', '>', '<', 'v', 'D', 'p', 'h', '8']
    colors = [(0, 0, 1), (0, 0.75, 0), (1, 0, 0)]
    cmap_name = 'myRGB'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    colormap = cms.coolwarm
    norm = plt.Normalize(np.min(temp_arr), np.max(temp_arr))
    color_array = colormap(norm(temp_arr))

    # Return based on plot types, handling single, combined, etc.
    if single_plot:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        return fig, [axes], symbol_array, color_array, norm
    elif combined_plot:
        fig = plt.figure(figsize=(17, 12))
        spec = mpl.gridspec.GridSpec(ncols=6, nrows=3)
        
        axes = [
            fig.add_subplot(spec[i, j:j+2], projection='polar' if (i == 2 and j == 2) else None)
            for i in range(3)
            for j in range(0, 6, 2)
        ]
        
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
        
        return fig, axes, symbol_array, color_array, norm
    
    elif lean_plot:
        fig = plt.figure(figsize=(17, 8))
        spec = mpl.gridspec.GridSpec(ncols=6, nrows=2)
        
        axes = [
            fig.add_subplot(spec[i, j:j+2], projection='polar' if (i == 1 and j == 0) else None)
            for i in range(2)
            for j in range(0, 6, 2)
        ]
        
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
        
        return fig, axes, symbol_array, color_array, norm
    elif model_fit3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        return fig, axes, symbol_array, color_array, norm
    elif model_fit:
        fig = plt.figure(figsize=(10, 12))
        spec = mpl.gridspec.GridSpec(ncols=4, nrows=3)
        axes = [fig.add_subplot(spec[i, j:j+2]) for i in range(3) for j in range(0, 4, 2)]
        return fig, axes, symbol_array, color_array, norm
    elif spectrum_fit:
        fig = plt.figure(figsize=(15, 8))
        spec = mpl.gridspec.GridSpec(ncols=3, nrows=2)
        axes = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(3)]
        
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
        
        return fig, axes, symbol_array, color_array, norm
    elif param_plot:
        fig = plt.figure(figsize=(10, 4))
        spec = mpl.gridspec.GridSpec(ncols=2, nrows=1)
        axes = [fig.add_subplot(spec[0, i]) for i in range(2)]
        return fig, axes, symbol_array, color_array, norm
    elif model_fit2:
        fig = plt.figure(figsize=(10, 8))
        spec = mpl.gridspec.GridSpec(ncols=4, nrows=2)
        axes = [fig.add_subplot(spec[i, j:j+2]) for i in range(2) for j in range(0, 4, 2)]
        return fig, axes, symbol_array, color_array, norm
    
    if 1 <= num_plots <= 3:
        fig, axes = plt.subplots(1, num_plots, figsize=(5.5*num_plots-1, 4))
        if num_plots == 1:
            for label in axes.get_xticklabels() + axes.get_yticklabels():
                    label.set_fontfamily('sans-serif')
                    label.set_fontname('DejaVu Sans') 
                    
            return fig, [axes], symbol_array, color_array, norm
        else:
            for ax in axes:
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily('sans-serif')
                    label.set_fontname('DejaVu Sans') 
            return fig, axes, symbol_array, color_array, norm
    elif num_plots == 4:
        fig = plt.figure(figsize=(11, 8))
        spec = mpl.gridspec.GridSpec(ncols=2, nrows=2)
        axes = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(2)]
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
        
        return fig, axes, symbol_array, color_array, norm
    elif num_plots == 5:
        fig = plt.figure(figsize=(15, 8))
        spec = mpl.gridspec.GridSpec(ncols=3, nrows=2)
        axes = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(3)]
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
        axes[5].axis('off')
        
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
        
        return fig, axes, symbol_array, color_array, norm
    elif num_plots == 6:
        fig = plt.figure(figsize=(15, 8))
        spec = mpl.gridspec.GridSpec(ncols=3, nrows=2)
        axes = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(3)]
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
                
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans')         
                
        return fig, axes, symbol_array, color_array, norm
    elif num_plots == 7:
        fig = plt.figure(figsize=(20, 8))
        spec = mpl.gridspec.GridSpec(ncols=4, nrows=2)
        axes = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(4)]
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
        axes[7].axis('off')
        
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('sans-serif')
                label.set_fontname('DejaVu Sans') 
                
        return fig, axes, symbol_array, color_array, norm

    return fig, ax, symbol_array, color_array, norm

def set_plot_params_temp(temp_arr=np.ones((50,)), model_fit=0, param_plot=0, combined_plot=0, model_fit2=0, single_plot=0, model_fit3=0):

    # plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, siunitx, bm} \renewcommand{\rmdefault}{ptm}'  # Removed sansmath
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['font.family'] = 'serif'

    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['errorbar.capsize'] = 3  # Sets the cap size for error bars
    plt.rcParams['lines.linewidth'] = 1 
    plt.rcParams['axes.labelsize'] = 14

    # Set global tick parameters for major ticks
    plt.rcParams['xtick.major.size'] = 5  
    plt.rcParams['xtick.major.width'] = 1 
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.direction'] = 'in'  
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.title_fontsize'] = 11
    plt.rcParams['grid.color'] = 'k'  # Grid color
    plt.rcParams['grid.alpha'] = 0.5  # Transparency of grid lines

    # Set global tick parameters for minor ticks
    plt.rcParams['xtick.minor.size'] = 3  # Length of minor ticks
    plt.rcParams['xtick.minor.width'] = 1  # Width of minor ticks
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.minor.width'] = 1

    # Set whether ticks are drawn on all sides of the plot
    plt.rcParams['xtick.top'] = True  # Draw ticks on the top axis
    plt.rcParams['xtick.bottom'] = True  # Draw ticks on the bottom axis
    plt.rcParams['ytick.right'] = True  # Draw ticks on the right axis
    plt.rcParams['ytick.left'] = True  # Draw ticks on the left axis

    all_markers = list(MarkerStyle.markers.keys())
    symbol_array = [
        'o',  # Circle (filled)
        's',  # Square (filled)
        '^',  # Triangle up (filled)
        '>',  # Triangle right (filled)
        '<',  # Triangle left (filled)
        'v',  # Triangle down (filled)
        'D',
        'p',  # Pentagon (filled)
        'h',  # Hexagon (filled)
        '8',  # Octagon (filled)
    ]

    colors = [(0, 0, 1), (0, 0.75, 0), (1, 0, 0)]  # R -> G -> B
    n_bins = 100  # Number of bins
    cmap_name = 'myRGB'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    colormap = cms.coolwarm
    norm = plt.Normalize(np.min(temp_arr), np.max(temp_arr))
    color_array = colormap(norm(temp_arr))

    if single_plot:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        return fig, axes, symbol_array, color_array, norm 
    if combined_plot:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
        return fig, axes, symbol_array, color_array, norm 
    if model_fit3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        return fig, axes, symbol_array, color_array, norm 
    if model_fit:
        fig = plt.figure(figsize=(10, 12))
        spec = mpl.gridspec.GridSpec(ncols=4, nrows=3) # 6 columns evenly divides both 2 & 3
        axes = []
        axes.append(fig.add_subplot(spec[0,1:3])) # row 0 with axes spanning 2 cols on evens
        axes.append(fig.add_subplot(spec[1,0:2]))
        axes.append(fig.add_subplot(spec[1,2:]))
        axes.append(fig.add_subplot(spec[2,0:2])) # row 0 with axes spanning 2 cols on odds
        axes.append(fig.add_subplot(spec[2,2:]))
        
        return fig, axes, symbol_array, color_array, norm
    if param_plot: 
        fig = plt.figure(figsize=(10, 4))
        spec = mpl.gridspec.GridSpec(ncols=2, nrows=1)  # Specify 2 columns and 1 row
        axes = []
        axes.append(fig.add_subplot(spec[0, 0]))  # First subplot in the first column
        axes.append(fig.add_subplot(spec[0, 1]))  # Second subplot in the second column
    
        return fig, axes, symbol_array, color_array, norm
    else:
        if model_fit2:
            fig = plt.figure(figsize=(10, 8))
            spec = mpl.gridspec.GridSpec(ncols=4, nrows=2) # 6 columns evenly divides both 2 & 3
            axes = []
            axes.append(fig.add_subplot(spec[0,0:2])) # row 0 with axes spanning 2 cols on evens
            axes.append(fig.add_subplot(spec[0,2:]))
            axes.append(fig.add_subplot(spec[1,1:3]))
            return fig, axes, symbol_array, color_array, norm
        else:
            if temp_arr[0] == np.ones((50,))[0]:
                fig, ax = plt.subplots(figsize=[6, 4])
            else:
                fig, ax = plt.subplots(figsize=[6, 4])

    return fig, ax, symbol_array, color_array, norm

def plot_normalized_data(prefix, normalized_data_dict, xcol, plot_column_indices=None, filename='NO_NAME'):
    lgd = []
    first = True
    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + 'tts_criteria')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H']
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    fig,ax, *_ = set_plot_params()
    storage_marker = dict(marker='o', linestyle='-')
    loss_marker = dict(marker='o', linestyle='-', fillstyle='none')
    x_values = xcol  # Extracting the x-axis values from the DataFrame
    if plot_column_indices is None:
        plot_column_indices = np.arange(len(normalized_data_dict['Original'].columns))[1:]
    all_columns = normalized_data_dict['Original'].columns[1:9]
    selected_columns = [normalized_data_dict['Original'].columns[i] for i in plot_column_indices]
    i = 0
    
    for key, normalized_df in normalized_data_dict.items():
        if first:  
            first = False 
            continue
        for column in all_columns:
            if column in normalized_df.columns:
                loss_marker["marker"] = markers[i]
                loss_marker["color"] = colors[i]
                
                if column == 'Shift variation':
                    ax.plot(x_values, normalized_df[column], linewidth=2, alpha=1.0, **loss_marker)
                    print('Shift variation is working')
                else:
                    if column in selected_columns:
                        err_col = normalized_data_dict['Original']['u_' + column] / normalized_data_dict['Original'][column].abs().max()
                        ax.errorbar(x_values, normalized_df[column], yerr= err_col, capsize=5, linewidth=2, **loss_marker)
                    else:
                        err_col = normalized_data_dict['Original']['u_' + column] / normalized_data_dict['Original'][column].abs().max()
                        ax.errorbar(x_values, normalized_df[column], yerr= err_col, capsize=5, linewidth=2, alpha=0.2, **loss_marker)
                lgd.append(f"{column}: {normalized_data_dict['Original'][column].mean():.4f}")
                i = i + 1

    ax.set_title('tTS criteria across temperatures')
    ax.set_xlabel("Temperature, $T$ [°C]")
    ax.set_ylabel("(Max Abs) Normalised criteria")
    ax.grid(False)
    ax.legend(lgd, loc='upper left', bbox_to_anchor=(1, 1), title="Average values")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tts_criteria_{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plotter(data_frames, temp_arr, tref, prefix, typ, dict_shift, mastercurve_df, logged=0, x1_str='Angular frequency', y1_str='Complex modulus', y2_str='', gmin=0, filename='NO_NAME'):

    i, j, k = 0, 0, 0
    lgd = []
    output_folder= " "
    flg=0
    # print(mastercurve_df)
    
    for key, value in dict_shift.items():
        aTs, bTs, daTs, dbTs = value

    fig, ax, symbol_array, color_array, norm = set_plot_params(temp_arr)

    storage_marker = dict(marker='.', linestyle='-', label='Filled Markers')
    loss_marker = dict(marker='.', linestyle='-', fillstyle='none', label='Empty Markers')

    # region PLOT TYPE SELECTION
    if typ == 'raw':
        nc = len(temp_arr) / 4
        aTs = np.ones_like(aTs)
        bTs = np.ones_like(bTs)
        daTs = np.zeros_like(aTs)
        dbTs = np.zeros_like(bTs)
        output_folder= "raw_fs"
    elif typ == 'master':
        symbol_array[1:] = symbol_array[0]
        nc = len(temp_arr)
        output_folder= "master"
    elif typ == 'shifted':
        nc = len(temp_arr)
        output_folder= "shifted"
    else:
        print("Enter a valid plot type\n")
    # endregion 

    # region PLOT LIMIT CALCULATIONS
    all_y1 = []
    all_y2 = []
    all_x1 = []
    a = 0

    if (y1_str=='Tan(delta)') and (x1_str=='Complex modulus'):
        aTs = bTs
        bTs = np.ones_like(bTs)

    if (y1_str=='Loss modulus') and (x1_str=='Storage modulus'):
        aTs = bTs
        # bTs = np.ones_like(bTs)

    if y2_str == '':
        y2_str = y1_str
        flg = 0

    for temp in temp_arr:
        sheet_name = 'Frequency sweep (' + str(round(temp, 1)) + ' °C)'
        df = data_frames[sheet_name]
        x1_var =  df[x1_str].astype(np.float64)
        y1_var =  df[y1_str].astype(np.float64)
        y2_var =  df[y2_str].astype(np.float64)

        all_y1.extend(bTs[a] * y1_var)
        all_y2.extend(bTs[a] * y2_var)
        all_x1.extend(aTs[a] * x1_var)
        a = a + 1

    min_x1 = min(all_x1)
    max_x1 = max(all_x1)
    min_y1 = min(all_y1)
    max_y1 = max(all_y1)
    min_y2 = min(all_y2)
    max_y2 = max(all_y2)
    # endregion

    if not logged:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, temp in enumerate(temp_arr):
        if (typ == 'master') and (i==2):
                break
        sheet_name='Frequency sweep ('+str(round(temp_arr[i], 1)) + ' °C)'
        df = data_frames[sheet_name]
        
        x1 = aTs[i] * df[x1_str].astype(np.float64)
        x1_err = daTs[i] * df[x1_str].astype(np.float64)
        y1 = bTs[i] * df[y1_str].astype(np.float64)
        y2 = bTs[i] * df[y2_str].astype(np.float64)

        if y1_str=='Storage modulus' or y1_str=='Loss modulus':
            y1_err =  bTs[i] * df[y1_str].astype(np.float64) * np.sqrt((dbTs[i] / bTs[i])**2 + (df['u_'+y1_str].astype(np.float64)/ df[y1_str].astype(np.float64))**2)
            y2_err =  bTs[i] * df[y2_str].astype(np.float64) * np.sqrt((dbTs[i] / bTs[i])**2 + (df['u_'+y2_str].astype(np.float64)/ df[y2_str].astype(np.float64))**2)
        else:
            if i == 0:
                print('Uncertainties are not available')
            y1_err = np.zeros_like(y1)
            y2_err = np.zeros_like(y2)
        
        if i % int(len(temp_arr) / nc) == 0 or i == len(temp_arr) - 1:

            storage_marker["marker"] = symbol_array[j]
            loss_marker["marker"] = symbol_array[j]
            storage_marker["color"] = color_array[i]
            loss_marker["color"] = color_array[i]

            me = 1
            if logged:
                ax.errorbar(np.log10(x1), np.log10(y1), yerr=(1/2.303) * (df['u_' + y1_str]/df[y1_str]), capsize=3, linewidth=1, **storage_marker)
                ax.errorbar(np.log10(x1), np.log10(y2), yerr=(1/2.303) * (df['u_' + y2_str]/df[y2_str]), capsize=3, linewidth=1, **loss_marker)  

                plt.axhline(y=np.log10(np.max(gmin*bTs)), color='r', linestyle='--', label='_nolegend_')
            else:
                if typ == 'master':
                    x1 = mastercurve_df['Angular frequency']
                    y1 = mastercurve_df['Storage modulus']
                    y2 = mastercurve_df['Loss modulus']
                    x1_err = mastercurve_df['u_Angular frequency']
                    y1_err = mastercurve_df['u_Storage modulus']
                    y2_err = mastercurve_df['u_Loss modulus']

                    # ax.errorbar(x1, y1, yerr=0*y1_err, xerr=x1_err, capsize=3, linewidth=1, zorder=2, **storage_marker)
                    ax.fill_between(x1, y1 - y1_err, y1 + y1_err, alpha=0.2, color='blue', linewidth=1, zorder=1)
                    ax.fill_betweenx(y1, x1 - x1_err, x1 + x1_err, alpha=0.2, color='red', linewidth=1, zorder=1)
                else:
                    ax.plot(x1, y1, linewidth=1, zorder=1, linestyle='-', color=color_array[i])
                    ax.errorbar(x1[::me], y1[::me], yerr=y1_err[::me], xerr=daTs[i]*x1[::me], capsize=3, linewidth=1, zorder=2, **storage_marker)
                # ax.fill_between(x1, y1 - y1_err, y1 + y1_err, alpha=0.2, color='blue', linewidth=1, zorder=1)
                if flg:
                    ax.plot(x1, y2, linewidth=1, zorder=1, linestyle='-', color=color_array[i])
                    ax.errorbar(x1[::me], y2[::me], yerr=y2_err[::me], xerr=daTs[i]*x1[::me], capsize=3, linewidth=1, zorder=2, **loss_marker)
                     
                # error_container = ax.plot(x1, y1, linewidth=1, zorder=1, **storage_marker)
                # error_container1 = ax.plot(x1, y2, linewidth=1, zorder=1, **loss_marker)  
                plt.axhline(y=(np.max(gmin*bTs)), color='r', linestyle='--', label='_nolegend_')           

            # lgd.append("$G':$ " + str(round(temp, 1)) + '°C')
            # lgd.append("$G'':$ " + str(round(temp, 1)) +'°C')

            j = (j+1) % len(symbol_array)
            k = (k+int(len(temp_arr) / nc) - 1) % len(color_array)

    # region PLOT BELLS AND WHISTLES
    if logged:
        max_value = np.log10(max(value for value in max(aTs)*df[x1_str] if not math.isnan(value)))
        min_value = np.log10(min(value for value in min(aTs)*df[x1_str] if not math.isnan(value)))

        ax.set_xlim(0.5 * min_x1, 3*max_x1)
        ax.set_ylim(0.5 * min(min_y1, min_y2), 5*max(max_y1, max_y2))  # Y-axis limits

        ax.set_xlabel(x1_str)
        ax.set_ylabel(y1_str)
    else:
        max_value = max(value for value in max(aTs)*df[x1_str] if not math.isnan(value))
        min_value = min(value for value in min(aTs)*df[x1_str] if not math.isnan(value))

        ax.set_xlim(0.5 * min_x1, 5*max_x1)
        ax.set_ylim(0.5 * min(min_y1, min_y2), 5*max(max_y1, max_y2))
        ax.set_xlabel(rheology_labels[x1_str])
        ax.set_ylabel(rheology_labels[y1_str])

    gradient_image = plt.imshow(np.array([[np.min(temp_arr), np.max(temp_arr)]]), cmap=cms.coolwarm, aspect='auto', norm=plt.Normalize(np.min(temp_arr), np.max(temp_arr)))
    gradient_image.set_visible(False) 
    cbar = plt.colorbar(gradient_image, orientation='vertical', pad=0.05, aspect=25)
    cbar.set_label('Temperature, $T$ [°C]', fontsize=14)
    if typ in ['master', 'shifted']:
        lgd = [y1_str]
        if flg:
            lgd = [y1_str, y2_str]
        ax.legend(lgd, title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C', loc='best')
        
    else:
        lgd = [y1_str]
        if flg:
            lgd = [y1_str, y2_str]
        ax.legend(lgd, title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C')
    # endregion
        
    plt.tight_layout()
    fig.savefig(os.path.join(parent_dir + 'figures', prefix + '_' + output_folder, prefix + '_' + output_folder + '_' + filename + '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 

def slope_plotter(data_frames, temp_arr, nc, prefix, typ, aTs, bTs):
    xvar = 'Angular frequency'
    # yvar1 = ''
    i, j, k = 0, 0, 0
    lgd = []
    nc = nc - 1
    if SELECT:
        nc = len(temp_arr)
    all_markers = list(markers.MarkerStyle.markers.keys())
    symbol_array = all_markers[2:]
    colormap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(np.min(temp_arr), np.max(temp_arr))
    color_array = colormap(norm(temp_arr))
    overlap_arr = np.zeros(len(temp_arr)-1)
    overlap_arr_num = np.zeros(len(temp_arr)-1)
    wls_arr = np.zeros(len(temp_arr))
    # prefix = prefix.replace("oss_", "")

    gp_artifacts = np.zeros(len(temp_arr)-1)
    gpp_artifacts = np.zeros(len(temp_arr)-1)

    plt.figure(figsize=(40, 8))

    output_folder= " "
    storage_marker = dict(marker='.', linestyle='-', markersize=4, label='Filled Markers')
    loss_marker = dict(marker='.', linestyle='-', markersize=4, fillstyle='none', label='Empty Markers')

    if typ == 'raw':
        aTs = np.ones_like(aTs)
        bTs = np.ones_like(bTs)
        output_folder= "raw_fs"
    elif typ == 'master':
        symbol_array[1:] = symbol_array[0]
        nc = len(temp_arr)
        output_folder= "master"
    elif typ == 'shifted':
        nc = len(temp_arr)
        output_folder= "shifted"
    else:
        print("Enter a valid plot type\n")

    all_storage_moduli = []
    all_loss_moduli = []
    all_frequencies = []
    a = 0
    for temp in temp_arr:
        sheet_name = 'Frequency sweep (' + str(round(temp, 1)) + ' °C)'
        df = data_frames[sheet_name]
        all_storage_moduli.extend(bTs[a] * df['Storage modulus'])
        all_loss_moduli.extend(bTs[a] * df['Loss modulus'])
        all_frequencies.extend(aTs[a] * df['Storage modulus'])
        a = a + 1

    plt.figure()
    plt.xscale("log")
    # plt.yscale("log")
    
    # Check if the directory exists before trying to create it
    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    min_x = 0
    max_x = 0
    min_dy = 0
    max_dy = 0
    
    for i in range(len(temp_arr)):
        sheet_name='Frequency sweep ('+str(round(temp_arr[i], 1)) + ' °C)'
        df = data_frames[sheet_name]
        temp = temp_arr[i]

        if i % int(len(temp_arr) / nc) == 0 or i == len(temp_arr) - 1:
            # angular_frequency = np.array([0.1, 0.158489, 0.251189, 0.398107, 0.630957, 1.0, 1.58489, 2.51189, 3.98107,
            #                   6.30957, 10.0, 15.8489, 25.1189, 39.8107, 63.0957, 100.0, 158.489, 251.189,
            #                   398.107, 500.0])
            # print(i)
            x_temp = np.array(aTs[i] * df['Angular frequency'].astype(np.float64))
            y_temp = np.log10(np.array(bTs[i] * df['Storage modulus'].astype(np.float64)))

            not_nan_indices = ~np.isnan(y_temp)
            x = x_temp[not_nan_indices]
            y = y_temp[not_nan_indices]

            xlog = np.log10(np.array(aTs[i] * df['Angular frequency'].astype(np.float64)))
            dx = np.diff(xlog)

            if i == 1 or (min_x > np.min(x)):
                min_x = np.min(x)
            if i == 1 or (max_x < np.max(x)):    
                max_x = np.max(x)
            dx = np.mean(dx)  # Use the first element as the common spacing
            
            # Initialize the derivative array
            dy_dx = np.zeros_like(y)

            if SLOPE_ORDER == 1:
                # First order accuracy (Forward difference for the interior and backward for the last point)
                dy_dx[:-1] = (y[1:] - y[:-1]) / dx
                dy_dx[-1] = (y[-1] - y[-2]) / dx
            elif SLOPE_ORDER == 2:
                # Second order accuracy (Central difference, except for the first and last points)
                dy_dx[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
                dy_dx[0] = (y[1] - y[0]) / dx  # Forward difference for the first point
                dy_dx[-1] = (y[-1] - y[-2]) / dx  # Backward difference for the last point
            elif SLOPE_ORDER == 4:
                # Fourth order accuracy (Central difference, with special cases for the first two and last two points)
                dy_dx[2:-2] = (y[:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:]) / (12 * dx)
                dy_dx[0] = (y[1] - y[0]) / dx  # Forward difference for the first point
                dy_dx[1] = (y[2] - y[1]) / dx  # Forward difference for the second point
                dy_dx[-2] = (y[-1] - y[-2]) / dx  # Backward difference for the second-to-last point
                dy_dx[-1] = (y[-1] - y[-2]) / dx  # Backward difference for the last point
            else:
                raise ValueError("Accuracy order must be 1, 2, or 4.")

            if i == 1 or (min_dy > np.min(dy_dx)):
                min_dy = np.min(dy_dx)
            if i == 1 or (max_dy < np.max(dy_dx)):    
                max_dy = np.max(dy_dx)

            storage_marker["marker"] = symbol_array[j]
            loss_marker["marker"] = symbol_array[j]
            storage_marker["color"] = color_array[k]
            loss_marker["color"] = color_array[k]

            # print(x)
            # print(dy_dx)
            plt.plot(x, dy_dx, linewidth=1, **storage_marker)
            lgd.append(f"Slope plot of $G'$: Order ${SLOPE_ORDER}$")

            j = (j+1) % len(symbol_array)
            k = (k+1) % len(color_array)

    # plt.xlim(0.1 * min(min(aTs) * df['Storage modulus']), 10*max(max(aTs) * df['Storage modulus']))  # X-axis limits
    plt.xlim(1.0 * min_x, 1.0*max_x)
    plt.ylim(1.0 * min_dy, 1.0*max_dy)  # Y-axis limits
    plt.xlabel("Angular frequency, $\\omega$ [Pa]", fontsize=14)
    plt.ylabel("Slope, $G\'$", fontsize=14)
    # plt.ylabel("$ \\tan{\\delta} $", fontsize=14)
    # plt.title(f"{prefix + '_' + output_folder + '_' + filename.split('rep', 1)[0]} \n " f"$OSS(G')$: {sum(gp_artifacts):.2f}, $OSS(G'')$: {sum(gpp_artifacts):.2f}", fontsize=14)
    plt.title(f"{prefix + '_' + output_folder + '_' + filename.split('rep', 1)[0]}", fontsize=14)
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.show()

    if typ in ['raw','master', 'shifted']:
        lgd = ["$G':$ " + str(tref) + '°C', "$G'':$ " + str(tref) + '°C']
        gradient_image = plt.imshow(np.array([[np.min(temp_arr), np.max(temp_arr)]]), cmap=colormap, aspect='auto', norm=norm)
        gradient_image.set_visible(False)  # Hide the gradient image, show only colorbar
        cbar = plt.colorbar(gradient_image, orientation='horizontal', pad=0.15)
        cbar.set_label('Temperature')

    plt.legend(lgd, loc='upper right' , bbox_to_anchor=(1.35, 1))    
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir + 'figures', prefix + '_' + output_folder, prefix + '_' + 'SLOPE' + '_' + output_folder + '_' + filename  + '.png'), dpi=300, bbox_inches='tight')
    plt.close()

def shift_plots(dicts, prefix, temp_arr, filename='NO_NAME'):
    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + 'shift_factors')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots(figsize=(5.3, 4))
    markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'H']
    storage_marker = dict(marker='o', linestyle='-', markersize=5, color='r')
    loss_marker = dict(marker='o', linestyle='-', markersize=5, color='b')
    for i, (key, value) in enumerate(dicts.items()):
        ats, bts, dats, dbts = value
        storage_marker["marker"] = markers[i]
        loss_marker["marker"] = markers[i] 
        ax.set_yscale("log")
        # plt.errorbar(aTs[i] * df['Angular frequency'], bTs[i] * df['Storage modulus'], yerr=df['uGp'], capsize=3, linewidth=1, **storage_marker)
        # ax.errorbar(temp_arr, np.log10(ats), yerr=(dats), capsize=3, linewidth=1, label = 'Horizontal Shift, $log(a_T)$', **storage_marker)
        # ax.errorbar(temp_arr, np.log10(bts), yerr=(dbts), capsize=3, linewidth=1, label = 'Vertical Shift, $log(b_T)$', **loss_marker)
        ax.errorbar(temp_arr, ats, yerr=(dats), capsize=3, linewidth=1, label = 'Horizontal Shift, $a_T$', **storage_marker)
        ax.errorbar(temp_arr, bts, yerr=(dbts), capsize=3, linewidth=1, label = 'Vertical Shift, $b_T$', **loss_marker)
        
    # ax.set_title(f"{'shift_comparison' + '_' + filename}", fontsize=14)
    ax.set_xlabel("Temperature, $T$ [°C]", fontsize=14)
    ax.set_ylabel("Shift factors, ($a_T$, $b_T$)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14, direction='in', grid_color='k', grid_alpha=0.5, top=1, right=1, length=6, width=1)
    plt.tick_params(axis='both', which='minor', labelsize=14, direction='in', grid_color='k', grid_alpha=0.5, top=1, right=1, length=3, width=1)
    ax.grid(False)
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"shift_factors_{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def vgp_plotter(data_frames, temp_arr, nc, prefix, typ, bTs):

    # bts = np.ones_like(bts)
    i, j, k = 0, 0, 0
    lgd = []
    nc = nc - 1
    if SELECT:
        nc = len(temp_arr)
    output_folder= " "
    fig, ax = plt.subplots(figsize=(6,4))

    # region MARKER AND COLOR SETTINGS
    all_markers = list(MarkerStyle.markers.keys())
    symbol_array = all_markers[2:]

    colors = [(0, 0, 1), (0, 0.75, 0), (1, 0, 0)]  # R -> G -> B
    n_bins = 100  # Number of bins
    cmap_name = 'myRGB'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    colormap = cms.coolwarm
    norm = plt.Normalize(np.min(temp_arr), np.max(temp_arr))
    color_array = colormap(norm(temp_arr))
    storage_marker = dict(marker='.', linestyle='-', markersize=5, label='Filled Markers')
    loss_marker = dict(marker='.', linestyle='-', markersize=5, fillstyle='none', label='Empty Markers')
    #endregion

    # region PLOT TYPE SELECTION
    if typ == 'raw':
        aTs = np.ones_like(bTs)
        bTs = np.ones_like(bTs)
        output_folder= "raw_fs"
    elif typ == 'master':
        symbol_array[1:] = symbol_array[0]
        nc = len(temp_arr)
        output_folder= "master"
    elif typ == 'shifted':
        nc = len(temp_arr)
        output_folder= "shifted"
    else:
        print("Enter a valid plot type\n")
    # endregion 

    # region PLOT LIMIT CALCULATIONS
    all_storage_moduli = []
    all_loss_moduli = []
    all_frequencies = []
    a = 0

    for i, temp in enumerate(temp_arr):
        sheet_name = 'Frequency sweep (' + str(round(temp, 1)) + ' °C)'
        df = data_frames[sheet_name]
        all_storage_moduli.extend(df['Tan(delta)'])
        # all_loss_moduli.extend(bTs[a] * df['Loss modulus'])
        all_frequencies.extend(bTs[i] * df['Complex modulus'])
        a = a + 1

    min_storage = min(all_storage_moduli)
    max_storage = max(all_storage_moduli)
    min_freq = min(all_frequencies)
    max_freq = max(all_frequencies)
    # endregion

    if cf.LOG_PLOT==0:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    # Check if the directory exists before trying to create it
    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, temp in enumerate(temp_arr):
        sheet_name='Frequency sweep ('+str(round(temp_arr[i], 1)) + ' °C)'
        df = data_frames[sheet_name]
        # wls_arr[i] = fit_wls_and_get_residuals(df['Angular frequency'], df['Storage modulus'])
        x1 = df['Complex modulus'].astype(np.float64)
        y1 = df['Tan(delta)'].astype(np.float64)
        # y2 = df['Loss modulus'].astype(np.float64)
        
        if i % int(len(temp_arr) / nc) == 0 or i == len(temp_arr) - 1:

            storage_marker["marker"] = symbol_array[j]
            loss_marker["marker"] = symbol_array[j]
            storage_marker["color"] = color_array[i]
            loss_marker["color"] = color_array[i]

            # print(f'{temp} - {(df['uGp']/df['Storage modulus'])}')
            if LOG_PLOT:
                ax.errorbar(np.log10(x1), np.log10(y1), yerr=(1/2.303) * (df['uGp']/df['Storage modulus']), capsize=3, linewidth=1, **storage_marker)
                ax.errorbar(np.log10(x1), np.log10(y2), yerr=(1/2.303) * (df['uGpp']/df['Loss modulus']), capsize=3, linewidth=1, **loss_marker)  
                plt.axhline(y=np.log10(np.max(gmin*bTs)), color='r', linestyle='--', label='_nolegend_')
            else:
                ax.errorbar(x1 * bTs[i],  y1, xerr=df['uGpp']*0, capsize=3, linewidth=1, **storage_marker) # NEED TO CHANGE THIS
                # ax.errorbar(x1, y2, yerr=df['uGpp'], capsize=3, linewidth=1, **loss_marker)  
                plt.axvline(x=(np.max(gmin)), color='r', linestyle='--', label='_nolegend_')            

            j = (j+1) % len(symbol_array)
            k = (k+int(len(temp_arr) / nc) - 1) % len(color_array)
        
        lgd.append("$G':$ " + str(round(temp, 1)) + '°C')
        lgd.append("$G'':$ " + str(round(temp, 1)) +'°C')
    
    max_value = max(value for value in df['Complex modulus'] if not math.isnan(value))
    min_value = min(value for value in df['Complex modulus'] if not math.isnan(value))

    # print(bts)
    # ax.set_xlim(0.5*df['Complex modulus'].min() * min(bts), 5*df['Complex modulus'].max() * max(bts))
    ax.set_xlim(0.5*min_freq, 2*max_freq)
    ax.set_ylim(0.75 * min(y1), 2*max(y1))
    # ax.set_ylim( min_storage, 10 + max_storage)  # Y-axis limits
    ax.set_xlabel(r"Complex modulus, $G^*$ [Pa]", fontsize=14)
    ax.set_ylabel(r"$\tan{\delta}$", fontsize=14)

    # ax.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14, direction='in', grid_color='k', grid_alpha=0.5, top=1, right=1, length=6, width=1)
    plt.tick_params(axis='both', which='minor', labelsize=14, direction='in', grid_color='k', grid_alpha=0.5, top=1, right=1, length=3, width=1)
    
    gradient_image = plt.imshow(np.array([[np.min(temp_arr), np.max(temp_arr)]]), cmap=colormap, aspect='auto', norm=norm)
    gradient_image.set_visible(False)  # Hide the gradient image, show only colorbar
    cbar = plt.colorbar(gradient_image, orientation='vertical', pad=0.05, aspect=25)
    cbar.set_label('Temperature', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(parent_dir + 'figures', prefix + '_' + output_folder, prefix + '_' + output_folder + '_' + filename + '.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plotter1(data_frames, temp_arr, tref, prefix, typ, dict_shift, mastercurve_df, logged=0, x1_str='Angular frequency', y1_str='Complex modulus', y2_str='', gmin=0, filename='NO_NAME'):

    i, j, k = 0, 0, 0
    lgd = []
    output_folder= " "
    flg = cf.FLAG
    # print(mastercurve_df)
    
    for key, value in dict_shift.items():
        aTs, bTs, daTs, dbTs = value

    fig, ax, symbol_array, color_array, norm = set_plot_params(temp_arr)

    storage_marker = dict(marker='.', linestyle='-', label='Filled Markers')
    loss_marker = dict(marker='.', linestyle='-', fillstyle='none', label='Empty Markers')

    # region PLOT TYPE SELECTION
    if typ == 'raw':
        me = 1
        nc = len(temp_arr) / 1
        aTs = np.ones_like(aTs)
        bTs = np.ones_like(bTs)
        daTs = np.zeros_like(aTs)
        dbTs = np.zeros_like(bTs)
        output_folder= "raw_fs"
    elif typ == 'master':
        me = 2
        symbol_array[1:] = symbol_array[0]
        nc = len(temp_arr)
        output_folder= "master"
    elif typ == 'shifted':
        nc = len(temp_arr)
        output_folder= "shifted"
    else:
        print("Enter a valid plot type\n")
    # endregion 

    # region PLOT LIMIT CALCULATIONS
    all_y1 = []
    all_y2 = []
    all_x1 = []
    a = 0

    if (y1_str=='Tan(delta)') and (x1_str=='Complex modulus'):
        aTs = bTs
        bTs = np.ones_like(bTs)

    if (y1_str=='Loss modulus') and (x1_str=='Storage modulus'):
        aTs = bTs
        # bTs = np.ones_like(bTs)

    if y2_str == '':
        y2_str = y1_str
        flg = 0

    for temp in temp_arr:
        sheet_name = 'Frequency sweep (' + str(round(temp, 1)) + ' °C)'
        df = data_frames[sheet_name]
        x1_var =  df[x1_str].astype(np.float64)
        y1_var =  df[y1_str].astype(np.float64)
        y2_var =  df[y2_str].astype(np.float64)

        all_y1.extend(bTs[a] * y1_var)
        all_y2.extend(bTs[a] * y2_var)
        all_x1.extend(aTs[a] * x1_var)
        a = a + 1

    min_x1 = min(all_x1)
    max_x1 = max(all_x1)
    min_y1 = min(all_y1)
    max_y1 = max(all_y1)
    min_y2 = min(all_y2)
    max_y2 = max(all_y2)
    # endregion

    if not logged:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, temp in enumerate(temp_arr):
        # if (typ == 'master') and (i==2):
        #         break
        sheet_name='Frequency sweep ('+str(round(temp_arr[i], 1)) + ' °C)'
        df = data_frames[sheet_name]
        
        x1 = aTs[i] * df[x1_str].astype(np.float64)
        x1_err = daTs[i] * df[x1_str].astype(np.float64)
        y1 = bTs[i] * df[y1_str].astype(np.float64)
        y2 = bTs[i] * df[y2_str].astype(np.float64)

        if y1_str=='Storage modulus' or y1_str=='Loss modulus':
            _, y1_err =  tt.variance_product(bTs[i], dbTs[i], df[y1_str].astype(np.float64), df['u_'+y1_str].astype(np.float64), typ='analytical', num_samples=2000)
            _, y2_err = tt.variance_product(bTs[i], dbTs[i], df[y2_str].astype(np.float64), df['u_'+y2_str].astype(np.float64), typ='analytical', num_samples=2000) 

            # y1_err =  bTs[i] * df[y1_str].astype(np.float64) * np.sqrt((dbTs[i] / bTs[i])**2 + (df['u_'+y1_str].astype(np.float64)/ df[y1_str].astype(np.float64))**2)
            # y2_err =  bTs[i] * df[y2_str].astype(np.float64) * np.sqrt((dbTs[i] / bTs[i])**2 + (df['u_'+y2_str].astype(np.float64)/ df[y2_str].astype(np.float64))**2)
        else:
            if i == 0:
                print('Uncertainties are not available')
            y1_err = np.zeros_like(y1)
            y2_err = np.zeros_like(y2)
        
        if i % int(len(temp_arr) / nc) == 0 or i == len(temp_arr) - 1:

            storage_marker["marker"] = symbol_array[j]
            loss_marker["marker"] = symbol_array[j]
            storage_marker["color"] = color_array[i]
            loss_marker["color"] = color_array[i]

            if logged:
                ax.errorbar(np.log10(x1), np.log10(y1), yerr=(1/2.303) * (df['u_' + y1_str]/df[y1_str]), capsize=3, linewidth=1, alpha=0.5, **storage_marker)
                ax.errorbar(np.log10(x1), np.log10(y2), yerr=(1/2.303) * (df['u_' + y2_str]/df[y2_str]), capsize=3, linewidth=1, alpha=0.5, **loss_marker)  

                plt.axhline(y=np.log10(np.max(gmin*bTs)), color='r', linestyle='--', label='_nolegend_')
            else:
                # if typ == 'master':
                #     x1 = mastercurve_df['Angular frequency']
                #     y1 = mastercurve_df['Storage modulus']
                #     y2 = mastercurve_df['Loss modulus']
                #     x1_err = mastercurve_df['u_Angular frequency']
                #     y1_err = mastercurve_df['u_Storage modulus']
                #     y2_err = mastercurve_df['u_Loss modulus']

                #     ax.errorbar(x1, y1, yerr=0*y1_err, xerr=x1_err, capsize=3, linewidth=1, zorder=2, **storage_marker)
                #     # ax.fill_between(x1, y1 - y1_err, y1 + y1_err, alpha=0.2, color='blue', linewidth=1, zorder=1)
                #     ax.fill_betweenx(y1, x1 - x1_err, x1 + x1_err, alpha=0.2, color='red', linewidth=1, zorder=1)
                # else:
                ax.errorbar(x1[::me], y1[::me], yerr=y1_err[::me], xerr=x1_err[::me], capsize=3, linewidth=1, alpha=0.4, zorder=2, **storage_marker)
                ax.errorbar(x1[::me], y1[::me], yerr=y1_err[::me]*0, xerr=x1_err[::me]*0, capsize=3, linewidth=1, alpha=1.0, zorder=2, **storage_marker)
                # ax.plot(x1, y1, linewidth=1, zorder=1, linestyle='-', color=color_array[i])
                # ax.fill_between(x1, y1 - y1_err, y1 + y1_err, alpha=0.2, color='blue', linewidth=1, zorder=1)
                if flg:
                    ax.errorbar(x1[::me], y2[::me], yerr=y2_err[::me], xerr=x1_err[::me], capsize=3, linewidth=1, alpha=0.5, zorder=2, **loss_marker)
                    ax.errorbar(x1[::me], y2[::me], yerr=y2_err[::me]*0, xerr=x1_err[::me]*0, capsize=3, linewidth=1, alpha=1.0, zorder=2, **loss_marker)
                    # ax.plot(x1, y2, linewidth=1, zorder=1, linestyle='-', color=color_array[i])
                     
                # error_container = ax.plot(x1, y1, linewidth=1, zorder=1, **storage_marker)
                # error_container1 = ax.plot(x1, y2, linewidth=1, zorder=1, **loss_marker)  
                plt.axhline(y=(np.max(gmin*bTs)), color='r', linestyle='--', label='_nolegend_')           

            # lgd.append("$G':$ " + str(round(temp, 1)) + '°C')
            # lgd.append("$G'':$ " + str(round(temp, 1)) +'°C')

            j = (j+1) % len(symbol_array)
            k = (k+int(len(temp_arr) / nc) - 1) % len(color_array)

    # region PLOT BELLS AND WHISTLES
    if logged:
        max_value = np.log10(max(value for value in max(aTs)*df[x1_str] if not math.isnan(value)))
        min_value = np.log10(min(value for value in min(aTs)*df[x1_str] if not math.isnan(value)))

        ax.set_xlim(0.5 * min_x1, 3*max_x1)
        ax.set_ylim(0.5 * min(min_y1, min_y2), 5*max(max_y1, max_y2))  # Y-axis limits

        ax.set_xlabel(x1_str)
        ax.set_ylabel(y1_str)
    else:
        max_value = max(value for value in max(aTs)*df[x1_str] if not math.isnan(value))
        min_value = min(value for value in min(aTs)*df[x1_str] if not math.isnan(value))

        ax.set_xlim(0.5 * min_x1, 3*max_x1)
        ax.set_ylim(0.5 * min(min_y1, min_y2), 5*max(max_y1, max_y2))
        ax.set_xlabel(rheology_labels[x1_str])
        ax.set_ylabel(rheology_labels[y1_str])

    gradient_image = plt.imshow(np.array([[np.min(temp_arr), np.max(temp_arr)]]), cmap=cms.coolwarm, aspect='auto', norm=plt.Normalize(np.min(temp_arr), np.max(temp_arr)))
    gradient_image.set_visible(False) 
    cbar = plt.colorbar(gradient_image, orientation='vertical', pad=0.05, aspect=25)
    cbar.set_label('Temperature, $T$ [°C]', fontsize=14)
    if typ in ['master', 'shifted']:
        lgd = [y1_str]
        if flg:
            lgd = [y1_str, y2_str]
        ax.legend(lgd, title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C')
        
    else:
        lgd = [y1_str]
        if flg:
            lgd = [y1_str, y2_str]
        ax.legend(lgd, title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C')
    # endregion
        
    plt.tight_layout()
    fig.savefig(os.path.join(parent_dir + 'figures', prefix + '_' + output_folder, prefix + '_' + output_folder + '_' + filename + '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 

def lean_plotter(data_frames, temp_arr, tref, prefix, typ, dict_shift, mastercurve_df, crit_df, logged=0, x1_str='Angular frequency', y1_str='Complex modulus', y2_str='', tobolsky=0, filename='NO_NAME', gmin=0, plot_column_indices=None):

    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + 'yolo')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    raw_df = pd.concat(data_frames.values(), ignore_index=True)
    raw_df.sort_values(by='Temperature', inplace=True)
    
    fig, axes, symbol_array, color_array, norm = set_plot_params(temp_arr, lean_plot=1)
    
    if not logged:
        axes[0].set_xscale("log")             # RAW DATA
        axes[0].set_yscale("log")             # RAW DATA
        
        axes[1].set_xscale("log")             # MASTERCURVE
        axes[1].set_yscale("log")             # MASTERCURVE
        
        # axes[2].set_xscale("log")           # SHIFT FACTORS
        axes[2].set_yscale("log")             # SHIFT FACTORS
        
        axes[4].set_xscale("log")             # phase space
        # axes1.set_yscale("log")
     
    # region SHIFT FACTOR PLOTS
        
    storage_marker = dict(marker='o', linestyle='-', markersize=5, color='r')
    loss_marker = dict(marker='o', linestyle='-', markersize=5, color='b', fillstyle='none')
    
    colors = ['r', 'g', 'b', 'm', 'k', 'c']
    legend = ['$G\'(\omega)$', '$G^*(\omega)$', '$G\'\'(\omega)$', 'tan $\delta(\omega)$']
    # shift_legend = ['Horizontal Shift, $a_T$', 'Vertical Shift, $b_T$']
    shift_legend = []
    
    for i, (key, value) in enumerate(dict_shift.items()):
        aTs, bTs, daTs, dbTs = value
        storage_marker["marker"] = symbol_array[i % 8]
        storage_marker["color"] = colors[i+1]
        loss_marker["marker"] = symbol_array[i % 8]
        loss_marker["color"] = colors[i+1]
        
        axes[2].errorbar(temp_arr, aTs, yerr=(daTs), capsize=3, linewidth=1, **storage_marker)
        # axes[2].errorbar(temp_arr, bTs, yerr=(dbTs), capsize=3, linewidth=1, **loss_marker)
        # axes[2].errorbar(temp_arr[1:], aTs[1:]/aTs[:-1] , yerr=(daTs[1:] * 0), capsize=3, linewidth=1, label = 'Horizontal Shift, $a_T$', **storage_marker)
        shift_legend.append(key)

    storage_marker = dict(marker='.', linestyle='-')
    loss_marker = dict(marker='.', linestyle='-', fillstyle='none')
    phase_marker = dict(marker='o', markersize=5, linestyle=' ')
    
    # endregion

    omega_limits = np.zeros((len(temp_arr), 2))
    shifted_omega_limits = np.zeros((len(temp_arr), 2))
    
    for i, temp in enumerate(temp_arr):
        sheet_name='Frequency sweep ('+str(round(temp_arr[i], 1)) + ' °C)'
        if tobolsky:
            sheet_name='Relaxation: '+ str(round(temp_arr[i], 1)) + " °C"
        df = data_frames[sheet_name]
        
        if y1_str == 'None':
            df[y1_str] = np.nan
            df['u_'+y1_str] = np.nan
            mastercurve_df[y1_str] = np.nan
            mastercurve_df['u_'+y1_str] = np.nan
            raw_df[y1_str] = np.nan
            raw_df['u_'+y1_str] = np.nan
        if y2_str == 'None':
            df[y2_str] = np.nan
            df['u_'+y2_str] = np.nan
            mastercurve_df[y2_str] = np.nan
            mastercurve_df['u_'+y2_str] = np.nan
            raw_df[y2_str] = np.nan
            raw_df['u_'+y2_str] = np.nan
        
        storage_marker["marker"] = symbol_array[i % 8]
        loss_marker["marker"] = symbol_array[i % 8]
        storage_marker["color"] = color_array[i]
        loss_marker["color"] = color_array[i]
        phase_marker['color'] = color_array[i]
        
        omega_limits[i, :] = np.array([df[x1_str].min(), df[x1_str].max()])
        
        # PLOTTING RAW DATA
        me=1
        n_curves = 1
        if not (i + 1) % n_curves:
            axes[0].errorbar(
                df.loc[::me, x1_str], 
                df.loc[::me, y1_str], 
                yerr=df.loc[::me, 'u_'+y1_str], 
                xerr=0, 
                capsize=3, 
                linewidth=1, 
                alpha=1.0, 
                zorder=2, 
                **storage_marker,
                )
            axes[0].errorbar(
                df.loc[::me, x1_str], 
                df.loc[::me, y2_str], 
                yerr=df.loc[::me, 'u_'+y2_str], 
                xerr=0, 
                capsize=3, 
                linewidth=1, 
                alpha=1.0, 
                zorder=2, 
                **loss_marker,
                )
        
        # PLOTTING MASTERCURVE
        me = 1
        # np.abs(shift_sheet["Temperature"].astype(np.float64) - temp) <= 0.5
        short_df = mastercurve_df.loc[np.abs(mastercurve_df['Temperature'].astype(np.float64) - temp) <= 0.5]
        axes[1].errorbar(
            short_df.loc[::me, x1_str], 
            short_df.loc[::me, y1_str], 
            # yerr=tt.variance_product(
            #         bTs[i], dbTs[i], 
            #         df.loc[::me, y1_str].astype(np.float64), df.loc[::me, 'u_'+y1_str].astype(np.float64), 
            #         typ='analytical', num_samples=2000
            #         )[1], 
            yerr=short_df.loc[::me, 'u_'+y1_str],
            # xerr=daTs[i] * df.loc[::me, x1_str].astype(np.float64),
            xerr=short_df.loc[::me, 'u_'+x1_str],
            capsize=3, 
            linewidth=1, 
            alpha=1.0, 
            zorder=2, 
            **storage_marker,
            ) 
        axes[1].errorbar(
            short_df.loc[::me, x1_str], 
            short_df.loc[::me, y2_str], 
            # yerr=tt.variance_product(
            #         bTs[i], dbTs[i], 
            #         df.loc[::me, y2_str].astype(np.float64), df.loc[::me, 'u_'+y2_str].astype(np.float64), 
            #         typ='analytical', num_samples=2000
            #         )[1], 
            yerr= short_df.loc[::me, 'u_'+y2_str],
            # xerr=daTs[i] * df.loc[::me, x1_str].astype(np.float64), 
            xerr= short_df.loc[::me, 'u_'+x1_str],
            capsize=3,
            linewidth=1, 
            alpha=1.0, 
            zorder=2, 
            **loss_marker,
            )
        
        axes[1].xaxis.set_major_locator(LogLocator(base=10.0, numticks=7))  # 5 major ticks
        axes[1].xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))  # Minor ticks
        
        # PHASE SPACE - time/temperature
        axes[4].plot(df[x1_str], np.ones_like(df[x1_str].values)*temp, **phase_marker, zorder=2)
     
    # region RAW, MASTER AND SHIFT PLOTS
    if y1_str == 'None':
        moduli_min = raw_df[y2_str].min()
        moduli_max = raw_df[y2_str].max()
    elif y2_str == 'None':
        moduli_min = raw_df[y1_str].min()
        moduli_max = raw_df[y1_str].max()
    else:
        moduli_min = min(raw_df[y1_str].min(), raw_df[y2_str].min())
        moduli_max = max(raw_df[y1_str].max(), raw_df[y2_str].max())
    
    axes[0].set_xlim(0.2*raw_df[x1_str].min(), 3*raw_df[x1_str].max())
    axes[0].set_ylim(0.1*moduli_min, 5*moduli_max)
    axes[0].set_xlabel(rheology_labels[x1_str])
    axes[0].set_ylabel(rheology_labels[y1_str])
    axes[0].legend([y1_str, y2_str], title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C')
    
    gradient_image = axes[0].imshow(
                np.array([[np.min(temp_arr), np.max(temp_arr)]]), 
                cmap=cms.coolwarm, 
                aspect='auto', 
                norm=plt.Normalize(np.min(temp_arr), np.max(temp_arr)))
    gradient_image.set_visible(False)
    divider = make_axes_locatable(axes[0])
    cbar_ax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(gradient_image, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Temperature, $T$ [°C]', fontsize=14)
    
    if y1_str == 'None':
        moduli_min = mastercurve_df[y2_str].min()
        moduli_max = mastercurve_df[y2_str].max()
    elif y2_str == 'None':
        moduli_min = mastercurve_df[y1_str].min()
        moduli_max = mastercurve_df[y1_str].max()
    else:
        moduli_min = min(mastercurve_df[y1_str].min(), mastercurve_df[y2_str].min())
        moduli_max = max(mastercurve_df[y1_str].max(), mastercurve_df[y2_str].max())
    
    axes[1].set_xlim(0.3*mastercurve_df[x1_str].min(), 3*mastercurve_df[x1_str].max())
    axes[1].set_ylim(0.1*moduli_min, 5*moduli_max)
    axes[1].set_xlabel(rheology_labels['master '+x1_str])
    axes[1].set_ylabel(rheology_labels['master '+y1_str])
    axes[1].legend([y1_str, y2_str], title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C')
    
    axes[2].set_xlabel("Temperature, $T$ [°C]", fontsize=14)
    axes[2].set_ylabel("Horizontal Shift factors, $a_T$", fontsize=14)
    axes[2].legend(shift_legend, loc='best')
    
    gradient_image = axes[1].imshow(
                np.array([[np.min(temp_arr), np.max(temp_arr)]]), 
                cmap=cms.coolwarm, 
                aspect='auto', 
                norm=plt.Normalize(np.min(temp_arr), np.max(temp_arr)))
    gradient_image.set_visible(False)
    divider = make_axes_locatable(axes[1])
    cbar_ax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(gradient_image, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Temperature, $T$ [°C]', fontsize=14)
    
    # endregion
    
    # region TIME-TEMPERATURE MAP
    extra_length = np.zeros_like(temp_arr)
    for i, temp in enumerate(temp_arr):
        ats_iref = aTs / aTs[i]
        shifted_omega_limits[i, 0] = min(omega_limits[:, 0] * ats_iref)
        shifted_omega_limits[i, 1] = max(omega_limits[:, 1] * ats_iref)
        extra_length[i] = (np.log10(omega_limits[i, 1]) - np.log10(omega_limits[i, 0])) / (np.log10(shifted_omega_limits[i, 1]) - np.log10(shifted_omega_limits[i, 0]))
    
    axes[4].plot(mastercurve_df[x1_str], 
                     np.ones_like(mastercurve_df[x1_str].values) * tref, 
                     linestyle=' ', marker='o', color='green', zorder=1, label='Chosen reference')
    
    # axes[4].scatter(mastercurve_df[x1_str], 
    #                  np.ones_like(mastercurve_df[x1_str].values) * tref, 
    #                  c=np.sqrt((mastercurve_df['u_'+y1_str].values / mastercurve_df[y1_str].values)**2 + (mastercurve_df['u_'+x1_str].values / mastercurve_df[x1_str].values)**2),
    #                  cmap='viridis',  
    #                  marker='o', zorder=1, label='Chosen reference')
    
    axes[4].fill_betweenx(temp_arr, omega_limits[:, 0], omega_limits[:, 1], color='gray', alpha=0.1, zorder=2)
    # axes[4].text(0.24, 0.88, 
    #              r'$\text{A}_{t-T} = \frac{\textcolor{red}{\text{Area}}_\text{measured}}{\textcolor{green}{\text{Area}_\text{total}}}$', 
    #              fontsize=16, transform=axes[4].transAxes, ha='center')
    axes[4].text(0.37, 0.88, 
                 fr'$\text{{Extrapolation length}} = {np.mean(extra_length):.4g}$', 
                 fontsize=16, transform=axes[4].transAxes, ha='center')
    axes[4].fill_betweenx(temp_arr, shifted_omega_limits[:, 0], shifted_omega_limits[:, 1], color='green', alpha=0.1, zorder=1)
    axes[4].set_ylim(temp_arr[0] - 40, temp_arr[-1] + 40)
    axes[4].set_ylabel(rheology_labels['Temperature'])
    axes[4].set_xlabel(rheology_labels[x1_str])  
    # axes[4].legend(loc='upper right')
    axes[4].xaxis.set_major_locator(LogLocator(base=10.0, numticks=9))
    axes[4].xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10)) 
     
    # endregion
    
    # region RADIAL PLOT OF CRITERIA
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False).tolist()
    
    column_set1 = ["overlap_area",
               "cos_measure",
               "pearson",
               "frechet",
               "shift_variation",
                ]
    column_set2 = ["linearity",
               "overlap_length",
               "overlap_num",
               "overlap_index",
               "phase_area",
                ]
    
    values1, unc_values1 = tt.calculate_values_and_uncertainties(crit_df)
    
    values1 = values1.tolist()
    unc_values1 = unc_values1.tolist()
    
    axes[3].fill(angles+angles[:1], values1+values1[:1], color='blue', alpha=0.25, label='extent of evidence')
    axes[3].plot(angles+angles[:1], values1+values1[:1], marker='o', color='blue', markersize=6, linewidth=2)
    
    unc_values1 += unc_values1[:1]
    upper_bound = [v + u for v, u in zip(values1+values1[:1], unc_values1)]
    lower_bound = [v - u for v, u in zip(values1+values1[:1], unc_values1)]
    
    axes[3].fill_between(angles+angles[:1], lower_bound, upper_bound, color='gray', alpha=0.5)
    axes[3].set_xticks(angles)
    labels = [criteria_labels[column] for column in (column_set1+column_set2)]
    axes[3].set_xticklabels(labels, fontsize=16)
    axes[3].set_rlim(0, 1.0)
    axes[3].tick_params(pad=15)
    
    # endregion
    
    # region CRITERIA TABLE
    
    data = [
        (r'$\text{OA}_k$', 'Overlap Area'),
        (r'$R^2$', 'Coefficient of Determination'),
        (r'$\text{OR}_{\text{len}}$', 'Overlap Range- Length'),
        (r'$\text{OR}_{\text{num}}$', 'Overlap Range- Number of points'),
        (r'$\text{Cos}$', 'Cosine of slope difference'),
        (r'$\text{PCC}_k$', 'Pearson Correlation'),
        (r'$\text{OI}_k$', 'Multiple Curve Overlap'),
        (r'$d_{\text{Fr}}$', 'Frechet Distance'),
        (r'$\text{Std}(a_T)$', 'Std deviation of shift factors'),
        (r'$\text{A}_{t-T}$', 'Ratio of areas in the $t-T$ space')
        ]
    df = pd.DataFrame(data, columns=[r'\textbf{Criteria notation}', r'\textbf{Explanation}'])
    
    axes[5].axis('off')
    table = axes[5].table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    
    # endregion
    
    if cf.CHOOSE_PLOTS:
        axes[0].set_visible(False)
        axes[1].set_visible(True)
        axes[2].set_visible(False)
        axes[4].set_visible(True)
        axes[3].set_visible(False)
        axes[5].set_visible(False)
        
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(cf.OUTPUT_DIR + 'combined_figures', filename + '.png'), dpi=600, bbox_inches='tight')
    # plt.close()
    
def combined_plotter(data_frames, temp_arr, tref, prefix, typ, dict_shift, mastercurve_df, crit_df, logged=0, x1_str='Angular frequency', y1_str='Complex modulus', y2_str='', tobolsky=0, filename='NO_NAME', gmin=0, plot_column_indices=None):

    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + 'yolo')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    raw_df = pd.concat(data_frames.values(), ignore_index=True)
    raw_df.sort_values(by='Temperature', inplace=True)
    
    fig, axes, symbol_array, color_array, norm = set_plot_params(temp_arr, combined_plot=1)
    
    if not logged:
        axes[0].set_xscale("log")             # RAW DATA
        axes[0].set_yscale("log")             # RAW DATA
        
        axes[1].set_xscale("log")             # MASTERCURVE
        axes[1].set_yscale("log")             # MASTERCURVE
        
        # axes[2].set_xscale("log")           # SHIFT FACTORS
        axes[2].set_yscale("log")             # SHIFT FACTORS
        
        # axes[3].set_xscale("log")           # tTS CRITERIA
        # axes[3].set_yscale("log")           # tTS CRITERIA
        
        axes[4].set_xscale("log")             # phase space
        # axes1.set_yscale("log")
     
    # region SHIFT FACTOR PLOTS
        
    storage_marker = dict(marker='o', linestyle='-', markersize=5, color='r')
    loss_marker = dict(marker='o', linestyle='-', markersize=5, color='b', fillstyle='none')
    
    colors = ['r', 'g', 'b', 'm', 'k', 'c']
    legend = ['$G\'(\omega)$', '$G^*(\omega)$', '$G\'\'(\omega)$', 'tan $\delta(\omega)$']
    # shift_legend = ['Horizontal Shift, $a_T$', 'Vertical Shift, $b_T$']
    shift_legend = []
    
    for i, (key, value) in enumerate(dict_shift.items()):
        aTs, bTs, daTs, dbTs = value
        storage_marker["marker"] = symbol_array[i % 8]
        storage_marker["color"] = colors[i+1]
        loss_marker["marker"] = symbol_array[i % 8]
        loss_marker["color"] = colors[i+1]
        
        axes[2].errorbar(temp_arr, aTs, yerr=(daTs), capsize=3, linewidth=1, **storage_marker)
        # axes[2].errorbar(temp_arr, bTs, yerr=(dbTs), capsize=3, linewidth=1, **loss_marker)
        # axes[2].errorbar(temp_arr[1:], aTs[1:]/aTs[:-1] , yerr=(daTs[1:] * 0), capsize=3, linewidth=1, label = 'Horizontal Shift, $a_T$', **storage_marker)
        shift_legend.append(key)

    storage_marker = dict(marker='.', linestyle='-')
    loss_marker = dict(marker='.', linestyle='-', fillstyle='none')
    phase_marker = dict(marker='o', markersize=5, linestyle=' ')
    
    # endregion

    omega_limits = np.zeros((len(temp_arr), 2))
    shifted_omega_limits = np.zeros((len(temp_arr), 2))
    
    for i, temp in enumerate(temp_arr):
        sheet_name='Frequency sweep ('+str(round(temp_arr[i], 1)) + ' °C)'
        if tobolsky:
            sheet_name='Relaxation: '+ str(round(temp_arr[i], 1)) + " °C"
        df = data_frames[sheet_name]
        
        if y1_str == 'None':
            df[y1_str] = np.nan
            df['u_'+y1_str] = np.nan
            mastercurve_df[y1_str] = np.nan
            mastercurve_df['u_'+y1_str] = np.nan
            raw_df[y1_str] = np.nan
            raw_df['u_'+y1_str] = np.nan
        if y2_str == 'None':
            df[y2_str] = np.nan
            df['u_'+y2_str] = np.nan
            mastercurve_df[y2_str] = np.nan
            mastercurve_df['u_'+y2_str] = np.nan
            raw_df[y2_str] = np.nan
            raw_df['u_'+y2_str] = np.nan
        
        storage_marker["marker"] = symbol_array[i % 8]
        loss_marker["marker"] = symbol_array[i % 8]
        storage_marker["color"] = color_array[i]
        loss_marker["color"] = color_array[i]
        phase_marker['color'] = color_array[i]
        
        omega_limits[i, :] = np.array([df[x1_str].min(), df[x1_str].max()])
        
        # PLOTTING RAW DATA
        me=1
        n_curves = 1
        if not (i + 1) % n_curves:
            axes[0].errorbar(
                df.loc[::me, x1_str], 
                df.loc[::me, y1_str], 
                yerr=df.loc[::me, 'u_'+y1_str], 
                xerr=0, 
                capsize=3, 
                linewidth=1, 
                alpha=1.0, 
                zorder=2, 
                **storage_marker,
                )
            axes[0].errorbar(
                df.loc[::me, x1_str], 
                df.loc[::me, y2_str], 
                yerr=df.loc[::me, 'u_'+y2_str], 
                xerr=0, 
                capsize=3, 
                linewidth=1, 
                alpha=1.0, 
                zorder=2, 
                **loss_marker,
                )
        
        # PLOTTING MASTERCURVE
        me = 1
        # np.abs(shift_sheet["Temperature"].astype(np.float64) - temp) <= 0.5
        short_df = mastercurve_df.loc[np.abs(mastercurve_df['Temperature'].astype(np.float64) - temp) <= 0.5]
        axes[1].errorbar(
            short_df.loc[::me, x1_str], 
            short_df.loc[::me, y1_str], 
            # yerr=tt.variance_product(
            #         bTs[i], dbTs[i], 
            #         df.loc[::me, y1_str].astype(np.float64), df.loc[::me, 'u_'+y1_str].astype(np.float64), 
            #         typ='analytical', num_samples=2000
            #         )[1], 
            yerr=short_df.loc[::me, 'u_'+y1_str],
            # xerr=daTs[i] * df.loc[::me, x1_str].astype(np.float64),
            xerr=short_df.loc[::me, 'u_'+x1_str],
            capsize=3, 
            linewidth=1, 
            alpha=1.0, 
            zorder=2, 
            **storage_marker,
            ) 
        axes[1].errorbar(
            short_df.loc[::me, x1_str], 
            short_df.loc[::me, y2_str], 
            # yerr=tt.variance_product(
            #         bTs[i], dbTs[i], 
            #         df.loc[::me, y2_str].astype(np.float64), df.loc[::me, 'u_'+y2_str].astype(np.float64), 
            #         typ='analytical', num_samples=2000
            #         )[1], 
            yerr= short_df.loc[::me, 'u_'+y2_str],
            # xerr=daTs[i] * df.loc[::me, x1_str].astype(np.float64), 
            xerr= short_df.loc[::me, 'u_'+x1_str],
            capsize=3,
            linewidth=1, 
            alpha=1.0, 
            zorder=2, 
            **loss_marker,
            )
        
        axes[1].xaxis.set_major_locator(LogLocator(base=10.0, numticks=7))  # 5 major ticks
        axes[1].xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))  # Minor ticks
        
        # inset_ax = fig.add_axes([0.53, 0.755, 0.08, 0.08])
        # inset_ax = inset_axes(axes[1], 
        #               width="40%",  # Slightly smaller width for padding
        #               height="40%",  # Slightly smaller height for padding
        #               loc="lower right",
        #               )
        # inset_ax = axes[1].inset_axes([0.55, 0.1, 0.4, 0.4])
        # # inset_ax = fig.add_axes([0.51, 0.755, 0.1, 0.1])
        # inset_ax.set_yscale("log") 
        # inset_ax.plot(temp_arr, aTs, linewidth=1, label = 'Horizontal Shift, $a_T$', **storage_marker)
        # inset_ax.plot(temp_arr, bTs, linewidth=1, label = 'Vertical Shift, $b_T$', **loss_marker)
        # # inset_ax.set_title('Shift factors', fontsize=12)
        # inset_ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        # inset_ax.tick_params(axis='x', which='major', labelsize=14, pad=6) 
        # inset_ax.tick_params(axis='y', which='major', labelsize=12, pad=6)
        # # inset_ax.set_xlabel("Temperature, $T$ [°C]", fontsize=12)
        # inset_ax.set_ylabel("($a_T$, $b_T$)", fontsize=14)
        
        # PHASE SPACE - time/temperature
        axes[4].plot(df[x1_str], np.ones_like(df[x1_str].values)*temp, **phase_marker, zorder=2)
    
        # PLOTTING CRITERIA
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H']
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(10)]
        storage_marker = dict(marker='o', linestyle='-')
        loss_marker = dict(marker='o', linestyle='-', fillstyle='none')
        if plot_column_indices is None:
            plot_column_indices = np.arange(len(crit_df.columns))[1:]
        all_columns = crit_df.columns[1:10]
        # selected_columns = [crit_df.columns[i] for i in plot_column_indices]
        selected_columns = cf.SELECTED_COLS
        # normalize_columns = [crit_df.columns[i] for i in [100, 101]]
        normalize_columns = all_columns
        worst_least_columns = all_columns
        legend_labels1 = []
        legend_labels2 = []
        values1 = np.zeros(10) 
        unc_values1 = np.zeros(10) 
        sums_cols = ['overlap_area', 'cos_measure','frechet']

        for i, column in enumerate(all_columns):
            loss_marker["marker"] = markers[i % 8]
            loss_marker["color"] = colors[i]
            
            if column in selected_columns:
                if column in normalize_columns:
                    y_err = crit_df['u_' + column].abs() / crit_df[column].abs().max()
                    y_values = crit_df[column]/ crit_df[column].abs().max()
                else:
                    y_err = crit_df['u_' + column]
                    y_values = crit_df[column]
                  
                if column in worst_least_columns:  
                    legend_labels1.append(criteria_labels[column] + f': {crit_df[column].abs().min():.3f}')
                else:
                    legend_labels1.append(criteria_labels[column] + f': {crit_df[column].abs().max():.3f}')

                if cf.DATA_UNCERT_TOGGLE or cf.SHIFT_UNCERT_TOGGLE:
                    axes[3].errorbar(crit_df['Temperature'], y_values, yerr=y_err, capsize=5, linewidth=2, **loss_marker)
                else:
                    axes[3].plot(crit_df['Temperature'], y_values, color=loss_marker["color"], linewidth=2)
            else:
                if column in normalize_columns:
                    y_err = crit_df['u_' + column] / crit_df[column].abs().max()
                    y_values = crit_df[column] / crit_df[column].abs().max()
                else:
                    y_err = crit_df['u_' + column]
                    y_values = crit_df[column]
                  
                if column in worst_least_columns:  
                    legend_labels2.append(criteria_labels[column] + f': {crit_df[column].abs().min():.3f}')
                else:
                    legend_labels2.append(criteria_labels[column] + f': {crit_df[column].abs().max():.3f}')

                if cf.DATA_UNCERT_TOGGLE or cf.SHIFT_UNCERT_TOGGLE:
                    axes[5].errorbar(crit_df['Temperature'], y_values, yerr=y_err, capsize=5, linewidth=2, **loss_marker)
                else:
                    axes[5].plot(crit_df['Temperature'], y_values, color=loss_marker["color"], linewidth=2)
     
    # region RAW, MASTER AND SHIFT PLOTS
    if y1_str == 'None':
        moduli_min = raw_df[y2_str].min()
        moduli_max = raw_df[y2_str].max()
    elif y2_str == 'None':
        moduli_min = raw_df[y1_str].min()
        moduli_max = raw_df[y1_str].max()
    else:
        moduli_min = min(raw_df[y1_str].min(), raw_df[y2_str].min())
        moduli_max = max(raw_df[y1_str].max(), raw_df[y2_str].max())
    
    axes[0].set_xlim(0.2*raw_df[x1_str].min(), 3*raw_df[x1_str].max())
    axes[0].set_ylim(0.1*moduli_min, 5*moduli_max)
    axes[0].set_xlabel(rheology_labels[x1_str])
    axes[0].set_ylabel(rheology_labels[y1_str])
    axes[0].legend([y1_str, y2_str], title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C')
    
    gradient_image = axes[0].imshow(
                np.array([[np.min(temp_arr), np.max(temp_arr)]]), 
                cmap=cms.coolwarm, 
                aspect='auto', 
                norm=plt.Normalize(np.min(temp_arr), np.max(temp_arr)))
    gradient_image.set_visible(False)
    divider = make_axes_locatable(axes[0])
    cbar_ax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = fig.colorbar(gradient_image, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Temperature, $T$ [°C]', fontsize=14)
    
    if y1_str == 'None':
        moduli_min = mastercurve_df[y2_str].min()
        moduli_max = mastercurve_df[y2_str].max()
    elif y2_str == 'None':
        moduli_min = mastercurve_df[y1_str].min()
        moduli_max = mastercurve_df[y1_str].max()
    else:
        moduli_min = min(mastercurve_df[y1_str].min(), mastercurve_df[y2_str].min())
        moduli_max = max(mastercurve_df[y1_str].max(), mastercurve_df[y2_str].max())
    
    axes[1].set_xlim(0.3*mastercurve_df[x1_str].min(), 3*mastercurve_df[x1_str].max())
    axes[1].set_ylim(0.1*moduli_min, 5*moduli_max)
    axes[1].set_xlabel(rheology_labels['master '+x1_str])
    axes[1].set_ylabel(rheology_labels['master '+y1_str])
    axes[1].legend([y1_str, y2_str], title=f"$T_{{\\mathrm{{ref}}}} = {tref}$"+ '°C')
    
    axes[2].set_xlabel("Temperature, $T$ [°C]", fontsize=14)
    axes[2].set_ylabel("Horizontal Shift factors, $a_T$", fontsize=14)
    axes[2].legend(shift_legend, loc='best')
    
    gradient_image = axes[1].imshow(
                np.array([[np.min(temp_arr), np.max(temp_arr)]]), 
                cmap=cms.coolwarm, 
                aspect='auto', 
                norm=plt.Normalize(np.min(temp_arr), np.max(temp_arr)))
    gradient_image.set_visible(False)
    divider = make_axes_locatable(axes[1])
    cbar_ax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = fig.colorbar(gradient_image, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Temperature, $T$ [°C]', fontsize=14)
    
    # endregion
    
    # region CRITERIA - TEMPERATURE PLOTS
    
    axes[3].set_xlim(temp_arr[0] - (temp_arr[1]-temp_arr[0]), temp_arr[-1])
    axes[3].set_ylim(-0.05, max(1.05, crit_df['shift_variation'].max()))
    axes[3].set_xlabel(rheology_labels['Temperature'])
    axes[3].set_ylabel("Criteria (/Max)")
    axes[3].grid(False)
    axes[3].legend(legend_labels1, title="Worst values",loc="lower right")
    
    axes[5].set_xlim(temp_arr[0] - (temp_arr[1]-temp_arr[0]), temp_arr[-1])
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].set_xlabel(rheology_labels['Temperature'])
    axes[5].set_ylabel("Criteria (/Max)")
    axes[5].grid(False)
    axes[5].legend(legend_labels2, title="Worst values",loc="lower right")
    
    # endregion
    
    # region TIME-TEMPERATURE MAP
    extra_length = np.zeros_like(temp_arr)
    for i, temp in enumerate(temp_arr):
        ats_iref = aTs / aTs[i]
        shifted_omega_limits[i, 0] = min(omega_limits[:, 0] * ats_iref)
        shifted_omega_limits[i, 1] = max(omega_limits[:, 1] * ats_iref)
        extra_length[i] = (np.log10(omega_limits[i, 1]) - np.log10(omega_limits[i, 0])) / (np.log10(shifted_omega_limits[i, 1]) - np.log10(shifted_omega_limits[i, 0]))
    
    axes[4].plot(mastercurve_df[x1_str], 
                     np.ones_like(mastercurve_df[x1_str].values) * tref, 
                     linestyle=' ', marker='o', color='green', zorder=1, label='Chosen reference')
    
    # axes[4].scatter(mastercurve_df[x1_str], 
    #                  np.ones_like(mastercurve_df[x1_str].values) * tref, 
    #                  c=np.sqrt((mastercurve_df['u_'+y1_str].values / mastercurve_df[y1_str].values)**2 + (mastercurve_df['u_'+x1_str].values / mastercurve_df[x1_str].values)**2),
    #                  cmap='viridis',  
    #                  marker='o', zorder=1, label='Chosen reference')
    
    axes[4].fill_betweenx(temp_arr, omega_limits[:, 0], omega_limits[:, 1], color='gray', alpha=0.1, zorder=2)
    # axes[4].text(0.24, 0.88, 
    #              r'$\text{A}_{t-T} = \frac{\textcolor{red}{\text{Area}}_\text{measured}}{\textcolor{green}{\text{Area}_\text{total}}}$', 
    #              fontsize=16, transform=axes[4].transAxes, ha='center')
    axes[4].text(0.37, 0.88, 
                 fr'$\text{{Extrapolation length}} = {np.mean(extra_length):.4g}$', 
                 fontsize=16, transform=axes[4].transAxes, ha='center')
    axes[4].fill_betweenx(temp_arr, shifted_omega_limits[:, 0], shifted_omega_limits[:, 1], color='green', alpha=0.1, zorder=1)
    axes[4].set_ylim(temp_arr[0] - 40, temp_arr[-1] + 40)
    axes[4].set_ylabel(rheology_labels['Temperature'])
    axes[4].set_xlabel(rheology_labels[x1_str])  
    # axes[4].legend(loc='upper right')
    axes[4].xaxis.set_major_locator(LogLocator(base=10.0, numticks=9))
    axes[4].xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10)) 
     
    # endregion
    
    # region RADIAL PLOT OF CRITERIA

    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False).tolist()
    values1 = values1.tolist()
    unc_values1 = unc_values1.tolist()
    
    column_set1 = ["overlap_area",
               "cos_measure",
               "pearson",
               "frechet",
               "shift_variation",
                ]
    column_set2 = ["linearity",
               "overlap_length",
               "overlap_num",
               "overlap_index",
               "phase_area",
                ]
    # for i, column in enumerate(column_set1 + column_set2[:-1]):
    #     Nc = len(crit_df[column]) - 1
    #     if column in sums_cols:
    #         # values1[i] = crit_df[column].abs().sum()
    #         values1[0] = np.tanh(1 / crit_df["overlap_area"].abs().sum())
    #         values1[3] = np.tanh(1 / crit_df["frechet"].abs().sum())
    #         # values1[3] = np.exp(- crit_df["frechet"].abs().sum())
    #         values1[1] = (np.cos(crit_df["cos_measure"].sum()) + 1) * 0.5
            
    #         unc_values1[0] = (1 - values1[0]**2) * (1 / crit_df["overlap_area"].abs().sum()**2) * np.sqrt(np.sum(crit_df['u_' + "overlap_area"].values.astype(np.float64)**2))
    #         unc_values1[3] = (1 - values1[3]**2) * (1 / crit_df["frechet"].abs().sum()**2) * np.sqrt(np.sum(crit_df['u_' + "frechet"].values.astype(np.float64)**2))
    #         unc_values1[1] = 0.5 * np.sin(crit_df["cos_measure"].abs().sum()) * np.sqrt(np.sum(crit_df['u_' + "cos_measure"].values.astype(np.float64)**2))
    #     else:
    #         values1[i] = crit_df[column].mean()
    #         unc_values1[i] = np.sqrt(np.sum((crit_df['u_' + column].values.astype(np.float64))**2)) / Nc
    #         if (column=='shift_variation') and (crit_df[column].mean() < 0):
    #             values1[i] = 0
    # values1[-1] = tt.phase_space_crit(np.array(temp_arr), omega_limits, shifted_omega_limits)
    
    values1, unc_values1 = tt.calculate_values_and_uncertainties(crit_df)
    
    values1 = values1.tolist()
    unc_values1 = unc_values1.tolist()
    
    # for label, value, unc_value in zip(column_set1 + column_set2[:-1], values1, unc_values1):
    #     print(f"{label}: {value} with unc: {unc_value}")
       
    axes[6].axis('off')
    
    axes[7].fill(angles+angles[:1], values1+values1[:1], color='blue', alpha=0.25, label='extent of evidence')
    axes[7].plot(angles+angles[:1], values1+values1[:1], marker='o', color='blue', markersize=6, linewidth=2)
    
    unc_values1 += unc_values1[:1]
    upper_bound = [v + u for v, u in zip(values1+values1[:1], unc_values1)]
    lower_bound = [v - u for v, u in zip(values1+values1[:1], unc_values1)]
    
    axes[7].fill_between(angles+angles[:1], lower_bound, upper_bound, color='gray', alpha=0.5)
    axes[7].set_xticks(angles)
    labels = [criteria_labels[column] for column in (column_set1+column_set2)]
    axes[7].set_xticklabels(labels, fontsize=16)
    axes[7].set_rlim(0, 1.0)
    axes[7].tick_params(pad=15)
    
    # endregion
    
    # region CRITERIA TABLE
    
    data = [
        (r'$\text{OA}_k$', 'Overlap Area'),
        (r'$R^2$', 'Coefficient of Determination'),
        (r'$\text{OR}_{\text{len}}$', 'Overlap Range- Length'),
        (r'$\text{OR}_{\text{num}}$', 'Overlap Range- Number of points'),
        (r'$\text{Cos}$', 'Cosine of slope difference'),
        (r'$\text{PCC}_k$', 'Pearson Correlation'),
        (r'$\text{OI}_k$', 'Multiple Curve Overlap'),
        (r'$d_{\text{Fr}}$', 'Frechet Distance'),
        (r'$\text{Std}(a_T)$', 'Std deviation of shift factors'),
        (r'$\text{A}_{t-T}$', 'Ratio of areas in the $t-T$ space')
        ]
    df = pd.DataFrame(data, columns=[r'\textbf{Criteria notation}', r'\textbf{Explanation}'])
    
    axes[8].axis('off')
    table = axes[8].table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    
    # endregion
    
    if cf.CHOOSE_PLOTS:
        axes[0].set_visible(True)
        axes[1].set_visible(True)
        axes[2].set_visible(True)
        axes[4].set_visible(True)
        axes[3].set_visible(False)
        axes[5].set_visible(False)
        axes[6].set_visible(True)
        axes[7].set_visible(True)
        
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(cf.OUTPUT_DIR + 'combined_figures', filename + '.png'), dpi=600, bbox_inches='tight')
    # plt.close()

def sampling_convergence(samp_list, df_samples, columns_of_interest, filename):
    norms = []
    for i in range(len(df_samples) - 1):
                df_current = np.abs(np.array(df_samples[i][columns_of_interest][:-2]))
                df_next = np.abs(np.array(df_samples[i+1][columns_of_interest][:-2])) 
                
                # Calculate the Frobenius norm of the difference
                norm = np.linalg.norm((df_current - df_next)/df_current, 'fro') * 100
                # norm = np.linalg.norm(df_current - df_next, 'fro') 
                norm2 = np.linalg.norm(df_current, 'fro') 
                print(f'This is the diff: {norm}')
                print(f'This is the current: {norm2}')
                norms.append(norm)

    fig, ax, symbol_array, color_array, norm = set_plot_params(
        2 * np.ones((50,)), model_fit=1
    )
    ax.loglog(samp_list[:-1], norms, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Number of samples', fontsize=14)
    ax.set_ylabel(r'Relative Norm: $\|\frac{u_{n+1} - u_n}{u_n}\| \times 10^2 \; $', fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(f"tts_convergence_{filename}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def uprop_lin_validation(master_df_analytical, temp_arr, data_frames, dicts, filename):
    
    fig, ax, symbol_array, color_array, norm = set_plot_params(
        2 * np.ones((50,)), model_fit=1
    )
    for num in [10, 100, 500, 1000, 2000, 2500]:
        master_df = tt.mastercurve_df(temp_arr, data_frames, dicts, typ='anal', num_samples=num)
        ax[0].semilogx(num, np.linalg.norm(np.mean(np.abs(master_df['u_Storage modulus'] - master_df_analytical['u_Storage modulus'])/ master_df_analytical['u_Storage modulus'])), marker='o', linestyle='-', color='b')
    ax[0].set_xlim([5, 3000])
    # ax.set_ylim([0, 1])
    ax[0].set_xlabel('Number of samples, $N$', fontsize=14)
    ax[0].set_ylabel(r'Relative Error: $1 - \frac{u_{\text{sampling}}}{u_{\text{analytical}}}$', fontsize=14)
    ax[0].grid(True)
    plt.tight_layout()
    fig.savefig(f"figures/convergence/uprop_verif_{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

def fit_prony_series(master_df, filename, spectra=0):
    
    os.chdir('/Users/asm18/Documents/python_repo/pyReSpect-freq')
    par = cs.readInput('/Users/asm18/Documents/python_repo/pyReSpect-freq/inp.dat')
    
    
    s, w, H, dH, dH_wls, G0, lamC, K, n = cs.getContSpec(par) 
    H = np.exp(H)
    dH = H * dH
    dH_wls = H * dH_wls
    gp = K[:n]
    gpp = K[n:]
    
    cov_matrix = np.loadtxt('/Users/asm18/Documents/python_repo/pyReSpect-freq/output/cov.dat', skiprows=1)
    
    # Hdata = np.loadtxt('/Users/asm18/Documents/python_repo/pyReSpect-freq/output/H.dat')
    # gfit = np.loadtxt('/Users/asm18/Documents/python_repo/pyReSpect-freq/output/Gfit.dat')
    
    # s       = Hdata[:, 0]
    # H       = Hdata[:, 1]
    # dH      = Hdata[:, 2]
    # dH_wls  = Hdata[:, 3]
    
    # w       = gfit[:, 0]
    # gp      = gfit[:, 1]
    # gpp     = gfit[:, 2]
    
    # H  = np.exp(H)
    # dH = H * dH
    # dH_wls = H * dH_wls
    
    # print(H)
    # print(dH_wls)
    
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    # corr_matrix = np.clip(corr_matrix, -1, 1)
    
    os.chdir('/Users/asm18/Documents/python_repo/time_temperature')
    fig, axes, *_ = set_plot_params(model_fit2=1)
    # fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    lw=2.0
    axes[0].plot(s, H, marker='o', color='k', label='Inferred Spectrum')
    axes[0].plot(s, H + dH, linestyle='-', linewidth=lw, color='blue', alpha=0.5, label=r'Error ($\lambda$)')
    axes[0].plot(s, H - dH, linestyle='-', linewidth=lw, color='blue', alpha=0.5)
    axes[0].plot(s, H + dH_wls, linestyle='-', linewidth=lw, color='red', alpha=0.5, label='Error (WLS)')
    axes[0].plot(s, H - dH_wls, linestyle='-', linewidth=lw, color='red', alpha=0.5)
    axes[0].axvline(x=np.exp(np.pi / 2)/np.max(w), linestyle='--', linewidth=lw, color='gray', label='D-A limits')
    axes[0].axvline(x=np.exp(-np.pi / 2)/ np.min(w), linestyle='--', linewidth=lw, color='gray')
    
    if spectra:
        axes[0].plot(spectra[1], spectra[0], linestyle='-', linewidth=lw, color='gray', label='Initial Spectrum')
    
    # axes[0].set_xlim(1E-5, 1E+6)
    # axes[0].set_ylim(1E+1, 7E+9)
    
    axes[0].set_xlabel(r'Relaxation times, $\tau_i$ [s]')
    axes[0].set_ylabel(r'Relaxation strength, $H_i$ [Pa]')
    axes[0].legend(loc='best')
    

    me=8
    axes[1].plot(w, gp, linestyle='-', linewidth=lw, color='red', alpha=1.0)
    axes[1].plot(w, gpp, linestyle='--', linewidth=lw, color='red', alpha=1.0)
    axes[1].plot(master_df['Angular frequency'][::me], master_df['Storage modulus'][::me], linestyle=' ', marker='o', color='k', label=r'Storage modulus')
    axes[1].plot(master_df['Angular frequency'][::me], master_df['Loss modulus'][::me], linestyle=' ',marker='o', color='k', fillstyle='none', label=r'Loss modulus')
    axes[1].set_xlabel('Reduced Angular frequency, $\omega_r$ [rad/s]')
    axes[1].set_ylabel('Reduced Moduli, $G\'_r, G\'\'_r$ [Pa]')
    axes[1].legend(loc='best')
    
    sns.heatmap(corr_matrix, cmap='coolwarm', cbar=True, vmin=-1, vmax=1, annot=False, ax=axes[2])
    cbar = axes[2].collections[0].colorbar
    # cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    axes[2].set_title('Correlation Parameter Matrix')
    axes[2].set_xlabel(r'Modes, $H_i$ [-]')
    axes[2].set_ylabel(r'Modes, $H_i$ [-]')
    
    # plt.show()
    plt.subplots_adjust(wspace=0.8)
    plt.tight_layout()
    fig.savefig(f"/Users/asm18/Documents/python_repo/time_temperature/tts_pronyfit_{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    # os.chdir('/Users/asm18/Documents/python_repo/time_temperature')

def complexity_crit(prefix, normalized_data_dict, xcol, plot_column_indices=None, filename='NO_NAME'):
    lgd = []
    first = True
    output_dir = os.path.join(parent_dir + 'figures', prefix + '_' + 'tts_criteria')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H']
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    fig, ax, *_ = set_plot_params()
    storage_marker = dict(marker='o', linestyle='-')
    loss_marker = dict(marker='o', linestyle='-', fillstyle='none')
    x_values = xcol  # Extracting the x-axis values from the DataFrame
    if plot_column_indices is None:
        plot_column_indices = np.arange(len(normalized_data_dict.columns))[1:]
    all_columns = normalized_data_dict.columns[1:9]
    selected_columns = [normalized_data_dict.columns[i] for i in plot_column_indices]
    i = 0

    normalized_df = normalized_data_dict
    for column in all_columns:
        if column in normalized_df.columns:
            loss_marker["marker"] = markers[i]
            loss_marker["color"] = colors[i]
            
            if column == 'Shift variation':
                ax.plot(x_values, normalized_df[column], linewidth=2, alpha=1.0, **loss_marker)
                print('Shift variation is working')
            else:
                if column in selected_columns:
                    err_col = normalized_data_dict['u_' + column] / normalized_data_dict[column].abs().max()
                    y_values = normalized_df[column]/ normalized_data_dict[column].abs().max()
                    ax.errorbar(x_values, y_values, yerr= err_col, capsize=5, linewidth=2, **loss_marker)
                else:
                    err_col = normalized_data_dict['u_' + column] / normalized_data_dict[column].abs().max()
                    y_values = normalized_df[column]/ normalized_data_dict[column].abs().max()
                    ax.errorbar(x_values, y_values, yerr= err_col, capsize=5, linewidth=2, alpha=0.2, **loss_marker)
            lgd.append(f"{column}")
            i = i + 1

    ax.set_title('Average tTS criteria across complexity')
    ax.set_xlabel("Complexity based on $C_2$")
    ax.set_ylabel("Absolute criteria values")
    ax.grid(False)
    ax.legend(lgd, loc='upper left', bbox_to_anchor=(1, 1), title="Average values")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"tts_criteria_icomplexity_{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()