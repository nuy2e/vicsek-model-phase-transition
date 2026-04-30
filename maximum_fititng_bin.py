"""
Susceptibility Peak Analyzer (Window-Focused)

Reads Vicsek model data from a CSV, entirely isolates a specified noise window, 
groups the runs within that window into discrete bins to reduce statistical noise, 
fits a parabola, and calculates the exact critical noise point (eta_c) with 
propagated uncertainties.

Author: Min Ki Hong
Date: April 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# Mathematical Model
# =============================================================================

def parabola(x, a, b, c):
    """Quadratic function for fitting the susceptibility peak."""
    return a * x**2 + b * x + c

# =============================================================================
# Data Processing (Strictly Windowed)
# =============================================================================

def load_and_bin_data(filename, target_n, num_buckets, window_min, window_max):
    """
    Loads raw simulation data, discards anything outside the window, 
    and bins the remaining data to crush statistical noise.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: Could not find '{filename}'. Run the simulation first!")

    df = pd.read_csv(filename)
    df_n = df[df['N'] == target_n].copy()
    
    if df_n.empty:
        raise ValueError(f"Error: No data found for N = {target_n} in {filename}")

    # 1. APPLY WINDOW FIRST: Drop all data outside the target region
    mask = (df_n['eta'] >= window_min) & (df_n['eta'] <= window_max)
    df_win = df_n[mask].copy()

    if df_win.empty:
        raise ValueError(f"Error: No data found in the window [{window_min}, {window_max}]")

    # 2. CREATE EXACT BINS: Force buckets to span exactly from window_min to window_max
    exact_bins = np.linspace(window_min, window_max, num_buckets + 1)
    df_win['eta_bucket'] = pd.cut(df_win['eta'], bins=exact_bins)

    # 3. GROUP DATA
    binned_df = df_win.groupby('eta_bucket', observed=False).agg(
        eta=('eta', 'mean'),                 
        chi_grand_mean=('chi_mean', 'mean'), 
        chi_std=('chi_mean', 'std'),         
        count=('chi_mean', 'count')          
    ).reset_index()

    binned_df = binned_df.dropna(subset=['chi_grand_mean'])

    # 4. CALCULATE ERRORS (Standard Error of the Mean)
    binned_df['chi_grand_error'] = binned_df['chi_std'] / np.sqrt(binned_df['count'])
    binned_df['chi_grand_error'] = binned_df['chi_grand_error'].fillna(0)

    eta_win = binned_df['eta'].values
    chi_win = binned_df['chi_grand_mean'].values
    err_win = binned_df['chi_grand_error'].values

    return eta_win, chi_win, err_win

# =============================================================================
# Peak Fitting & Analysis
# =============================================================================

def fit_peak(eta_win, chi_win, err_win):
    """
    Fits a parabola to the pre-windowed data to find the critical noise 
    point (peak) and its statistical uncertainty.
    """
    if len(eta_win) < 3:
        raise ValueError("Error: The window is too narrow! At least 3 points required to fit a parabola.")

    # Perform the Curve Fit (Unweighted)
    initial_guess = [-0.1, np.mean(eta_win), np.max(chi_win)]
    popt, pcov = curve_fit(parabola, eta_win, chi_win, p0=initial_guess)
    a, b, c = popt
    
    # Calculate Residuals & Reduced Chi-Squared
    residuals = chi_win - parabola(eta_win, *popt)
    safe_err_win = np.where(err_win == 0, np.inf, err_win)
    chisq = np.sum((residuals / safe_err_win)**2)
    dof = len(eta_win) - len(popt)
    red_chisq = chisq / dof if dof > 0 else 0

    if a >= 0:
        print("\nWarning: The fitted parabola opens upwards! Check your window bounds.")
        
    # Calculate the Maximum Point (Critical Noise)
    eta_max = -b / (2 * a)

    # Propagate Error to the Maximum Point
    var_a, var_b, cov_ab = pcov[0, 0], pcov[1, 1], pcov[0, 1]
    df_da = b / (2 * a**2)
    df_db = -1 / (2 * a)
    
    variance_eta_max = (df_da**2 * var_a) + (df_db**2 * var_b) + (2 * df_da * df_db * cov_ab)
    error_eta_max = np.sqrt(variance_eta_max)

    print("=" * 50)
    print("ANALYSIS RESULTS (WINDOWED FOCUS)")
    print("=" * 50)
    print(f"Data points in fit window : {len(eta_win)}")
    print(f"Degrees of Freedom (dof)  : {dof}")
    print(f"Reduced Chi-Squared       : {red_chisq:.4f}")
    print(f"Fitted Parabola           : y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}")
    print("-" * 50)
    print(f"Critical Noise (eta_c)    : {eta_max:.4f} ± {error_eta_max:.4f}")
    print("=" * 50)

    return {
        'eta_win': eta_win, 'chi_win': chi_win, 'err_win': err_win,
        'popt': popt, 'eta_max': eta_max, 'error_eta_max': error_eta_max
    }

# =============================================================================
# Plotting
# =============================================================================

def _apply_custom_theme(ax, bg_color="#0a1628", text_color="#88ccee"):
    """Applies a consistent dark theme formatting to the given matplotlib axis."""
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=text_color, labelsize=12)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)

def plot_peak_analysis(fit_results, target_n, radius, window_bounds):
    """Generates a presentation-ready plot of the windowed data and the fit."""
    os.makedirs("plots", exist_ok=True)
    
    # Theme Colors
    bg_color = "#0a1628"      
    text_color = "#88ccee"    
    marker_edge = "#d4e6b5"   
    line_color = "#77967d"    
    fit_color = "#ff6b6b"     
    peak_color = "#ffd93d"    

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg_color)
    _apply_custom_theme(ax, bg_color, text_color)

    # Plot the WINDOWED binned data
    ax.errorbar(fit_results['eta_win'], fit_results['chi_win'], yerr=fit_results['err_win'], 
                fmt='s', color=line_color, markerfacecolor=bg_color, 
                markeredgecolor=marker_edge, markeredgewidth=2, markersize=8, 
                capsize=4, capthick=1.5, linewidth=2.5, linestyle='none', 
                label='Binned Data (Windowed)')
    
    # Plot the Smooth Parabola Fit
    a, b, c = fit_results['popt']
    eta_max, error_eta_max = fit_results['eta_max'], fit_results['error_eta_max']
    
    eta_smooth = np.linspace(window_bounds[0] - 0.05, window_bounds[1] + 0.05, 100)
    chi_smooth = parabola(eta_smooth, a, b, c)
    ax.plot(eta_smooth, chi_smooth, '--', color=fit_color, linewidth=2.5, label='Parabolic Fit')

    # Draw vertical line at maximum
    ax.axvline(eta_max, color=peak_color, linestyle='-', linewidth=2, 
               label=f'Calculated Peak: $\\eta={eta_max:.2f}$')
    
    # Draw shaded region for error margin
    ax.axvspan(eta_max - error_eta_max, eta_max + error_eta_max, 
               color=peak_color, alpha=0.15, label='Peak Error Margin')

    ax.set_xlabel(r'Noise strength $\eta$', fontsize=13)
    ax.set_ylabel(r'Susceptibility $\chi$ (Binned Grand Mean)', fontsize=13)
    ax.set_title(f'Focused Peak Analysis at $r = {radius}$ (N={target_n})', 
                 fontsize=15, color=text_color)
    
    # Add headroom to y-axis
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.25)

    ax.legend(loc='upper right', fontsize=11, facecolor=bg_color, 
              edgecolor=text_color, labelcolor=text_color, framealpha=1)
              
    ax.grid(False)
    plt.tight_layout()
    
    save_path = "plots/peak_analysis.png"
    plt.savefig(save_path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    print(f"  Saved plot to: {save_path}")
    plt.show()
    
    plt.close(fig)

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # Analysis Parameters
    RADIUS = 1.25
    CSV_FILENAME = f"data/susceptibility_data_r{RADIUS}.csv"
    TARGET_N = 300        
    
    # Focus Window
    ETA_WINDOW_MIN = 4.5
    ETA_WINDOW_MAX = 5.0
    NUM_BUCKETS = 5      # Will be applied *only* between 4.5 and 5.0

    try:
        # 1. Load, pre-filter, and bin the data
        eta_win, chi_win, err_win = load_and_bin_data(
            CSV_FILENAME, TARGET_N, NUM_BUCKETS, ETA_WINDOW_MIN, ETA_WINDOW_MAX
        )
        
        # 2. Fit the peak
        fit_results = fit_peak(eta_win, chi_win, err_win)
        
        # 3. Plot the results
        plot_peak_analysis(fit_results, TARGET_N, RADIUS, (ETA_WINDOW_MIN, ETA_WINDOW_MAX))
                           
    except Exception as e:
        print(e)