"""
Vicsek Model: Critical Noise vs. Interaction Area

This module visualizes the relationship between the interaction area 
(proportional to radius squared, r^2) and the critical noise parameter (eta_c) 
in the Vicsek model. It reads the compiled critical noise data extracted from 
the susceptibility peak analysis and plots it against the theoretical physical limit.

Author: Min Ki Hong
Date: April 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Data Handling
# =============================================================================

def load_critical_noise_data(filepath):
    """
    Loads and parses the critical noise and interaction radius data.
    
    Args:
        filepath (str): Path to the data file.
        
    Returns:
        tuple: (r_sq, eta_c, eta_err) containing the squared interaction radius,
               the critical noise, and its associated error.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Could not find '{filepath}'. Check the file path!")

    df = pd.read_csv(filepath, skipinitialspace=True)
    
    # In 2D, the interaction area scales with r^2
    r_sq = (df['r'].values)**2
    eta_c = df['eta'].values
    
    # Safely handle variations in column naming conventions
    if 'eta err' in df.columns:
        eta_err = df['eta err'].values
    elif 'eta_err' in df.columns:
        eta_err = df['eta_err'].values
    else:
        raise KeyError("Error: Could not find the error column. Looked for 'eta err' or 'eta_err'.")
        
    return r_sq, eta_c, eta_err

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

def plot_critical_noise_vs_radius(r_sq, eta_c, eta_err, output_dir="plots", filename="eta_vs_radius.png"):
    """
    Generates a publication-quality plot of the critical noise as a function 
    of the squared interaction radius.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Theme Colors
    bg_color = "#0a1628"      
    text_color = "#88ccee"    
    marker_edge = "#d4e6b5"   
    line_color = "#77967d"    

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg_color)
    _apply_custom_theme(ax, bg_color, text_color)

    # Plot the simulated data points
    ax.errorbar(
        r_sq, eta_c, yerr=eta_err, fmt='o',
        linestyle='-', linewidth=2.5,
        color=line_color,         
        markerfacecolor=bg_color, 
        markeredgecolor=marker_edge,
        markeredgewidth=2,
        markersize=8,
        capsize=4, capthick=1.5,
        label=r'Simulated $\eta_c$'
    )

    # Add the Theoretical Physical Limit
    ax.axhline(2 * np.pi, color=text_color, linestyle=':', alpha=0.7, 
               linewidth=2, label=r'Physical Limit ($2\pi$)')

    # Formatting
    ax.set_xlabel(r'Interaction Radius Squared $r^2$', fontsize=13)
    ax.set_ylabel(r'Critical Noise $\eta_c$', fontsize=13)
    ax.set_title('Critical Noise vs. Interaction Area', fontsize=15, color=text_color)
    
    ax.legend(loc='lower right', fontsize=11, facecolor=bg_color, 
              edgecolor=text_color, labelcolor=text_color, framealpha=1)
              
    ax.grid(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    print(f"  Saved plot to: {save_path}")
    plt.show()
    plt.close(fig)

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # Configuration
    CSV_FILE = "data/radius_vs_critnoise.txt"
    
    try:
        # Load Data
        r_sq, eta_c, eta_err = load_critical_noise_data(CSV_FILE)
        
        # Plot Results
        plot_critical_noise_vs_radius(r_sq, eta_c, eta_err)
        
    except Exception as e:
        print(e)