"""
Vicsek Model: Single System Phase Transition Analysis

This module simulates the self-propelled particle (SPP) model to study 
collective motion (e.g., fish schooling or flocking). It evaluates the 
steady-state order parameter and susceptibility across varying noise 
strengths to identify the critical noise point.

Author: Min Ki Hong & Jongheon Lee
Date: April 2026
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =============================================================================
# Vicsek Model Class
# =============================================================================

class VicsekModel:
    """
    Simulates N self-propelled particles in a 2D periodic square box under the
    standard Vicsek alignment-plus-noise rule.

    Attributes:
        N (int): Number of particles.
        L (float): Length of the square simulation box.
        v (float): Constant particle speed.
        r (float): Interaction radius for alignment.
        eta (float): Noise strength bound [0, 2*pi].
    """

    def __init__(self, N: int, L: float, v: float, r: float, eta: float):
        self.N = N
        self.L = L
        self.v = v
        self.r = r
        self.eta = eta

        # Initial conditions: Uniform random positions and headings
        self.x = np.random.uniform(0, L, N)
        self.y = np.random.uniform(0, L, N)
        self.theta = np.random.uniform(0, 2 * np.pi, N)

    def step(self):
        """Advances the simulation by one discrete time step."""
        # --- Step A: Alignment ---
        dx = np.subtract.outer(self.x, self.x)
        dy = np.subtract.outer(self.y, self.y)

        # Apply minimum image convention for periodic boundary conditions
        dx -= self.L * np.round(dx / self.L)
        dy -= self.L * np.round(dy / self.L)

        dist_sq = dx**2 + dy**2
        neighbors = dist_sq <= self.r**2

        sum_sin = neighbors @ np.sin(self.theta)
        sum_cos = neighbors @ np.cos(self.theta)
        avg_theta = np.arctan2(sum_sin, sum_cos)

        noise = np.random.uniform(-self.eta / 2, self.eta / 2, self.N)
        self.theta = avg_theta + noise

        # --- Step B: Propagation ---
        self.x += self.v * np.cos(self.theta)
        self.y += self.v * np.sin(self.theta)

        # Enforce periodic boundary conditions
        self.x %= self.L
        self.y %= self.L

    def order_parameter(self) -> float:
        """
        Computes the normalized macroscopic scalar order parameter (phi).

        Returns:
            float: Order parameter bounded between [0, 1].
        """
        sum_cos = np.sum(np.cos(self.theta))
        sum_sin = np.sum(np.sin(self.theta))
        return np.sqrt(sum_cos**2 + sum_sin**2) / self.N

# =============================================================================
# Phase-Transition Sweep
# =============================================================================

def phase_transition_sweep(N=300, L=7.0, v=0.03, r=1.0, eta_min=0.0, 
                           eta_max=5.0, eta_steps=20, iterations=300, 
                           tail=50, trials=5) -> dict:
    """
    Sweeps noise strength (eta) to measure the steady-state order parameter 
    and susceptibility to map the kinetic phase transition.

    Returns:
        dict: Compiled results including means, errors, and time series data.
    """
    time_series = {}
    etas = np.linspace(eta_min, eta_max, eta_steps)
    
    phi_means, phi_errors = np.zeros(eta_steps), np.zeros(eta_steps)
    chi_means, chi_errors = np.zeros(eta_steps), np.zeros(eta_steps)

    print(f"Phase-transition sweep (N={N}): {eta_steps} noise values, "
          f"{trials} trials each, {iterations} steps per trial.\n")

    for i, eta in enumerate(etas):
        trial_phi = []
        trial_chi = []

        for t in range(trials):
            model = VicsekModel(N, L, v, r, eta)
            history = np.zeros(iterations)

            for step in range(iterations):
                model.step()
                history[step] = model.order_parameter()

            # Store the first trial's history for time-series visualization
            if t == 0:
                time_series[eta] = history.copy()
                    
            tail_data = history[-tail:]

            # <phi>: time-averaged order parameter
            trial_phi.append(np.mean(tail_data))
            # chi = N * Var(phi): susceptibility
            trial_chi.append(N * np.var(tail_data))

        trial_phi = np.array(trial_phi)
        trial_chi = np.array(trial_chi)

        phi_means[i] = np.mean(trial_phi)
        phi_errors[i] = np.std(trial_phi, ddof=1) / np.sqrt(trials)
        chi_means[i] = np.mean(trial_chi)
        chi_errors[i] = np.std(trial_chi, ddof=1) / np.sqrt(trials)

        print(f"\reta = {eta:5.2f} | phi = {phi_means[i]:.3f} | chi = {chi_means[i]:.3f}", end="", flush=True)

    print("\nSweep complete!")

    return {
        'etas': etas,
        'phi_mean': phi_means, 'phi_error': phi_errors,
        'chi_mean': chi_means, 'chi_error': chi_errors,
        'time_series': time_series,
        'N': N
    }

# =============================================================================
# Plotting & Data Output Functions
# =============================================================================

PLOT_DIR = "plots"

def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)

def _apply_custom_theme(ax, bg_color="#0a1628", text_color="#88ccee"):
    """Applies a consistent dark theme formatting to the given matplotlib axis."""
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=text_color, labelsize=12)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)

def plot_phase_transition(results: dict, filename="phase_transition.png"):
    _ensure_plot_dir()
    bg_color, text_color = "#0a1628", "#88ccee"
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg_color)
    _apply_custom_theme(ax, bg_color, text_color)

    ax.errorbar(
        results['etas'], results['phi_mean'], yerr=results['phi_error'],
        fmt='o', linestyle='-', linewidth=2.5, color="#77967d",
        markerfacecolor=bg_color, markeredgecolor="#d4e6b5", markeredgewidth=2,
        markersize=8, capsize=4, capthick=1.5, label=r'Steady-state $\varphi$'
    )
    
    ax.set_xlabel(r'Noise strength $\eta$', fontsize=13)
    ax.set_ylabel(r'Order parameter $\varphi$', fontsize=13)
    ax.set_title('Phase Transition in the Vicsek Model', fontsize=15, color=text_color)
    ax.set_ylim(-0.05, 1.05)
    
    ax.legend(loc='upper right', fontsize=11, facecolor=bg_color, 
              edgecolor=text_color, labelcolor=text_color, framealpha=1)
    
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def plot_susceptibility(results: dict, filename="susceptibility.png"):
    _ensure_plot_dir()
    bg_color, text_color = "#0a1628", "#88ccee"

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg_color)
    _apply_custom_theme(ax, bg_color, text_color)

    ax.errorbar(
        results['etas'], results['chi_mean'], yerr=results['chi_error'],
        fmt='s', linestyle='-', linewidth=2.5, color="#77967d",
        markerfacecolor=bg_color, markeredgecolor="#a8d8b9", markeredgewidth=2,
        markersize=8, capsize=4, capthick=1.5, label=r'Susceptibility $\chi$'
    )
    
    ax.set_xlabel(r'Noise strength $\eta$', fontsize=13)
    ax.set_ylabel(r'Susceptibility $\chi = N \mathrm{Var}(\varphi)$', fontsize=13)
    ax.set_title('Susceptibility Peak near Critical Point', fontsize=15, color=text_color)
    
    ax.legend(loc='best', fontsize=11, facecolor=bg_color, 
              edgecolor=text_color, labelcolor=text_color, framealpha=1)
              
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def plot_time_series(results: dict, tail=50, filename="time_series.png"):
    """Plots the order parameter vs time to verify steady-state thermalization."""
    _ensure_plot_dir()
    
    time_series_data = results.get('time_series', {})
    if not time_series_data:
        print("Error: No 'time_series' data found in results dict!")
        return

    bg_color, text_color = "#0a1628", "#88ccee"
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    _apply_custom_theme(ax, bg_color, text_color)
        
    iterations = len(next(iter(time_series_data.values())))
    steps = np.arange(iterations)
    
    etas = sorted(time_series_data.keys())
    norm = plt.Normalize(vmin=min(etas), vmax=max(etas))
    
    try:
        cmap = plt.colormaps['cool']
    except AttributeError:
        cmap = cm.get_cmap('cool')
    
    for eta in etas:
        history = time_series_data[eta]
        ax.plot(steps, history, color=cmap(norm(eta)), alpha=0.8, linewidth=1.5)
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    
    cbar.set_label(r'Noise strength $\eta$', fontsize=12, color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
    cbar.outline.set_edgecolor(text_color)
    cbar.outline.set_linewidth(1.2)

    ax.axvspan(iterations - tail, iterations, color=text_color, alpha=0.15, 
               label=f'Averaging Window (Last {tail} steps)')
    
    ax.set_xlabel('Time step (Iterations)', fontsize=13)
    ax.set_ylabel(r'Order parameter $\varphi$', fontsize=13)
    ax.set_title('System Thermalization (Steady State Check)', fontsize=15, color=text_color)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, iterations)
    
    ax.legend(loc='lower left', fontsize=11, facecolor=bg_color, 
              edgecolor=text_color, labelcolor=text_color, framealpha=1)
              
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def append_chi_to_csv(results: dict, filename="susceptibility_data.csv"):
    """
    Appends the eta and chi (susceptibility) values from a simulation run to a CSV file. 
    """
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['eta', 'chi_mean', 'chi_error', 'N'])
            
        etas = results['etas']
        chi_means = results['chi_mean']
        chi_errors = results['chi_error']
        N = results['N']
        
        for i in range(len(etas)):
            writer.writerow([f"{etas[i]:.6f}", f"{chi_means[i]:.6f}", f"{chi_errors[i]:.6f}", N])
            
    print(f"Appended {len(etas)} rows of data to '{filename}'")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    # System parameters
    N = 300           # Number of particles
    L = 7.0           # Box size
    v = 0.03          # Particle speed
    r = 1.25         # Interaction radius

    # Noise Sweep parameters
    ETA_MIN = 0      
    ETA_MAX = 5.5     
    ETA_STEPS = 15

    # Trial Configuration
    ITERATIONS = 500  # Steps per trial
    TAIL = 300        # Steady-state sampling window
    TRIALS = 5        # Independent trials per eta value

    print("=" * 50)
    print("1. Phase-transition sweep (Single System Size)")
    print("=" * 50)
    
    eta_results = phase_transition_sweep(
        N=N, L=L, v=v, r=r,
        eta_min=ETA_MIN, eta_max=ETA_MAX, eta_steps=ETA_STEPS,
        iterations=ITERATIONS, tail=TAIL, trials=TRIALS
    )

    plot_phase_transition(eta_results)
    plot_susceptibility(eta_results)
    plot_time_series(eta_results, tail=TAIL)
    append_chi_to_csv(eta_results, filename=f"data/susceptibility_data_r{r}.csv")