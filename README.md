# Active Matter Dynamics in Fish Schooling (Vicsek Model)

## Project Context
This repository contains the computational simulation and analysis codebase for the "Active Matter" section of a larger University of Manchester group project titled *"Nuclear Fishion and Marine Field Theory: Do schools of fish behave analogously to nuclei and active matter?"*. 

While the broader project explored macroscopic similarities between fish schools and nuclear fission, this specific sub-project, developed in collaboration with Jongheon Lee, focuses entirely on the statistical mechanics of the school. We investigated the kinetic phase transitions of self-propelled particles to determine how biological active matter differs from traditional physical systems like magnetic spins.

## Physics Background & TL;DR
Can a school of fish be modeled like atomic spins in a magnet? To answer this, we simulated a school using the Vicsek model, a standard framework for active matter driven by local orientation alignment and random noise.
* **The Analogy:** In our simulation, the random "noise" in a fish's movement is analogous to thermal temperature in the Ising model. The "order parameter" measures how aligned the school is (1 = perfectly aligned, 0 = complete chaos).
* **The Methodology:** We swept through varying noise levels to find the "critical noise", the exact point where the school undergoes a phase transition from order to total disorder. This critical point is identified by a massive spike in the system's "susceptibility" (the variance of the order parameter).
* **The Discovery:** We ran these simulations across various interaction radii. Unlike the Mean Field Theory of magnetism where critical temperature scales linearly with the interaction radius squared, the critical noise in our biological fish school model approaches a plateauing limit (maxed at $2\pi. This mathematically demonstrates that a shoal is driven by visual instinct and topological limits rather than purely additive physical forces.

## Repository Structure
```text
.
|-- data/                                    # Simulated susceptibility and critical noise outputs
|-- plots/                                   # Automatically generated analysis plots
|-- fish_thermalization_susceptibility.py    # Core Vicsek model simulator and phase transition sweep
|-- maximum_fititng_bin.py                   # Statistical peak analysis script 
|-- results_plotting.py                      # Final visualization script
|-- Group_18_Poster.pdf                      # Full group project poster (Contains Nuclear & Active Matter sections)
`-- README.md
```
## Code Module Descriptions

### 1. Simulation Engine (`fish_thermalization_susceptibility.py`)
This script acts as the core physics engine. It simulates $N$ self-propelled particles in a 2D periodic boundary box. It sweeps through defined noise strengths ($\eta$), allows the system to thermalize into a steady state, and calculates both the macroscopic order parameter ($\varphi$) and its corresponding susceptibility ($\chi$). The results are logged to CSV files for statistical analysis.

### 2. Critical Point Extraction (`maximum_fititng_bin.py`)
Because the susceptibility peak near a phase transition is highly sensitive to finite-size scaling and statistical noise, this script isolates the transition window. It bins the raw trial data to crush statistical noise, fits a quadratic parabola to the binned susceptibility curve, and algebraically extracts the exact critical noise point ($\eta_c$) along with carefully propagated uncertainties.

### 3. Physical Evaluation (`results_plotting.py`)
This module aggregates the extracted critical noise values ($\eta_c$) across different simulation runs varying the interaction radius ($r$). It plots the critical noise against the interaction area ($r^2$) to visually demonstrate the plateauing divergence from standard Mean Field Theory predictions.
