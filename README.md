# Low-Rank Fields in Recurrent Neural Networks

This repository contains the code necessary to reproduce the simulations and results of the paper:

**Linking Neural Manifolds to Circuit Structure in Recurrent Networks**

*Louis Pezon, Valentin Schmutz, Wulfram Gerstner*

bioRxiv preprint: https://doi.org/10.1101/2024.02.28.582565

## Overview

This project investigates how low-dimensional neural manifolds emerge in recurrent neural networks (RNNs) and how they relate to the underlying circuit connectivity structure. The code implements low-rank field theory for analyzing and simulating recurrent networks with structured connectivity patterns.

## Key Features

- **Low-Rank Network Simulations**: Simulate recurrent neural networks with low-rank connectivity structures
- **Multiple Connectivity Models**: Implementations of various connectivity patterns (ring, clustered, manifold-structured, etc.)
- **Neural Trajectory Analysis**: Tools for analyzing neural population dynamics and latent trajectories
- **Context-Dependent Manifolds (CDM)**: Simulation and analysis of context-dependent neural dynamics
- **Visualization Tools**: Comprehensive plotting utilities for phase portraits, neural trajectories, and embeddings

## Installation

### Requirements

- [Julia](https://julialang.org/) (tested with Julia 1.x)
- Required Julia packages (see below)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/louispezon/paper-lowrank-fields.git
cd paper-lowrank-fields
```

2. Install required Julia packages:
```julia
include("packages.jl")
```

This will install the following dependencies:
- ProgressBars, ProgressMeter
- LaTeXStrings
- SparseArrays
- SpecialFunctions
- Printf
- JLD2
- Random
- MultivariateStats
- PyPlot
- LinearAlgebra
- NLsolve
- Colors, ColorSchemes

## Repository Structure

```
paper-lowrank-fields/
├── README.md                  # This file
├── src.jl                     # Core simulation functions and utilities
├── packages.jl                # Package dependencies
├── examples/                  # Example scripts for specific figures
│   ├── limit_cycle_step_nonlin.jl
│   ├── periodic_states_ring.jl
│   ├── pw_Gauss_erf.jl
│   ├── ring_vs_gaus_embeds_sort.jl
│   └── ring_vs_gaus_latent_traj_erf.jl
├── simulate_CDM/              # Context-Dependent Manifold simulations
│   ├── RUN.jl                 # Main simulation script
│   ├── defs.jl                # Parameter definitions
│   ├── models.jl              # Model definitions
│   └── setup_input.jl         # Input configuration
├── plot_CDM/                  # CDM analysis and plotting scripts
│   ├── plot_CDM.jl            # Main plotting script
│   ├── PCA_conndefined.jl     # PCA analysis
│   ├── PC_embeds_tuning.jl    # Principal component embeddings
│   ├── TDR.jl                 # Targeted dimensionality reduction
│   ├── plot_hidden.jl         # Hidden variable visualization
│   ├── plot_utils.jl          # Plotting utilities
│   └── more_utils.jl          # Additional utilities
└── data/                      # Directory for simulation results
```

## Usage

### Running Basic Examples

The `examples/` directory contains scripts that generate specific figures from the paper. Each script is self-contained and can be run independently:

```julia
# Example: Ring vs Gaussian models with limit cycles
include("examples/ring_vs_gaus_latent_traj_erf.jl")

# Example: Limit cycles with step nonlinearity
include("examples/limit_cycle_step_nonlin.jl")
```

### Running Context-Dependent Manifold (CDM) Simulations

To simulate networks with context-dependent manifolds:

```julia
# Navigate to the simulate_CDM directory
cd("simulate_CDM/")

# Run the main simulation
include("RUN.jl")
```

This will:
1. Initialize networks with different connectivity patterns (ring, modular-specific, population-specific, clustered, hidden)
2. Simulate network dynamics across multiple trials
3. Save results to the `data/` directory

### Analyzing and Plotting Results

After running simulations, analyze and visualize the results:

```julia
# Navigate to the plot_CDM directory
cd("plot_CDM/")

# Load and analyze simulation results
include("plot_CDM.jl")
```

This script performs:
- Low-pass filtering of neural activity
- PCA and dimensionality reduction
- Trajectory plotting and phase portraits
- Manifold analysis

## Core Concepts

### Low-Rank Connectivity
The code implements networks with connectivity matrices that can be decomposed as low-rank structures, allowing for theoretical analysis of the emergent dynamics.

### Latent Variables (κ)
The low-rank structure induces a low-dimensional latent space characterized by variables κ that capture the essential dynamics of the network.

### Transfer Functions
Various transfer functions (φ) are supported:
- Step function: `φ(x) = R*(x > 0)`
- Error function: `φ(x) = R*(erf(x)+1)/2`
- Custom nonlinearities

### Network Models
Several connectivity patterns are implemented:
- **Ring model**: Neurons arranged on a 1D ring with distance-dependent connectivity
- **Gaussian model**: Neurons in feature space with Gaussian-shaped connectivity
- **Modular-Specific (MS)**: Context-specific connectivity patterns
- **Population-Specific (PS)**: Population-level connectivity structure
- **Clustered**: Modular connectivity with distinct clusters
- **Hidden**: Networks with hidden contextual variables

## Key Functions

### Simulation Functions (src.jl)
- `run_()`: Basic network simulation with optional spiking
- `run_ext_input()`: Simulation with external low-dimensional input
- `run_full_ext_input()`: Simulation with full external input
- `run_from_init_conds()`: Run simulations from multiple initial conditions

### Analysis Functions
- `Z_score()`: Z-score normalization with low-pass filtering
- `embed_spikes()`: Embed spike trains into lower dimensions
- `low_pass_filter()`: Temporal filtering of neural activity

### Visualization Functions
- `plot_traj()`: Plot neural trajectories in latent space
- `plot_flow()`: Visualize flow fields and phase portraits
- `plot_trajs_models()`: Compare trajectories across models
- `raster_plot()`: Generate spike raster plots

## Output

Simulation results are saved in the `data/` directory in JLD2 format, containing:
- Neural activity traces (h_t)
- Latent variable trajectories (κ_t)
- Spike trains (spikes)
- Input patterns (inputs)
- Model parameters and metadata

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pezon2024linking,
  title={Linking Neural Manifolds to Circuit Structure in Recurrent Networks},
  author={Pezon, Louis and Schmutz, Valentin and Gerstner, Wulfram},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.02.28.582565}
}
```

## Contact

For questions, issues, or additional details:
- **Louis Pezon**: louis.pezon@epfl.ch

## License

Please refer to the repository for license information.

## Acknowledgments

This research was conducted at the École Polytechnique Fédérale de Lausanne (EPFL).
