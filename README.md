# Low-Rank Fields in Recurrent Neural Networks

This repository contains the code necessary to reproduce the simulations and results of the paper:

**Linking Neural Manifolds to Circuit Structure in Recurrent Networks**

*Louis Pezon, Valentin Schmutz, Wulfram Gerstner*

bioRxiv preprint: https://doi.org/10.1101/2024.02.28.582565

## Installation

1. Clone this repository:
```bash
git clone https://github.com/louispezon/paper-lowrank-fields.git
cd paper-lowrank-fields
```

2. Install required Julia packages:
```julia
include("packages.jl")
```

## Repository Structure

- **`examples/`**: Code to run and plot specific examples from the paper (except the context-dependent decision-making models)
- **`simulate_CDM/`**: Code to set up, run, and save simulations for the context-dependent decision-making (CDM) task with 5 different RNN models described in the paper
- **`plot_CDM/`**: Code to load, analyze, and plot results of the CDM task for the 5 models
- **`src.jl`**: Useful functions to simulate low-rank RNNs
- **`packages.jl`**: List of required packages

## Usage

### Running Examples

The `examples/` directory contains scripts that generate specific figures from the paper:

```julia
# Example: Ring vs Gaussian models with limit cycles
include("examples/ring_vs_gaus_latent_traj_erf.jl")
```

### Running CDM Simulations

To simulate the context-dependent decision-making task with the 5 RNN models:

```julia
cd("simulate_CDM/")
include("RUN.jl")
```

This will simulate network dynamics across multiple trials and save results to the `data/` directory.

### Analyzing CDM Results

To load, analyze, and plot the CDM results:

```julia
cd("plot_CDM/")
include("plot_CDM.jl")
```

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

For questions or additional details: louis.pezon@epfl.ch
