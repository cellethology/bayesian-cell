# Cellular Bayes Filtering for In Vivo Migration

A comprehensive simulation framework for studying cell migration in tissue environments using Bayesian filtering and biochemical navigation strategies. This repository implements both MATLAB-based cellular simulations and traditional Python-based robotics algorithms.

## Key Features

- **Multiple Navigation Strategies**: Biochemical gradient sensing, Bayesian filtering
- **Tissue Environment Simulation**: Realistic tissue environments
- **LEGI Models**: Local Excitation, Global Inhibition models
- **Robotic Implementation**: Python equivalent using advanced filtering techniques (EKF)

## Directory Structure

```
.
├── src/                    # Core MATLAB navigation simulation code
├── tissue_sim/            # Tissue environment generation and simulation
│   ├── koff_variants/     # Different binding kinetics environments
│   └── point_source/      # Point source gradient environments
├── LEGI/                  # LEGI model implementations
├── robotic/               # Python robotic implementation
│   ├── core/             # Base environment and configuration
│   ├── filters/          # EKF, UKF filter implementations
│   ├── histogram_filter/ # Bayesian histogram filter
│   └── analysis/         # Comparison and analysis tools
├── run.m                  # Main MATLAB example script
└── requirements.txt       # Python dependencies
```

## Requirements

### MATLAB Requirements

- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox (for statistical functions)
- Signal Processing Toolbox (recommended)

### Python Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - numpy, scipy, matplotlib
  - filterpy (for Kalman filtering)
  - Other scientific computing libraries

## Installation

### MATLAB Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd bayesian-cell
   ```
2. Add the project directory and subdirectories to your MATLAB path:

   ```matlab
   addpath(genpath('/path/to/bayesian-cell'))
   ```

### Python Setup

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run Python simulations:

   ```bash
   cd robotic
   python main.py
   ```

## Key Functions

### racing_cells Function

```matlab
[schemerate,unifrate,statsummary] = racing_cells(fname, param, varargin)
```

Primary function for simulating cell migration in tissue environments.

**Parameters:**

- `fname`: Environment file name (e.g., "tissue_point_noflow")
- `param`: Simulation parameters structure
- **Optional Parameters:**
  - `'task'`: "localization" or "retention" (default: "localization")
  - `'envmodel'`: "tissue" or "grad" (default: "tissue")
  - `'receptor'`: "feedback", "uniform", or "w1dist" (default: "feedback")
  - `'decoder_method'`: "optimal_noise", "perfect", or "randomwalk"
  - `'source'`: "edge" or "point" (default: "edge")
  - `'save_data'`: 0 (no save), 1 (basic), 2 (all data) (default: 2)
  - `'makeplot'`: Enable/disable plotting (default: true)

**Returns:**

- `schemerate`: Success rate using specified navigation strategy
- `unifrate`: Success rate using uniform random strategy
- `statsummary`: Detailed statistics summary

## Usage Examples

### Basic MATLAB Simulation

Run the default example:

```matlab
run
```

### Custom Simulations

#### Bayesian vs Gradient-tracking Comparison

```matlab
% Bayesian filtering approach
bayes_rate = racing_cells("tissue_env", scheme_param, ...
    "receptor", "feedback");

% Pure biochemical approach  
biochem_rate = racing_cells("tissue_env", scheme_param, ...
    "receptor", "uniform");
```

#### Environment Variants

```matlab
% Different binding kinetics
racing_cells("tissue_env_koff=1e-3", scheme_param);
racing_cells("tissue_env_koff=1e-1", scheme_param);

% Point source vs edge source
racing_cells("tissue_point", scheme_param, "source", "point");
```

### Python Robotic Implementation

```bash
cd robotic
python main.py
```

## Tissue Environments

### Available Environment Files

- **tissue_env.mat**: Standard tissue environment with gradient
- **tissue_point.mat**: Point source gradient

### Environment Parameters

- Gradient strength and direction
- Binding/unbinding rates (kon, koff)
- Tissue geometry and boundary conditions
- Flow fields (where applicable)

## LEGI Models

The `LEGI/` directory contains Local Excitation, Global Inhibition models (code from [Shi et al (2013)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003122)):

- **LEGI_BEN_POL.m**: LEGI model with BEN polarization
- **LEGI_BEN_POL_LSM.m**: LEGI with Lateral Segregation Model
- **RDS_grdt_sde.m**: Reaction-diffusion system with gradient and stochastic differential equations
- **make_init.m**: Initialize LEGI model parameters
- **plot_BEN_POL.m**: Visualization tools for LEGI results
