# Cellular bayes filtering for in vivo migration

This repository contains MATLAB code to simulate cell migration in a tissue environment. The simulation includes multiple navigation strategies including a purely biochemical approach and a statistical approach based on Bayes filtering. Both implementations are equivalent in their outcomes.


## Directory Structure

```
.
├── src/           # Core simulation code
├── env/           # Tissue environment definitions
└── run.m          # Example usage script
```


### racing_cells Function

```matlab
racing_cells(environment, strategy, params)
```

Primary function for simulating cell migration with configurable:

- Tissue environments
- Migration strategies
- Simulation parameters


## Usage

1. Clone the repository
2. Add the project directory to your MATLAB path
3. Run the example script:

```matlab
run
```

Or create custom simulations:

```matlab
params = struct('param1', value1, 'param2', value2);
racing_cells('tissue_env1', params);
```
