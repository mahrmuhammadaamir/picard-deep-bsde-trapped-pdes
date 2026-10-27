# picard-deep-bsde-trapped-pdes
Python codes for the numerical experiments in the paper on A Picard-Coupled Deep BSDE Method for High-Dimensional Trapped PDEs on Unbounded Domain
# Python codes for the numerical examples

This repository contains Python codes for selected numerical examples in
the manuscript:

**A Picard-Coupled Deep BSDE Method for High-Dimensional Trapped PDEs on Unbounded Domains**

## Contents

- `example_linear_TPDE.py`: a linear trapped PDE in 50 dimensions. The numerical approximation is compared with an independent Monte Carlo reference based on the inverse subordinator.

- `example_trapped_allen_cahn.py`: a trapped Allen–Cahn equation with a manufactured exact solution. The numerical approximation is compared with the exact value.

## Requirements

Python 3.9 or later and the following packages:

- PyTorch
- NumPy
- SciPy
- Matplotlib

If the `requirements.txt` file is included, install the packages using:

```bash
pip install -r requirements.txt
```

## Running the codes

The parameter values for the numerical experiments are described in the manuscript. Each script can be run using Python:

```bash
python example_linear_TPDE.py
```

```bash
python example_trapped_allen_cahn.py
```

## Output

Each run writes its output to the folder specified by `--out_dir`. The scripts produce plots of the approximate solution and relative error. The final approximation, reference value, relative error, and wall-clock time are printed at the end of the run.

## Note

The supplied scripts correspond to Examples linear and nonlinear cases. First can be implemented by modifying the parameters and the example solution in `example_linear_TPDE.py`.
