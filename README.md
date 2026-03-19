# IRMAE + Neural ODE for Delay-Embedded Flow Data

This repository implements a reduced-order modeling framework for time-resolved experimental flow data using **delay embedding**, **PCA**, **IRMAE**, and a **Neural ODE**.

The workflow first constructs delay-embedded snapshots from the raw signal, projects them onto a lower-dimensional PCA space, and learns a nonlinear latent representation using IRMAE. A Neural ODE is then trained in this latent space to model the continuous-time evolution of the system. The predicted latent trajectory is decoded back to PCA space and then reconstructed in the original delay-embedded space for comparison with the input data.

## Workflow

1. Load data from a MATLAB `.mat` file
2. Normalize and split into train/test sets
3. Build delay embeddings
4. Standardize using training statistics
5. Apply PCA on the training data
6. Train IRMAE on PCA coefficients
7. Extract latent trajectories
8. Train a Neural ODE in latent space
9. Roll out predictions over train and test trajectories
10. Decode predictions back to PCA space and original delay space
11. Generate error metrics and visualization plots

## Applications

This framework is useful for nonlinear reduced-order modeling of unsteady experimental systems, especially flow datasets such as pitching and heaving airfoil experiments.
