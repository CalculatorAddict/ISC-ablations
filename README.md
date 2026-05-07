# ISC-ablations
Ablations for the Integrated Semantics and Control (ISC) model, investigating blocked vs interleaved training curricula. See Giallanza et al. (2024) for the base model.
## Notebooks
| File | Contents |
|---|---|
| `ablation_experiments.ipynb` | Ablations on cued curriculum, human baseline |
| `hebbian_learning.ipynb` | Hebbian and Oja's rule learning experiments |
| `oja_verification.ipynb` | Oja's rule convergence to first PC and EMA experiment |
| `learning_dynamics.ipynb` | Learning dynamics of ISC model vs ablated MLP |
| `error_gating.ipynb` | Error-gating exploration |
| `new_directions.ipynb` | Easy-hard curriculum and Hebbian explorations |
| `oja_experiment.jl`, `oja_eigenvals.jl` | Oja's rule under autocorrelation |
Julia figures output to `figures/`, Python figures output to `images/`. Experimental configs in `experimental_setups/`.
## Reproducing
Python notebooks: run sequentially within each file. Originally run locally on M2 MacBook Pro. Saved model weights are in `models/` for selected experiments.
Julia scripts require Julia:
```julia
julia oja_experiment.jl
```