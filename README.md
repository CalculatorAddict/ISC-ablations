# ISC-ablations

Code and results for *Neural Networks that Learn Like Humans* (MMathCompSci dissertation, Oxford, 2026).
It replicates and ablates the Integrated Semantics and Control (ISC) model of Giallanza et al. (2024) on the
word–picture distractor task, then adds Hebbian (Oja's-rule) context gating and sluggish task units to make
the model prefer temporally blocked over interleaved training, as humans do.

## Repository map
```
isc_ablations/            Python package: models + the scripts that generate the data
  common/                 data loading (data.py), device choice (utils.py), cue-blocking bar plots (plotting.py)
  isc_model/              ISC model, pretrained-model loader, ISC experiments (incl. easy/hard, thresholding)
  mlp_model/ ablated_model/ onelayer_model/   ablation baselines
  hebbian_model/          Oja/Hebbian ISC model with sluggish (EMA) task units + its experiments
  error_gating_model/     error-gated variant (ongoing)
experiments/              one folder per experiment: notebooks (+ Julia scripts for 03)
  01_ablations/  02_easy_hard/  03_oja/  04_sluggishness/  05_error_gating/
data/                     inputs/ (Leuven concepts, human data) + one results folder per experiment
models/                   pretrained/ (10 base ISC models) + checkpoints per experiment
figures/                  figures per experiment (PNG from Python, PDF from Julia)
```
Every result lives in the folder of the experiment whose code writes it. Other experiments read it from there.

## Experiments
| # | Notebook(s) | Question | Dates | Dissertation |
|---|---|---|---|---|
| 01 | [`cue_blocking_ablations`](experiments/01_ablations/cue_blocking_ablations.ipynb), [`learning_dynamics`](experiments/01_ablations/learning_dynamics.ipynb), [`temporal_blocking`](experiments/01_ablations/temporal_blocking.ipynb) | Is the ISC model's cue-blocking preference due to its architecture or the cue? Does it prefer *temporal* blocking? | Dec 2025 – May 2026 | Ch 2 (Figs 2.3–2.5) |
| 02 | [`easy_hard_curriculum`](experiments/02_easy_hard/easy_hard_curriculum.ipynb) | Does an easy→hard curriculum help the ISC model? | Jan – Feb 2026 | not included |
| 03 | [`oja_verification`](experiments/03_oja/oja_verification.ipynb), [`oja_autocorrelation.jl`](experiments/03_oja/oja_autocorrelation.jl), [`oja_eigenvalues.jl`](experiments/03_oja/oja_eigenvalues.jl) | Does Oja's rule find the first PC, when does it diverge, and does Oja gating help blocked training? | Apr – May 2026 | Ch 3–4 (Figs 3.1, 4.1–4.4, 4.6), Fig 5.3 |
| 04 | [`hebbian_sluggishness`](experiments/04_sluggishness/hebbian_sluggishness.ipynb) | Do sluggish task units make the Hebbian model prefer temporal blocking? | Feb – May 2026 | Ch 5 (Figs 5.1, 5.2) |
| 05 | [`error_gating`](experiments/05_error_gating/error_gating.ipynb) | Can an error signal replace the context cue? | May 2026 – ongoing | follows up §6.1 |

Each notebook opens with a header (question, inputs, outputs) and has a dated log block per section
(status: final / superseded / exploratory, which code produced which file, and the result).

## Dissertation figure index
| Fig | File | Produced by |
|---|---|---|
| 2.3 | [`figures/01_ablations/cue_blocking_isc_mlp_human.png`](figures/01_ablations/cue_blocking_isc_mlp_human.png) | `cue_blocking_ablations` §2 call, but with `include_human=True` (the saved cell has `False`) |
| 2.4 | [`figures/01_ablations/isc_vs_mlp_learning_curve.png`](figures/01_ablations/isc_vs_mlp_learning_curve.png) | `learning_dynamics` §3 |
| 2.5 | [`figures/01_ablations/isc_temporal_blocked_vs_interleaved.png`](figures/01_ablations/isc_temporal_blocked_vs_interleaved.png) | `temporal_blocking`, last cell (saved by hand from the inline output) |
| 3.1 | [`figures/03_oja/pc_convergence/final_weights_raw_50seeds.png`](figures/03_oja/pc_convergence/final_weights_raw_50seeds.png), [`…_pc1_projection_50seeds.png`](figures/03_oja/pc_convergence/final_weights_pc1_projection_50seeds.png) | `oja_verification` §1c |
| 4.1 | [`figures/03_oja/autocorrelation/phase_diagram.pdf`](figures/03_oja/autocorrelation/phase_diagram.pdf), [`fluctuation_map.pdf`](figures/03_oja/autocorrelation/fluctuation_map.pdf) | `oja_autocorrelation.jl` |
| 4.2 | [`figures/03_oja/autocorrelation/norm_slow_iid.pdf`](figures/03_oja/autocorrelation/norm_slow_iid.pdf), [`norm_slow_corr.pdf`](figures/03_oja/autocorrelation/norm_slow_corr.pdf) | `oja_autocorrelation.jl` |
| 4.3 | [`figures/03_oja/autocorrelation/stability_map.pdf`](figures/03_oja/autocorrelation/stability_map.pdf) | `oja_autocorrelation.jl` |
| 4.4 | [`figures/03_oja/eigenvalues/stability_eigenvals.pdf`](figures/03_oja/eigenvalues/stability_eigenvals.pdf) | `oja_eigenvalues.jl` |
| 4.6 | [`figures/03_oja/oja_isc_baseline.png`](figures/03_oja/oja_isc_baseline.png) (left), [`oja_learning_curves.png`](figures/03_oja/oja_learning_curves.png) (right) | `oja_verification` §3. The dissertation's left panel is the 2026-05-01 render (removed from history in the 2026-10 purge); the current file is a later re-render |
| 5.1 | [`figures/04_sluggishness/alpha_grid_search_n20.png`](figures/04_sluggishness/alpha_grid_search_n20.png) | `hebbian_sluggishness` §5.2 (20 models; the dissertation caption says 10 runs) |
| 5.2 | [`figures/04_sluggishness/hebbian_vs_human_cue_blocking.png`](figures/04_sluggishness/hebbian_vs_human_cue_blocking.png) | `hebbian_sluggishness` §6 |
| 5.3 | [`figures/03_oja/oja_vs_isc_learning_curve.png`](figures/03_oja/oja_vs_isc_learning_curve.png) | `oja_verification` §3 |

The other figures are schematics or taken from cited papers.

## Reproducing
- **Python:** the notebooks used a conda `base` env with Python 3.13. Install the packages in [`requirements.txt`](requirements.txt) (versions not pinned).
  Each notebook's first code cell `cd`s to the repo root and puts it on `sys.path`, so open the notebooks from anywhere.
- **Data generators** are modules run from the repo root, e.g.
  ```bash
  python -m isc_ablations.isc_model.experiment
  ```
  These scripts run their experiment on import, so don't `import` them from elsewhere.
- **Julia** 1.12 with `Plots` and `LaTeXStrings`, run from the repo root:
  ```bash
  julia experiments/03_oja/oja_autocorrelation.jl
  ```
- ⚠ **Cache-or-train:** most experiment code loads a checkpoint from `models/` if it exists, and otherwise
  **retrains and overwrites** the corresponding outputs. Re-running a cell isn't a no-op.
- Notebooks were run out of order during research; the per-section logs say which cells write which files.
- Several scripts hard-code `device='mps'` (Apple silicon); the work was done on an M2 MacBook Pro.

## Known issues
- `isc_ablations/mlp_model/experiment_batch1.py` imports a module that never existed, so the batch-size-1 data can't be regenerated.
- `error_gating_model.experiment.run_error_experiment` lists `models/05_error_gating/alpha_grid_search/`, which isn't in git. Create it before running.
- The easy/hard checkpoints and the 20- and 40-model easy/hard results were removed to save space. Those notebook cells keep their saved outputs.
- The Hebbian bars in `temporal_blocking` and sections 1–4 of `hebbian_sluggishness` used earlier model versions and can't be reproduced from the current code.
- Result files keep the paths they were written with (e.g. the `weights_path` column of `ablation_learning_curve_bce.csv`), and so do notebook outputs.
- Fig 2.5's error bars are standard deviations; the dissertation text says standard error.

## References
- T. Giallanza et al. *An integrated model of semantics and control.* Psychological Review 132(5), 2024.
- T. Flesch et al. *Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals.* PLoS Comput. Biol. 19(1), 2023.

Licensed under the terms in [`LICENSE`](LICENSE).
