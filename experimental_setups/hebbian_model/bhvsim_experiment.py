import os
import warnings

import data
import numpy as np
import pandas as pd
import torch

from ..isc_model.model import load_isc_models
from .oja_variant_model import SimpleHebbianModel as OjaModel
from .experiment import prepare_experiment_data

warnings.filterwarnings('ignore')

model_path = 'models'
num_models = 10
alpha = np.linspace(0.05, 1, 30)[25]

isc_models = load_isc_models(num_models)

# val set -- used for TTC check and final eval
val_x, val_y, size_conditions, cat_conditions, random_cat_conditions, blocks = (
    data.make_behavioral_experiment_training_data(distractor_strength=0.975)
)

# mean-center task context, zero out everything except blocked
task_context = val_x[1]
task_context_centered = task_context - task_context.mean(dim=0, keepdim=True)
blocked_mask = torch.tensor([b == 'blocked' for b in blocks], dtype=torch.bool)
task_context_eval = torch.zeros_like(task_context_centered)
task_context_eval[blocked_mask] = task_context_centered[blocked_mask]
val_x_eval = [val_x[0], task_context_eval]

# train set
train_x, train_y = prepare_experiment_data()

def calc_model_error(sim, x, y):
    with torch.no_grad():
        return torch.abs(sim(x) - y)[:, [2541, 2542]].mean(axis=-1).cpu().numpy()

error_data = []
for model_idx in range(num_models):
    print(f'Seed {model_idx + 1}/{num_models}')
    simulation_model = OjaModel(
        device='mps',
        num_tasks=5,
        num_context_dependent_hidden_units=128,
        alpha_ema=alpha,
    )
    save_file = f'oja_model-{model_idx}.torch'

    if save_file in os.listdir(model_path):
        print(f'  loading cached model from {save_file}')
        simulation_model.load_state_dict(torch.load(os.path.join(model_path, save_file)))
    else:
        print(f'  training to criterion...')
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict())
        c = 0
        errors = calc_model_error(simulation_model,train_x,train_y)

        while errors.mean() > 0.18:
            simulation_model.train(train_x, train_y, epochs=1, batch_size=38, is_blocked=True)
            errors = calc_model_error(simulation_model,train_x,train_y)
            c += 1
        print(f'  converged at epoch {c}, error {errors.mean():.4f}')
        torch.save(simulation_model.state_dict(), os.path.join(model_path, save_file))

    print(f'  evaluating...')
    preds = simulation_model(val_x_eval)
    mae = torch.abs(preds - val_y)[:, [2541, 2542]].mean(axis=-1).cpu().detach().numpy()
    accs = ((preds[:, 2541] > preds[:, 2542]) == val_y[:, 2541]).float().cpu().detach().numpy()

    error_data.append(pd.DataFrame({
        'model': [model_idx] * len(accs),
        'rt': mae,
        'error': 1 - accs,
        'size_condition': size_conditions,
        'cat_condition': cat_conditions,
        'rand_condition': random_cat_conditions,
        'block_type': blocks,
    }))

error_data = pd.concat(error_data, axis=0)
error_data.to_csv('data/oja_simulation_data_0200.csv', index=False)
print('Saved to data/oja_simulation_data_0200.csv')