import os
import warnings

import data

import numpy as np
import pandas as pd
import torch

from ..isc_model.model import load_isc_models
from .oja_variant_model import SimpleHebbianModel as OjaModel  # adjust import path as needed

warnings.filterwarnings('ignore')


def run_oja_simulation(
    num_models: int = 10,
    num_epochs: int = 40,
    alpha: float = np.linspace(0.05, 1.0, 30)[24],  # best α from EMA sweep
    lr_hebb: float = 1e-2,
    batch_size: int = 38,
    distractor_strength: float = 0.975,
    save_models: bool = False,
    output_csv: str = 'data/oja_simulation_data_0200.csv',
    model_path: str = 'models',
):
    print(f'Running Oja simulation: {num_models} seeds, alpha={alpha:.4f}, {num_epochs} epochs')
    print('Loading ISC base models...')
    isc_models = load_isc_models(num_models)

    print('Preparing experiment data...')
    train_x, train_y, size_conditions, cat_conditions, random_cat_conditions, _ = (
        data.make_behavioral_experiment_training_data(distractor_strength=distractor_strength)
    )

    def train_or_load(model_idx, is_blocked):
        sim = OjaModel(
            device='mps',
            num_tasks=5,
            num_context_dependent_hidden_units=128,
            alpha_ema=alpha,
        )
        tag = 'blocked' if is_blocked else 'interleaved'
        save_file = f'oja_{tag}-{model_idx}.torch'

        if save_file in os.listdir(model_path):
            print(f'  [seed {model_idx}, {tag}] loading cached model from {save_file}')
            sim.load_state_dict(torch.load(os.path.join(model_path, save_file)))
        else:
            print(f'  [seed {model_idx}, {tag}] training from scratch...')
            sim.load_old_model_weights(isc_models[model_idx].state_dict())
            sim.train(
                train_x, train_y,
                epochs=num_epochs,
                batch_size=batch_size,
                is_blocked=is_blocked,
            )
            if save_models:
                torch.save(sim.state_dict(), os.path.join(model_path, save_file))
                print(f'  [seed {model_idx}, {tag}] saved to {save_file}')
        return sim

    error_data = []
    for i in range(num_models):
        print(f'Seed {i + 1}/{num_models}')
        for is_blocked, tag in [(True, 'blocked'), (False, 'interleaved')]:
            sim = train_or_load(i, is_blocked)
            print(f'  [seed {i}, {tag}] evaluating...')
            preds = sim(train_x)
            mae = torch.abs(preds - train_y)[:, [2541, 2542]].mean(axis=-1)
            mae = mae.cpu().detach().numpy()
            accs = (
                (preds[:, 2541] > preds[:, 2542]) == train_y[:, 2541]
            ).float().cpu().detach().numpy()

            error_data.append(pd.DataFrame({
                'model': [i] * len(accs),
                'rt': mae,
                'error': 1 - accs,
                'size_condition': size_conditions,
                'cat_condition': cat_conditions,
                'random_cat_condition': random_cat_conditions,
                'block_type': [tag] * len(accs),
            }))

    print('Concatenating results...')
    error_data = pd.concat(error_data, axis=0)
    error_data.to_csv(output_csv, index=False)
    print(f'Saved to {output_csv}')
    return error_data


if __name__ == '__main__':
    run_oja_simulation()