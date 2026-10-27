'''
Easy-to-hard generalization replication study with 10 separate models trained for each curriculum type.

Noiseless variant of easy_hard_generalization. Writes data/02_easy_hard/easyhard_noiseless_n{num_models}.csv;
checkpoints models/02_easy_hard/ (not in git, created on first run).

Run from the repo root; the script runs on import (no __main__ guard):
    python -m isc_ablations.isc_model.easy_hard_noiseless
Used by: experiments/02_easy_hard/easy_hard_curriculum.ipynb.
'''

import os
import warnings

from isc_ablations.common import data
import pandas as pd
import torch

from . import model

warnings.filterwarnings('ignore')

model_path = 'models/pretrained'
easyhard_model_path = 'models/02_easy_hard'  # not tracked in git; created on first run
os.makedirs(easyhard_model_path, exist_ok=True)
num_models = 10
num_bootstrap_sims = 10000
num_training_epochs = 30
num_training_epochs_comparison = 10
train_model = False
train_comparison_model = False

num_objects = 350
num_tasks = 36
num_task_context_units = 16
num_context_independent_units = 64
num_context_dependent_units = 128
size_idx = 2541  # Index of the "is_small" feature
size_task_idx = 33  # Index of the size task

isc_models = model.load_isc_models(num_models, train_model, num_training_epochs, model_path)

train_x_easy,train_y_easy,_,_,_,_ = data.make_behavioral_experiment_training_data(distractor_strength=.750)
train_x_hard,train_y_hard,_,_,_,_ = data.make_behavioral_experiment_training_data(distractor_strength=.975)

# filter inputs to always contain context
train_x_easy = [train_x_easy[0][:20*19], train_x_easy[1][:20*19]]
train_x_hard = [train_x_hard[0][:20*19], train_x_hard[1][:20*19]]

# filter outputs to match inputs
train_y_easy = train_y_easy[:20*19]
train_y_hard = train_y_hard[:20*19]

def calc_model_error(model,train_x,train_y,noise=0):
    errors = torch.abs(model(train_x,noise=noise)-train_y)[:,[2541,2542]].mean(axis=-1)
    return errors.cpu().detach().numpy()

# non-sequential curriculum
error_data = []
nonseq_models = []
for model_idx in range(num_models):
    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-{model_idx}-easyhard-nonseq.torch'
    if save_file in os.listdir(easyhard_model_path):
        simulation_model.load_state_dict(torch.load(os.path.join(easyhard_model_path,save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x_hard, train_y_hard, epochs = 2)
        torch.save(simulation_model.state_dict(),os.path.join(easyhard_model_path,save_file))
    nonseq_models += [simulation_model]

for i in range(num_models):
    preds = nonseq_models[i](train_x_hard)
    accs = ((preds[:,2541]>preds[:,2542])==train_y_hard[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'curriculum':['non-sequential']*len(accs)}))

# interleaved curriculum
interl_models = []
for model_idx in range(num_models):
    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-{model_idx}-easyhard-interl.torch'
    if save_file in os.listdir(easyhard_model_path):
        simulation_model.load_state_dict(torch.load(os.path.join(easyhard_model_path,save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)

        train_x = [torch.cat(p) for p in zip(train_x_easy, train_x_hard)]
        train_y = torch.cat((train_y_easy, train_y_hard))

        simulation_model.train(train_x, train_y, epochs = 1)
        torch.save(simulation_model.state_dict(),os.path.join(easyhard_model_path,save_file))
    interl_models += [simulation_model]

for i in range(num_models):
    preds = interl_models[i](train_x_hard)
    accs = ((preds[:,2541]>preds[:,2542])==train_y_hard[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'curriculum':['interleaved']*len(accs)}))

# blocked curriculum
blockd_models = []
for model_idx in range(num_models):
    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-{model_idx}-easyhard-blockd.torch'
    if save_file in os.listdir(easyhard_model_path):
        simulation_model.load_state_dict(torch.load(os.path.join(easyhard_model_path,save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x_easy, train_y_easy, epochs = 1)
        simulation_model.train(train_x_hard, train_y_hard, epochs = 1)
        torch.save(simulation_model.state_dict(),os.path.join(easyhard_model_path,save_file))
    blockd_models += [simulation_model]

for i in range(num_models):
    preds = blockd_models[i](train_x_hard)
    accs = ((preds[:,2541]>preds[:,2542])==train_y_hard[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'curriculum':['seq-blocked']*len(accs)}))
error_data = pd.concat(error_data,axis=0)
error_data.to_csv(f'data/02_easy_hard/easyhard_noiseless_n{num_models}.csv')
