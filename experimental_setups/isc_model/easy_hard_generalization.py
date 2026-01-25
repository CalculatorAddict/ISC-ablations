import os
import warnings

import data
import pandas as pd
import torch
from scipy.stats import ttest_rel

from . import model

warnings.filterwarnings('ignore')

model_path = 'models'
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

train_x_easy,train_y_easy,_,_,_,_ = data.make_behavioral_experiment_training_data(distractor_strength=.750)[:20*19]
train_x_hard,train_y_hard,_,_,_,_ = data.make_behavioral_experiment_training_data(distractor_strength=.975)[:20*19]

def calc_model_error(model,train_x,train_y,noise=0):
    errors = torch.abs(model(train_x,noise=noise)-train_y)[:,[2541,2542]].mean(axis=-1)
    return errors.cpu().detach().numpy()

# non-sequential curriculum
error_data = []
for model_idx in range(1):
    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-{model_idx}-easyhard-nonseq.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x_hard, train_y_hard, epochs = 2)
        torch.save(simulation_model.state_dict(),os.path.join('models',save_file))

for i in range(10):
    preds = simulation_model(train_x_hard,noise=1.175)
    accs = ((preds[:,2541]>preds[:,2542])==train_y_hard[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'curriculum':['non-sequential']*len(accs)}))

# interleaved curriculum
for model_idx in range(1):
    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-{model_idx}-easyhard-interl.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x_easy + train_x_hard, train_y_easy + train_y_hard, epochs = 1)
        torch.save(simulation_model.state_dict(),os.path.join('models',save_file))

for i in range(10):
    preds = simulation_model(train_x_hard,noise=1.175)
    accs = ((preds[:,2541]>preds[:,2542])==train_y_hard[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'curriculum':['interleaved']*len(accs)}))

# blocked curriculum
for model_idx in range(1):
    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-{model_idx}-easyhard-blockd.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x_easy, train_y_easy, epochs = 1)
        simulation_model.train(train_x_hard, train_y_hard, epochs = 1)
        torch.save(simulation_model.state_dict(),os.path.join('models',save_file))

for i in range(10):
    preds = simulation_model(train_x_hard,noise=1.175)
    accs = ((preds[:,2541]>preds[:,2542])==train_y_hard[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'curriculum':['seq-blocked']*len(accs)}))
error_data = pd.concat(error_data,axis=0)
error_data.to_csv(f'data/easyhard_curriculum_data.csv')
