import os, sys
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

train_x,train_y,size_conditions,cat_conditions,random_cat_conditions,blocks = data.make_behavioral_experiment_training_data(distractor_strength=.975)

def calc_model_error(model,train_x,train_y,noise=0):
    errors = torch.abs(model(train_x,noise=noise)-train_y)[:,[2541,2542]].mean(axis=-1)
    return errors.cpu().detach().numpy()

error_data = []
for model_idx in range(1):
    simulation_model = model.MLPModel(device='mps',num_tasks=5,num_context_dependent_hidden_units=128)
    save_file = f'batch1-bhvsim.torch'

    # model training
    errors = calc_model_error(simulation_model,train_x,train_y)
    c=1
    while errors.mean() > 0.18:
        simulation_model.train(train_x,train_y,epochs=1,batch_size=1)
        errors = calc_model_error(simulation_model,train_x,train_y)
        c+=1
    print(model_idx, errors.mean(),c)
    torch.save(simulation_model.state_dict(),os.path.join('models',save_file))

for i in range(71):
    errors = calc_model_error(simulation_model,train_x,train_y,noise=1.2)
    preds = simulation_model(train_x,noise=1.175)
    accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(errors),'rt':errors,'error':1-accs,
                                    'size_condition':size_conditions,'cat_condition':cat_conditions,
                                    'rand_condition':random_cat_conditions,'block_type':blocks}))
error_data = pd.concat(error_data,axis=0)
error_data.to_csv(f'data/batch1_simulation_data_0200.csv')
