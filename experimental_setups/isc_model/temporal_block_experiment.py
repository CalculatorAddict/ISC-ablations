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

train_x,train_y,size_conditions,cat_conditions,random_cat_conditions = data.make_altered_experiment_training_data(distractor_strength=.975)

def calc_model_error(model,train_x,train_y,noise=0):
    errors = torch.abs(model(train_x,noise=noise)-train_y)[:,[2541,2542]].mean(axis=-1)
    return errors.cpu().detach().numpy()

def load_blocked():
    # use the category context to split into animal and instrument blocks
    zero_context = torch.zeros((190,5),dtype=torch.float,device=data.set_torch_device())

    mask = train_x[1][:,0]==1
    train_x_animal = [train_x[0][mask], zero_context]
    train_y_animal = train_y[mask]
    train_x_instrument = [train_x[0][~mask], zero_context]
    train_y_instrument = train_y[~mask]

    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-blocked.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[0].state_dict(),use_old_size_starting_point = True)
        errors = calc_model_error(simulation_model,train_x,train_y)
        c=1
        while errors.mean() > 0.18:
            simulation_model.train(train_x_animal,train_y_animal,epochs=1)
            simulation_model.train(train_x_instrument,train_y_instrument,epochs=1)
            errors = calc_model_error(simulation_model,train_x,train_y)
            c+=1
        print(0, errors.mean(),c)
        torch.save(simulation_model.state_dict(),os.path.join('models',save_file))
    
    return simulation_model

def load_interleaved():
    # remove the category context
    zero_context = torch.zeros((380,5),dtype=torch.float,device=data.set_torch_device())
    train_x_interleaved = [train_x[0], zero_context]

    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-interleaved.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[0].state_dict(),use_old_size_starting_point = True)
        errors = calc_model_error(simulation_model,train_x,train_y)
        c=1
        while errors.mean() > 0.18:
            simulation_model.train(train_x_interleaved,train_y,epochs=1)
            errors = calc_model_error(simulation_model,train_x,train_y)
            c+=1
        print(0, errors.mean(),c)
        torch.save(simulation_model.state_dict(),os.path.join('models',save_file))
    
    return simulation_model

if __name__=='__main__':
    blocked_model = load_blocked()
    interleaved_model = load_interleaved()

    error_data = []
    for i in range(71):
        # test the blocked model
        errors = calc_model_error(blocked_model,train_x,train_y,noise=1.2)
        preds = blocked_model(train_x,noise=1.175)
        accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
        error_data.append(pd.DataFrame({'model':[i]*len(errors),'rt':errors,'error':1-accs,
                                        'size_condition':size_conditions,'cat_condition':cat_conditions,
                                        'random_cat_condition':random_cat_conditions,'block_type':['blocked']*len(errors)}))

        # test the interleaved model
        errors = calc_model_error(interleaved_model,train_x,train_y,noise=1.2)
        preds = interleaved_model(train_x,noise=1.175)
        accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
        error_data.append(pd.DataFrame({'model':[i]*len(errors),'rt':errors,'error':1-accs,
                                        'size_condition':size_conditions,'cat_condition':cat_conditions,
                                        'random_cat_condition':random_cat_conditions,'block_type':['interleaved']*len(errors)}))
    error_data = pd.concat(error_data,axis=0)
    error_data.to_csv(f'data/isc_temporal_data_0200.csv')