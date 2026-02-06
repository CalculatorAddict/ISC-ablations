import os, sys
import warnings

import data
import pandas as pd
import torch
from scipy.stats import ttest_rel

from . import model
from .. import isc_model

warnings.filterwarnings('ignore')

model_path = 'models'
num_models = 10
num_bootstrap_sims = 10000
num_training_epochs = 30
num_training_epochs_comparison = 10
num_epochs = 10 # Number of epochs of finetuning
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

# filter data to always contain category context
train_x = [train_x[0][:20*19], train_x[1][:20*19]]
train_y = train_y[:20*19]

# reorder data to have two blocks
mask = train_x[1][:,0]==1
train_x = [torch.cat((train_x[0][mask],train_x[0][~mask])),torch.cat((train_x[1][mask],train_x[1][~mask]))]
train_y = torch.cat((train_y[mask],train_y[~mask]))

def calc_model_error(model,train_x,train_y,noise=0):
    errors = torch.abs(model(train_x,noise=noise)-train_y)[:,[2541,2542]].mean(axis=-1)
    return errors.cpu().detach().numpy()

error_data = []

isc_models = isc_model.model.load_isc_models(num_models)

simulation_models_interleaved = []
for model_idx in range(num_models):
    simulation_model = model.HebbianModel(device='mps',num_tasks=5,num_context_dependent_hidden_units=128)
    save_file = f'hebbian_interl-{model_idx}.torch'

    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x, train_y, epochs = num_epochs, batch_size = 38, is_blocked=False)
        # torch.save(simulation_model.state_dict(),os.path.join('models',save_file))

    simulation_models_interleaved += [simulation_model]
    sys.exit(0)

for i in range(num_models):
    preds = simulation_models_interleaved[i](train_x)
    accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'architecture':['Hebbian']*len(accs),'condition':['interleaved']*len(accs)}))


simulation_models_blocked = []
for model_idx in range(num_models):
    simulation_model = model.HebbianModel(device='mps',num_tasks=5,num_context_dependent_hidden_units=128)
    save_file = f'hebbian_blockd-{model_idx}.torch'

    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x, train_y, epochs = num_epochs, batch_size = 38, is_blocked=True)
        # torch.save(simulation_model.state_dict(),os.path.join('models',save_file))

    simulation_models_blocked += [simulation_model]

for i in range(num_models):
    preds = simulation_models_blocked[i](train_x)
    accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'architecture':['Hebbian']*len(accs),'condition':['blocked']*len(accs)}))


train_x_animal = [train_x[0][mask], train_x[1][mask]]
train_y_animal = train_y[mask]
train_x_instrument = [train_x[0][~mask], train_x[1][~mask]]
train_y_instrument = train_y[~mask]

baseline_models_interleaved = []
for model_idx in range(num_models):
    simulation_model = isc_model.model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_interl-{model_idx}.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x,train_y,epochs=num_epochs, batch_size=38)
        # torch.save(simulation_model.state_dict(),os.path.join('models',save_file))
    baseline_models_interleaved += [simulation_model]
    
for i in range(num_models):
    preds = baseline_models_interleaved[i](train_x)
    accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'architecture':['ISC']*len(accs),'condition':['interleaved']*len(accs)}))


baseline_models_blocked = []
for model_idx in range(num_models):
    simulation_model = isc_model.model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_blockd-{model_idx}.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        simulation_model.train(train_x_animal,train_y_animal,epochs=num_epochs, batch_size=38)
        simulation_model.train(train_x_instrument,train_y_instrument,epochs=num_epochs, batch_size=38)
        # torch.save(simulation_model.state_dict(),os.path.join('models',save_file))
    baseline_models_blocked += [simulation_model]

for i in range(num_models):
    preds = baseline_models_blocked[i](train_x)
    accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
    error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'architecture':['ISC']*len(accs),'condition':['blocked']*len(accs)}))

error_data = pd.concat(error_data,axis=0)
error_data.to_csv(f'data/hebbian_simulation_data.csv')
