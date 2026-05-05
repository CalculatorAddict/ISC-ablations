import os, sys
import warnings

import random
import pandas as pd
import torch
from scipy.stats import ttest_rel

import data
import utils
from .. import isc_model
from .oja_variant_model import ErrorGatingModel

warnings.filterwarnings('ignore')

model_path = 'models'
num_bootstrap_sims = 10000
num_training_epochs = 30
num_training_epochs_comparison = 10
num_epochs = 40 # Number of epochs of finetuning
train_model = False
train_comparison_model = False

num_objects = 350
num_tasks = 36
num_task_context_units = 16
num_context_independent_units = 64
num_context_dependent_units = 128
size_idx = 2541  # Index of the "is_small" feature
size_task_idx = 33  # Index of the size task

def prepare_experiment_data():
    train_x,train_y,_,_,_,_ = data.make_behavioral_experiment_training_data(distractor_strength=.975)

    # filter data to always contain category context
    train_x = [train_x[0][:20*19], train_x[1][:20*19]]
    train_y = train_y[:20*19]

    # reorder data to have two blocks
    mask = train_x[1][:,0]==1
    train_x = [torch.cat((train_x[0][mask],train_x[0][~mask])),torch.cat((train_x[1][mask],train_x[1][~mask]))]
    train_y = torch.cat((train_y[mask],train_y[~mask]))

    return train_x, train_y

def calc_model_error(model,train_x,train_y,noise=0):
    errors = torch.abs(model(train_x,noise=noise)-train_y)[:,[2541,2542]].mean(axis=-1)
    return errors.cpu().detach().numpy()

def run_error_experiment(
        num_epochs: int = 40,
        num_models: int = 10,
        alpha: float = 0.05,
        lr_hebb: float = 10e-3,
        batch_size: int = 38,
        save_models: bool = True,
        unfreeze_task_to_cd_weights: bool=False,
        track_learning_curves: bool = False,
        device = None,
        ):
    if device is None:
        device = utils.set_torch_device()
    device = torch.device(device)

    isc_models = isc_model.load_isc_models(num_models, device=device)

    train_x, train_y = prepare_experiment_data()
    train_x = [t.to(device) for t in train_x]
    train_y = train_y.to(device)
    error_data = []
    lc_data = []

    simulation_models_interleaved = []
    for model_idx in range(num_models):
        simulation_model = ErrorGatingModel(device=device,num_tasks=5,num_context_dependent_hidden_units=128)
        interl_save_file = f'oja_a{alpha:.2f}_interl-{model_idx}.torch'

        if interl_save_file in os.listdir('models/alpha_grid_search'):
            simulation_model.load_state_dict(
                torch.load(os.path.join('models','alpha_grid_search',interl_save_file), map_location=device),
                strict=False,
            )
        else:
            simulation_model.load_old_model_weights(isc_models[model_idx].state_dict())
            simulation_model.train(train_x, train_y, epochs = num_epochs, batch_size = batch_size, is_blocked=False)
            if save_models:
                torch.save(simulation_model.state_dict(),os.path.join('models','alpha_grid_search',interl_save_file))

        simulation_models_interleaved += [simulation_model]
        if track_learning_curves:
            bce_values = simulation_model.metrics[0].values
            lc_data.append(pd.DataFrame({
                'model': model_idx,
                'epoch': range(len(bce_values)),
                'bce': bce_values,
                'condition': 'interleaved',
                'architecture': 'Hebbian',
            }))

    for i in range(num_models):
        preds = simulation_models_interleaved[i](train_x)
        accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
        error_data.append(pd.DataFrame({
            'model':[i]*len(accs),
            'acc':accs,
            'architecture':['Hebbian']*len(accs),
            'condition':['interleaved']*len(accs),
            'alpha':[alpha]*len(accs),
            'lr_hebb':[lr_hebb]*len(accs),
        }))


    simulation_models_blocked = []
    for model_idx in range(num_models):
        simulation_model = ErrorGatingModel(device=device,num_tasks=5,num_context_dependent_hidden_units=128)
        blockd_save_file = f'oja_a{alpha:.2f}_blockd-{model_idx}.torch'

        if blockd_save_file in os.listdir('models/alpha_grid_search'):
            simulation_model.load_state_dict(
                torch.load(os.path.join('models','alpha_grid_search',blockd_save_file), map_location=device),
                strict=False,
            )
        else:
            simulation_model.load_old_model_weights(isc_models[model_idx].state_dict())
            simulation_model.train(train_x, train_y, epochs = num_epochs, batch_size = batch_size, is_blocked=True)
            if save_models:
                torch.save(simulation_model.state_dict(),os.path.join('models','alpha_grid_search',blockd_save_file))

        simulation_models_blocked += [simulation_model]
        if track_learning_curves:
            bce_values = simulation_model.metrics[0].values
            lc_data.append(pd.DataFrame({
                'model': model_idx,
                'epoch': range(len(bce_values)),
                'bce': bce_values,
                'condition': 'blocked',
                'architecture': 'Hebbian',
            }))

    for i in range(num_models):
        preds = simulation_models_blocked[i](train_x)
        accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
        error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'architecture':['Hebbian']*len(accs),'condition':['blocked']*len(accs),'alpha':[alpha]*len(accs)}))

    if track_learning_curves:
        return error_data, lc_data
    return error_data


def run_error_baseline(num_epochs: int = 40, num_models: int = 10, device = None):
    if device is None:
        device = utils.set_torch_device()
    device = torch.device(device)

    isc_models = isc_model.load_isc_models(num_models, device=device)

    train_x, train_y = prepare_experiment_data()
    train_x = [t.to(device) for t in train_x]
    train_y = train_y.to(device)
    error_data = []

    mask = train_x[1][:,0]==1
    train_x_animal = [train_x[0][mask], train_x[1][mask]]
    train_y_animal = train_y[mask]
    train_x_instrument = [train_x[0][~mask], train_x[1][~mask]]
    train_y_instrument = train_y[~mask]

    baseline_models_interleaved = []
    for model_idx in range(num_models):
        simulation_model = isc_model.model.ISCModel(device=device,num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
        save_file = f'isc_interl-{model_idx}.torch'
        if save_file in os.listdir('models'):
            simulation_model.load_state_dict(torch.load(os.path.join('models',save_file), map_location=device))
        else:
            simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
            simulation_model.train(train_x,train_y,epochs=num_epochs, batch_size=1)
            # torch.save(simulation_model.state_dict(),os.path.join('models',save_file))
        baseline_models_interleaved += [simulation_model]
        
    for i in range(num_models):
        preds = baseline_models_interleaved[i](train_x)
        accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
        error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'architecture':['ISC']*len(accs),'condition':['interleaved']*len(accs)}))


    baseline_models_blocked = []
    for model_idx in range(num_models):
        simulation_model = isc_model.model.ISCModel(device=device,num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
        save_file = f'isc_blockd-{model_idx}.torch'
        if save_file in os.listdir('models'):
            simulation_model.load_state_dict(torch.load(os.path.join('models',save_file), map_location=device))
        else:
            simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
            if random.random() < 0.5:
                simulation_model.train(train_x_animal,train_y_animal,epochs=num_epochs, batch_size=1)
                simulation_model.train(train_x_instrument,train_y_instrument,epochs=num_epochs, batch_size=1)
            else:
                simulation_model.train(train_x_instrument,train_y_instrument,epochs=num_epochs, batch_size=1)
                simulation_model.train(train_x_animal,train_y_animal,epochs=num_epochs, batch_size=1)
            # torch.save(simulation_model.state_dict(),os.path.join('models',save_file))
        baseline_models_blocked += [simulation_model]

    for i in range(num_models):
        preds = baseline_models_blocked[i](train_x)
        accs = ((preds[:,2541]>preds[:,2542])==train_y[:,2541]).float().cpu().detach().numpy()
        error_data.append(pd.DataFrame({'model':[i]*len(accs),'acc':accs,'architecture':['ISC']*len(accs),'condition':['blocked']*len(accs)}))

    return error_data

if __name__=='__main__':
    error_data = []
    error_data.append(run_error_experiment())
    error_data.append(run_error_baseline())

    error_data = pd.concat(error_data,axis=0)
    error_data.to_csv(f'data/error_simulation_n{num_epochs}.csv')
