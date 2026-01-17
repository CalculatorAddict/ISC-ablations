import os
import warnings

import data
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

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

train_x,train_y,size_conditions,cat_conditions,random_cat_conditions,blocks = data.make_behavioral_experiment_training_data(distractor_strength=.975)

def calc_model_error(model,train_x,train_y,noise=0):
    errors = torch.abs(model(train_x,noise=noise)-train_y)[:,[2541,2542]].mean(axis=-1)
    return errors.cpu().detach().numpy()

error_data = []
for model_idx in range(1):
    simulation_model = model.ISCModel(device='mps',num_tasks=5,num_task_context_units=16,num_context_dependent_hidden_units=128)
    save_file = f'isc_model-{model_idx}-bhvsim.torch'
    if save_file in os.listdir('models'):
        simulation_model.load_state_dict(torch.load(os.path.join('models',save_file)))
    else:
        simulation_model.load_old_model_weights(isc_models[model_idx].state_dict(),use_old_size_starting_point = True)
        errors = calc_model_error(simulation_model,train_x,train_y)
        c=1
        while errors.mean() > 0.18:
            simulation_model.train(train_x,train_y,epochs=1)
            errors = calc_model_error(simulation_model,train_x,train_y)
            c+=1
        print(model_idx, errors.mean(),c)
        torch.save(simulation_model.state_dict(),os.path.join('models',save_file))

# indices of stimuli used for behavioral simulation
animal_indices = [118,104,30,48,116,105,115,29,57,59]
instrument_indices = [248,252,261,263,257,262,253,266,267,260]

experiment_stimulus_indices = animal_indices+instrument_indices

# want context dependent reps for blocked setting
embeddings = simulation_model.get_context_dependent_reps(experiment_stimulus_indices)[1]
X = np.asarray(embeddings)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
plt.scatter(X_pca[:10, 0], X_pca[:10, 1], color='red', label='Animals')
plt.scatter(X_pca[10:, 0], X_pca[10:, 1], color='blue', label='Instruments')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Embeddings (blocked)")
plt.legend()
plt.show()

# want context dependent reps for interleaved setting
embeddings = simulation_model.get_context_dependent_reps(experiment_stimulus_indices)[1]
X = np.asarray(embeddings)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
plt.scatter(X_pca[:10, 0], X_pca[:10, 1], color='red', label='Animals')
plt.scatter(X_pca[10:, 0], X_pca[10:, 1], color='blue', label='Instruments')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Embeddings (blocked)")
plt.legend()
plt.show()