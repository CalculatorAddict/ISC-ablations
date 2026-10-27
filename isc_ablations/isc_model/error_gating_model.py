import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

from isc_ablations.common import data
import numpy as np
import random
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from isc_ablations.common import utils
import sys
from .model import BCEMetric

class ErrorGatingISCModel(nn.Module):
    """
    Creates a Controlled Semantic Cognition (CSC) Model.

    Parameters
    ----------
    num_objects (int): Number of objects/inputs for the model. Default: 350
    num_hub_hidden_units (int): Number of hidden units in the hub layer. Default: 64
    num_context_dependent_hidden_units (int): Number of hidden units in the context-dependent layer. Default: 128
    num_task_context_units (int): Number of hidden units in the task context layer. Default: 16
    num_output (int): Number of output units. Default: 2541+2+3+350
    num_tasks (int): Number of tasks. Default: 36
    lr (float): Learning rate for the model. Default: 0.05
    device (str): Device to use for training. Default: None
    biases (bool): If ``True``, uses biases in the model. Default: True

    Attributes
    ----------
    item_input_to_hub_weights (torch.nn.Linear): Weights from the item input to the hub.
    context_input_to_task_context_rep_weights (torch.nn.Linear): Weights from the context input to the task context layer.
    task_context_rep_to_context_dependent_rep_weights (torch.nn.Linear): Weights from the task context layer to the context dependent layer.
    hub_to_context_dependent_rep_weights (torch.nn.Linear): Weights from the hub to the context dependent layer.
    context_dependent_rep_to_output_weights (torch.nn.Linear): Weights from the context dependent layer to the output layer.
    hub_to_output_weights (torch.nn.Linear): Weights from the hub to the output layer.
    loss_fn (torch.nn.BCEWithLogitsLoss): Loss function for the model.
    optimizer (torch.optim.Adam): Optimizer for the model.
    metrics (list): List of metrics to track during training.
    num_objects (int): Number of objects/inputs for the model.
    num_tasks (int): Number of tasks for the model.
    num_context_dependent_hidden_units (int): Number of hidden units in the context-dependent layer.
    device (str): Device to use for training.

    Methods
    -------
    freeze_weights()
        Freezes the weights of the model.
    load_old_model_weights(state_dict,use_old_size_starting_point=True)
        Loads weights from a previous model.
    get_context_independent_rep(x)
        Gets the context-independent representation of the model for a given input.
    get_task_context_rep(x)
        Gets the task context representation of the model for a given input.
    get_context_dependent_rep(x)
        Gets the context-dependent representation of the model for a given input.
    forward(x,take_sigmoid=True)
        Forward pass of the model.
    train(x,y,epochs=1,batch_size=64)
        Trains the model.
    plot_metrics()
        Plots the metrics of the model.
    get_task_context_reps()
        Gets the task context representations of the model for all input combinations.
    get_context_independent_reps()
        Gets the context-independent representations of the model for all input combinations.
    get_context_dependent_reps()
        Gets the context-dependent representations of the model for all input combinations.
    """
    def __init__(self, num_objects: int = 350, num_hub_hidden_units: int = 64,
                 num_context_dependent_hidden_units: int = 128,
                 num_task_context_units: int = 16,
                 num_output: int = 2541+2+3+350,
                 num_tasks: int = 36, lr: float = .05,
                 device=None, biases: bool = True) -> None:
        super().__init__()
        if device is None:
            device = utils.set_torch_device()

        self.item_input_to_hub_weights = nn.Linear(num_objects,num_hub_hidden_units,device=device,bias=biases)
        self.context_input_to_task_context_rep_weights = nn.Linear(num_tasks,num_task_context_units,device=device,bias=biases)
        self.task_context_rep_to_context_dependent_rep_weights = nn.Linear(num_task_context_units,num_context_dependent_hidden_units,device=device,bias=biases)
        self.hub_to_context_dependent_rep_weights = nn.Linear(num_hub_hidden_units,num_context_dependent_hidden_units,device=device,bias=biases)
        self.context_dependent_rep_to_output_weights = nn.Linear(num_context_dependent_hidden_units,num_output,device=device,bias=biases)
        self.hub_to_output_weights = nn.Linear(num_hub_hidden_units,num_output,device=device)
        
        # Match the task-context representation used by SimpleHebbianModel so
        # its context-to-hidden weights can be loaded directly.
        initial_context = torch.zeros((1, num_tasks), dtype=torch.float, device=device)
        initial_context[0, 0] = 1.0
        if num_tasks > 1:
            initial_context[0, 1] = 0.0
        self.register_buffer(
            "context",
            initial_context,
        )

        nn.init.uniform_(self.item_input_to_hub_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.context_input_to_task_context_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.task_context_rep_to_context_dependent_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.hub_to_context_dependent_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.context_dependent_rep_to_output_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.hub_to_output_weights.weight,a=-.01,b=.01)

        if biases:
            nn.init.uniform_(self.item_input_to_hub_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.context_input_to_task_context_rep_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.task_context_rep_to_context_dependent_rep_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.hub_to_context_dependent_rep_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.context_dependent_rep_to_output_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.hub_to_output_weights.bias,a=-.01,b=.01)
        else:
            with torch.no_grad():
                self.hub_to_output_weights.bias.copy_(torch.ones(self.hub_to_output_weights.bias.shape,device=device)*-2)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.metrics = [BCEMetric()]
        self.num_objects = num_objects
        self.num_tasks = num_tasks
        self.num_context_dependent_hidden_units = num_context_dependent_hidden_units
        self.device = device


    def freeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = False


    def load_old_model_weights(self, state_dict: dict, use_old_size_starting_point: bool=True) -> None:
        for name, param in state_dict.items():
            if name not in ['context_input_to_task_context_rep_weights.weight','context_input_to_task_context_rep_weights.bias']:
                self.state_dict()[name].copy_(param)
                param.requires_grad = False
        if use_old_size_starting_point:
            old_size_weights = state_dict['context_input_to_task_context_rep_weights.weight'][:,-3]
            old_size_bias = state_dict['context_input_to_task_context_rep_weights.bias'][-3]
            with torch.no_grad():
                for i in range(self.num_tasks):
                    self.context_input_to_task_context_rep_weights.weight[:,i] = old_size_weights
                    self.context_input_to_task_context_rep_weights.bias[i] = old_size_bias


    def get_context_independent_rep(self, x: torch.Tensor) -> torch.Tensor:
        hub_rep = torch.sigmoid(self.item_input_to_hub_weights(x[0]))
        return hub_rep


    def get_task_context_rep(self, x: torch.Tensor) -> torch.Tensor:
        task_context_rep = torch.sigmoid(self.context_input_to_task_context_rep_weights(x[1]))
        return task_context_rep


    def get_context_dependent_rep(self, x: torch.Tensor) -> torch.Tensor:
        hub_rep = self.get_context_independent_rep(x)
        task_context_rep = self.get_task_context_rep(x)
        item_in_context_rep = torch.sigmoid(self.task_context_rep_to_context_dependent_rep_weights(task_context_rep)+
                                            self.hub_to_context_dependent_rep_weights(hub_rep))
        return item_in_context_rep


    def forward(self, x: torch.Tensor, take_sigmoid: bool=True, noise: float=0) -> torch.Tensor:
        hub_rep = torch.sigmoid(self.item_input_to_hub_weights(x[0]))
        task_context_rep = torch.sigmoid(self.context_input_to_task_context_rep_weights(self.context))
        item_in_context_rep = torch.sigmoid(self.task_context_rep_to_context_dependent_rep_weights(task_context_rep)+
                                            self.hub_to_context_dependent_rep_weights(hub_rep))
        output = self.hub_to_output_weights(hub_rep)+\
                 self.context_dependent_rep_to_output_weights(item_in_context_rep)
        if take_sigmoid:
            output = torch.sigmoid(output)

        self._update_context(x, output)

        return output
    

    def plot_metrics(self) -> None:
        for metric in self.metrics:
            print(metric.name)
            if type(metric.values) is list:
                fig = px.line(y=metric.values)
            else:
                fig = px.line(metric.values,x='x',y='y',color='color')
            fig.show()


    def get_task_context_reps(self) -> np.array:
        item_x = torch.zeros((self.num_tasks,self.num_objects),device=self.device)
        context_x = torch.eye(self.num_tasks,device=self.device)
        tc_reps = self.get_task_context_rep([item_x,context_x]).cpu().detach().numpy()
        return tc_reps
    

    def get_context_independent_reps(self, indices = None) -> np.array:
        """
        Returns context-independent representations of the items corresponding to indices
        
        Parameters:
            indices (list): A list of relevant array indices, whose representations will be returned. Default is all indices.
        
        Returns a np.array with the embeddings.
        """
        if indices is None:
            indices = range(len(self.num_objects))
        item_x = torch.eye(self.num_objects,device=self.device)[indices]
        context_x = torch.zeros((len(indices),self.num_tasks),device=self.device)
        ind_reps = self.get_context_independent_rep([item_x,context_x]).cpu().detach().numpy()
        return ind_reps
    

    def get_context_dependent_reps(self, indices = None) -> np.array:
        if indices is None:
            indices = range(len(self.num_objects))
        item_x = torch.eye(self.num_objects,device=self.device)[indices]
        context_x = torch.zeros((len(indices),self.num_tasks),device=self.device)
        dep_reps = np.zeros((self.num_tasks, len(indices), self.num_context_dependent_hidden_units))
        for context in range(self.num_tasks):
            context_x = torch.zeros((len(indices),self.num_tasks),device=self.device)
            context_x[:,context] = 1
            dep_reps[context] = self.get_context_dependent_rep([item_x,context_x]).cpu().detach().numpy()
        return dep_reps

    def _context_for_batch(self, batch_size: int) -> torch.Tensor:
        return self.context.detach().expand(batch_size, -1).clone()

    def _determine_true_label(self, x):
        """
        Determine true label for input to distractor task.

        Pre: x must have only one fully-activated one-hot input (ie there is exactly one idx s.t. x[0][idx]>=1.0)
        """

        x_item = x[0]
        if x_item.dim() == 1:
            x_item = x_item.unsqueeze(0)

        small_idxs = torch.tensor([105,115,29,57,59,262,253,266,267,260], device=x_item.device)
        large_idxs = torch.tensor([118,104,30,48,116,248,252,261,263,257], device=x_item.device)

        idx = torch.argmax(x_item, dim=1)

        labels = torch.full(idx.shape, torch.nan, dtype=torch.float, device=x_item.device)
        labels[torch.isin(idx, small_idxs)] = 0.0
        labels[torch.isin(idx, large_idxs)] = 1.0

        if torch.isnan(labels).any():
            raise ValueError("Input x did not have any recognized size-experiment item")
            
        return labels

        

    def _update_context(self, x, output):
        y_hat = (output[:, 2542] > output[:, 2541]).float()
        y_true = self._determine_true_label(x)

        error_signal = (y_hat - y_true).abs().mean()

        # error signal gives probability of switching contexts
        if random.random() <= error_signal:
            self.context[0, :2] = self.context[0, :2].mul_(-1.)
