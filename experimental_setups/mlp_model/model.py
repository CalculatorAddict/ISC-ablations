import data
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import utils


class BCEMetric:
    """
    Binary cross-entropy metric for tracking during training.
    """
    def __init__(self) -> None:
        self.name='bce'
        self.values = []
        self.fn = nn.BCELoss()


    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor, model=None) -> list[np.array]:
        self.values.append(self.fn(y,y_hat).cpu().detach().numpy())
    

class MLPModel(nn.Module):
    """
    Creates a one-layer fully-connected Multi-layer Perceptron Model.

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
    """
    def __init__(self, num_objects: int = 350, num_hub_hidden_units: int = 64,
                 num_context_dependent_hidden_units: int = 128,
                 num_task_context_units: int = 16,
                 num_output: int = 2541+2+3+350,
                 num_tasks: int = 36, lr: float = .05,
                 device=None, biases: bool = True, hidden_layers: int = 1) -> None:
        super().__init__()
        if device is None:
            device = utils.set_torch_device()

        self.input_to_embedding_weights = nn.Linear(num_objects+num_tasks,num_context_dependent_hidden_units,device=device,bias=biases)
        if hidden_layers == 1:
            self.embedding_to_output_weights = nn.Linear(num_context_dependent_hidden_units,num_output,device=device,bias=biases)
        else:
            self.embedding_to_output_weights = nn.Linear(num_context_dependent_hidden_units,num_output,device=device,bias=biases)
            self.embedding_to_output_weights = nn.Linear(num_context_dependent_hidden_units,num_output,device=device,bias=biases)

        nn.init.uniform_(self.input_to_embedding_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.embedding_to_output_weights.weight,a=-.01,b=.01)

        if biases:
            nn.init.uniform_(self.input_to_embedding_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.embedding_to_output_weights.bias,a=-.01,b=.01)
        else:
            pass
            # with torch.no_grad():
            #     self.hub_to_output_weights.bias.copy_(torch.ones(self.hub_to_output_weights.bias.shape,device=device)*-2)

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


    def forward(self, x: torch.Tensor, take_sigmoid: bool=True, noise: float=0) -> torch.Tensor:
        concatenated_inputs = torch.cat(x[0],x[1])
        
        if noise:
            embedding = self.input_to_embedding_weights(concatenated_inputs)
            output = self.embedding_to_output_weights(torch.sigmoid(embedding))
            if take_sigmoid:
                output = torch.sigmoid(output)+torch.randn_like(output)*noise
            return output

        embedding = torch.sigmoid(self.input_to_embedding_weights(concatenated_inputs))
        output = self.embedding_to_output_weights(embedding)
        if take_sigmoid:
            output = torch.sigmoid(output)
        return output


    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1, batch_size: int = 64) -> list:
        if epochs < 1:
            return [0]
        for metric in self.metrics:
            metric(self(x),y,self)
        for epoch in range(epochs):
            n_steps = 0
            batch_idxs = np.random.permutation(range(len(y)))
            for batch_start in range(0,len(y),batch_size):
                batch_idx = batch_idxs[batch_start:min(batch_start+batch_size,len(y))]
                self.optimizer.zero_grad()
                output = self([x[0][batch_idx],x[1][batch_idx]],take_sigmoid=False)
                loss = self.loss_fn(output,y[batch_idx])
                loss.backward()
                self.optimizer.step()
                n_steps += 1

            for metric in self.metrics:
                metric(self(x),y,self)
        return self.metrics
    

    def plot_metrics(self) -> None:
        for metric in self.metrics:
            print(metric.name)
            if type(metric.values) is list:
                fig = px.line(y=metric.values)
            else:
                fig = px.line(metric.values,x='x',y='y',color='color')
            fig.show()