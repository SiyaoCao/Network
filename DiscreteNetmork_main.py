#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing necessary packages
get_ipython().system('pip install optuna')
import optuna
import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
from csv import writer
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[ ]:


from Scripts.GetData import getDataLoaders, loadData
from Scripts.Training import train
from Scripts.PlotResults import plotResults
from Scripts.SavedParameters import hyperparams
import pandas as pd


# In[ ]:


#setting plotting parameters
sns.set_style("darkgrid")
sns.set(font = "Times New Roman")
sns.set_context("paper")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt_kws = {"rasterized": True}


# In[ ]:


torch.set_default_dtype(torch.float32)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


datacase = int(input("Which datacase do you want to work with?\n"))
percentage_train = float(input("Which percentage of the dataset do you want to use for training? Choose among 0.1,0.2,0.4,0.8\n"))

print(f"\n\n Case with percentage_train={percentage_train} and datacase={datacase}\n\n")

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

class approximate_curve(nn.Module):
        def __init__(self, is_res = True, normalize = True, act_name='tanh', nlayers=3, hidden_nodes = 50, output_dim = 204):
            super().__init__()

            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
            self.act_dict = {"tanh":lambda x : torch.tanh(x),
                                "sigmoid":lambda x : torch.sigmoid(x),
                                "swish":lambda x : x*torch.sigmoid(x),
                                "relu":lambda x : torch.relu(x),
                                "lrelu":lambda x : F.leaky_relu(x)}
            self.is_norm = normalize
            self.is_res = is_res
            self.act = self.act_dict[act_name]
            self.nlayers = nlayers
            self.first = nn.Linear(8,hidden_nodes)
            self.linears = nn.ModuleList([nn.Linear(hidden_nodes,hidden_nodes) for i in range(self.nlayers)])
            self.last = nn.Linear(hidden_nodes,output_dim)

        def forward(self,x):

            if self.is_norm:
                x[:,0] = (x[:,0]-1.5)/1.5
                x[:,4] = (x[:,4]-1.5)/1.5
            x  = self.act(self.first(x))
            for i in range(self.nlayers):
                if self.is_res: #ResNet
                    x = x + self.act(self.linears[i](x))
                else: #MLP
                    x = self.act(self.linears[i](x))

            return self.last(x)
        
num_nodes, _,_ = loadData(datacase)

def define_model(trial):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    is_res = False
    normalize = True
    act_name = "tanh"
    nlayers = trial.suggest_int("n_layers", 0, 10)
    hidden_nodes = trial.suggest_int("hidden_nodes", 10, 1000)

    model = approximate_curve(is_res, normalize, act_name, nlayers, hidden_nodes,output_dim=int(4*(num_nodes-2)))
    return model

from torch.utils.data import Dataset, DataLoader

def objective(trial):

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # Generate the model
    model = define_model(trial)
    model.to(device);

    lr = 1e-3
    weight_decay = 0
    gamma = trial.suggest_float("gamma",0,1e-2)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    criterion = nn.MSELoss()

    batch_size = 32
    _, _, _, _,x_val,y_val,trainloader,_,valloader = getDataLoaders(batch_size, datacase, percentage_train)

    print("Current test with :\n\n")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("\n\n")

    epochs = 300
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(0.45*epochs), gamma = 0.1)
    
    loss = train(model,gamma,criterion,scheduler,optimizer,epochs,trainloader,valloader,device)
    print('Loss ',loss.item())
    error = 1000

    if not torch.isnan(loss):
        model.eval();

        learned_traj = np.zeros_like(y_val)
        
        bcs_val = torch.from_numpy(x_val.astype(np.float32)).to(device)
        learned_traj = model(bcs_val).detach().cpu().numpy()
        error = np.mean((learned_traj-y_val)**2)

        print(f"The error on the validation trajectories is: {error}.")

    #Saving the obtained results
    if trial.number == 0:
        labels = []
        for lab, _ in trial.params.items():
            labels.append(str(lab))
        labels.append("MSE")
        with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(labels)
            f_object.close()

    results = []
    for _, value in trial.params.items():
        results.append(str(value))

    results.append(error)

    with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()
    return error

optuna_study = input("Do you want to do hyperparameter test? Type yes or no: ")
params = {}
if optuna_study=="yes":
    optuna_study = True
else:
    optuna_study = False
if optuna_study:
    study = optuna.create_study(direction="minimize",study_name="Euler Elastica")
    study.optimize(objective, n_trials=5)
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    params = study.best_params

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

manual_input = False
if params=={}:
    # We can input them manually by uncommenting the lines below
    if manual_input:
        print("No parameters have been specified. Let's input them:\n\n")
        nlayers = int(input("How many layers do you want the network to have? "))
        hidden_nodes = int(input("How many hidden nodes do you want the network to have? "))
        weight_decay = float(input("What weight decay do you want to use? "))
        gamma = float(input("What value do you want for gamma? "))
        batch_size = int(input("What batch size do you want? "))

        params = {'n_layers': nlayers,
                'hidden_nodes': hidden_nodes,
                'gamma': gamma}
    else:
    # or we can use the combinations found by Optuna that yield the best results for the mentioned datacases
        params = hyperparams(datacase, percentage_train)
print(f'The hyperparameters yelding the best results for this case are: {params}')
def define_best_model():
    
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    normalize = True
    act = "tanh"
    nlayers = params["n_layers"]
    hidden_nodes = params["hidden_nodes"]
    is_res = False
    
    print("Nodes: ",hidden_nodes)
    
    model = approximate_curve(is_res, normalize, act, nlayers, hidden_nodes, int(4*(num_nodes-2)))

    return model
model = define_best_model()
model.to(device);

TrainMode = input("Train Mode True or False? Type 0 for False and 1 for True: ")=="1"
weight_decay = 0.
lr = 1e-3
gamma = params["gamma"]
nlayers = params["n_layers"]
hidden_nodes = params["hidden_nodes"]
batch_size = 32
epochs = 300
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(0.45*epochs), gamma = 0.1)
criterion = nn.MSELoss()
x_train, y_train, x_test, y_test, x_val, y_val,trainloader, testloader,valloader = getDataLoaders(batch_size, datacase, percentage_train)
model.to(device);

if TrainMode:
    loss = train(model,gamma,criterion,scheduler,optimizer,epochs,trainloader,valloader,device)
    if datacase == 1:
        torch.save(model.state_dict(), f'TrainedModels/BothEnds{percentage_train}data.pt')
    if datacase == 2:
        torch.save(model.state_dict(), f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt')
else:
    if datacase == 1:
        pretrained_dict = torch.load(f'TrainedModels/BothEnds{percentage_train}data.pt',map_location=device)
    if datacase == 2:
        pretrained_dict = torch.load(f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt',map_location=device)
    model.load_state_dict(pretrained_dict)
model.eval();

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

model.eval();

# printing the accuracies and plotting the results
plotResults(model, device, x_train, y_train, x_test, y_test, x_val, y_val, num_nodes, datacase, percentage_train, gamma, nlayers, hidden_nodes)


# In[ ]:


import time
test_bvs = torch.from_numpy(x_test.astype(np.float32))
initial_time = time.time()
preds = model(test_bvs)
final_time = time.time()
total_time = final_time-initial_time
print("Number of trajectories in the test set : ",len(test_bvs))
print("Total time to predict test trajectories : ",total_time)
print("Average time to predict test trajectories : ",total_time / len(test_bvs))

