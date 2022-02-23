import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances

from .qsne_utils import _utils

class ParametricQSNE(object):
    def __init__(self, model, optimizer, criterion,
                 q=2.0, perplexity=30., metric="euclidean"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.q = q
        self.perp = perplexity
        self.metric = metric
        
    def fit(self, EPOCH, trainloader, validation_mode=True,
            scheduler=None, device="cuda:0"):
        losses = {"train":[]}
        for epoch in range(EPOCH):
            print(f"epoch{epoch+1}")
            self.train(trainloader, device, self.metric)
            if validation_mode:
                print("Training data results-----------------------------")
                loss = self.test(trainloader, device, self.metric)
                losses["train"].append(loss)
            if scheduler is not None:
                scheduler.step()
        return losses
    
    def train(self, dataloader, device="cuda:0", metric="euclidean"):
        device = torch.device(device)
        self. model.train()
        for (inputs, _) in tqdm(dataloader):
            self.optimizer.zero_grad()
            P = self.joint_probabilities(inputs.view(inputs.shape[0], -1).numpy(),
                                         metric).to(device)
            inputs = inputs.to(device)
            
            outputs = self.model(inputs)
            Q = self.low_probabilities(outputs)
            
            loss = self.criterion(Q.log(), P)
            
            loss.backward()
            self.optimizer.step()
    
    def test(self, dataloader, device="cuda:0", metric="euclidean"):
        device = torch.device(device)
        sum_loss = 0.
        
        self.model.eval()
        for (inputs, _) in tqdm(dataloader):
            P = self.joint_probabilities(inputs.view(inputs.shape[0], -1).numpy(),
                                         metric).to(device)
            inputs = inputs.to(device)
            
            outputs = self.model(inputs)
            Q = self.low_probabilities(outputs)
            
            loss = self.criterion(Q.log(), P)
            
            sum_loss += loss.item()*inputs.shape[0]
        sum_loss /= len(dataloader.dataset)
        print(f"mean_loss={sum_loss}")
        
        return sum_loss
    
    def getOutputs(self, dataloader, based_labels=None, device="cuda:0"):
        if based_labels is None:
            based_labels = [str(x) for x in np.unique(labels)]
        data_dict = dict(zip(based_labels, [[] for i in range(len(based_labels))]))
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)
            
            for data, label in zip(outputs, labels):
                data_dict[based_labels[label]].append(data.cpu().detach().numpy())
            
        for key in based_labels:
            data_dict[key] = np.vstack(data_dict[key])
        return data_dict
            
            
    def joint_probabilities(self, inputs, metric="euclidean", verbose=0):
        if metric == "euclidean":
            distances = pairwise_distances(inputs, metric=metric,
                                           squared=True)
        else:
            distances = pairwise_distances(X, metric=metric,
                                            n_jobs=-1)
        distances = distances.astype(np.float32, copy=False)
        conditional_P = _utils._binary_search_perplexity(
            distances, self.perp, verbose)
        P = conditional_P + conditional_P.T
        sum_P = np.maximum(np.sum(P), 1e-8)
        P = np.maximum(P/sum_P, 1e-8).astype(np.float32)
        return torch.tensor(P)
    
    
    def low_probabilities(self, y):
        n = y.shape[0]
        
        sum_y = torch.sum(y**2, 1)
        D = sum_y + (sum_y - 2*torch.matmul(y, y.T)).T
        D[range(n), range(n)] = 0.0
        
        prob = (self.q - 1.0)/(3.0 - self.q)
        prob = prob*D
        prob = 1.0 + prob
        prob = prob**(-1.0/(self.q - 1.0))
        prob[range(n), range(n)] = 0.0
        
        Q = prob/(torch.sum(prob) + 1e-10)
        Q = Q.clamp(min=1e-8)
        
        return Q