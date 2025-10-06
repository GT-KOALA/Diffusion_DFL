#/usr/bin/env python3

import numpy as np
import scipy.stats as st
import operator
from functools import reduce

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.optim as optim

from qpth.qp import QPFunction
from constants import *

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_2(x))
        x = self.fc3(x)
        return x
    
class GaussianMLP(nn.Module):
    def __init__(self, x_dim, hidden_dims, n):
        super().__init__()
        layers = []
        prev = x_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.feature = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, n)
        self.logvar_head = nn.Linear(prev, n)

    def forward(self, x):
        h = self.feature(x)
        mu = self.mu_head(h)                   # (B, n)
        logvar = self.logvar_head(h)           # (B, n)
        return mu, logvar

class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        # Initialize linear layer with least squares solution
        X_ = np.hstack([X, np.ones((X.shape[0],1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T)
        b.data = torch.Tensor(Theta[-1,:])
        
        # Set up non-linear network of 
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))
        
    def forward(self, x):
        return self.lin(x) + self.net(x), \
            self.sig.expand(x.size(0), self.sig.size(1))
    
    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred-Y)**2, 0)
        self.sig.data = torch.sqrt(var).data.unsqueeze(0)


def GLinearApprox(gamma_under, gamma_over):
    """ Linear (gradient) approximation of G function at z"""
    class GLinearApproxFn(Function):
        @staticmethod    
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.cdf(
                z.cpu().numpy()) - gamma_under)
            if USE_GPU:
                res = res.to(DEVICE)
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            grad_output = grad_output.to(z.device)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=z.device)
            
            dz = (gamma_under + gamma_over) * pz
            dmu = -dz
            dsig = -(gamma_under + gamma_over)*(z-mu) / sig * pz
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GLinearApproxFn.apply


def GQuadraticApprox(gamma_under, gamma_over):
    """ Quadratic (gradient) approximation of G function at z"""
    class GQuadraticApproxFn(Function):
        @staticmethod
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.pdf(
                z.cpu().numpy()))
            if USE_GPU:
                res = res.to(DEVICE)
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            grad_output = grad_output.to(z.device)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=z.device)
            
            dz = -(gamma_under + gamma_over) * (z-mu) / (sig**2) * pz
            dmu = -dz
            dsig = (gamma_under + gamma_over) * ((z-mu)**2 - sig**2) / \
                (sig**3) * pz
            
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GQuadraticApproxFn.apply


class SolveSchedulingQP(nn.Module):
    """ Solve a single SQP iteration of the scheduling problem"""
    def __init__(self, params):
        super(SolveSchedulingQP, self).__init__()
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = torch.tensor(np.vstack([D,-D]), dtype=torch.double, device=DEVICE)
        self.h = (self.c_ramp * torch.ones((self.n - 1) * 2, device=DEVICE)).double()
        self.e = torch.DoubleTensor()
        if USE_GPU:
            self.e = self.e.to(DEVICE)
        
    def forward(self, z0, mu, dg, d2g):
        nBatch, n = z0.size()
        
        Q = torch.cat([torch.diag(d2g[i] + 1).unsqueeze(0) 
            for i in range(nBatch)], 0).double()
        p = (dg - d2g*z0 - mu).double()
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        G = G.to(z0.device)
        h = h.to(z0.device)
        self.e = self.e.to(z0.device)
        
        out = QPFunction(verbose=False)(Q, p, G, h, self.e, self.e)
        return out


class SolveScheduling(nn.Module):
    """ Solve the entire scheduling problem, using sequential quadratic 
        programming. """
    def __init__(self, params):
        super(SolveScheduling, self).__init__()
        self.params = params
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = torch.tensor(np.vstack([D,-D]), dtype=torch.double, device=DEVICE)
        self.h = (self.c_ramp * torch.ones((self.n - 1) * 2, device=DEVICE)).double()
        self.e = torch.DoubleTensor()
        if USE_GPU:
            self.e = self.e.to(DEVICE)
        
    def forward(self, mu, sig):
        nBatch, n = mu.size()
        
        # Find the solution via sequential quadratic programming, 
        # not preserving gradients
        z0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        mu0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        sig0 = sig.detach() # Variable(1. * sig.data, requires_grad=False)
        for i in range(20):
            dg = GLinearApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            d2g = GQuadraticApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            dg = dg.to(z0.device)
            d2g = d2g.to(z0.device)
            z0_new = SolveSchedulingQP(self.params)(z0, mu0, dg, d2g)
            solution_diff = (z0-z0_new).norm().item()
            # print("+ SQP Iter: {}, Solution diff = {}".format(i, solution_diff))
            z0 = z0_new
            if solution_diff < 1e-10:
                break
                  
        # Now that we found the solution, compute the gradient-propagating 
        # version at the solution
        dg = GLinearApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        d2g = GQuadraticApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        dg = dg.to(z0.device)
        d2g = d2g.to(z0.device)
        return SolveSchedulingQP(self.params)(z0, mu, dg, d2g)
