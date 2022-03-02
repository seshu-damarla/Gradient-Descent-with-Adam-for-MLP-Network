# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:11:47 2022

@author: Seshu Kumar Damarla
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 22:28:45 2022

@author: Seshu Kumar Damarla
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 11:55:20 2022

@author: Seshu Kumar Damarla
"""

"""
Multilayer percetron neural network trained by batch gradient descent algorithm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# data import
xtrain = pd.read_csv('trainxdata.csv',header=None)
ytrain=pd.read_csv('trainydata.csv',header=None)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtest = pd.read_csv('testxdata.csv', header=None)
ytest = pd.read_csv('testydata.csv', header=None)

xtest = np.array(xtest)
ytest = np.array(ytest)

#print(xtest)
#print(ytest)
#
#print(xtest.shape)
#print(ytest.shape)
#
#print(xtest[0,:])
#print(ytest[0,:])

# noralization
xmean = np.mean(xtrain,axis=0, keepdims=True, dtype=np.float)
xstd = np.std(xtrain, axis=0, keepdims=True, dtype=np.float)

ymean = np.mean(ytrain, axis=0, keepdims=True, dtype=np.float)
ystd = np.std(ytrain, axis=0, keepdims=True, dtype=np.float)

#print(xmean)
#print(xstd)
#print(ymean)
#print(ystd)

xtrain = (xtrain-xmean)/xstd
ytrain = (ytrain-ymean)/ystd

# initialization of weights and biases
def initialization(nh, n_inputs):
    np.random.seed(0)
    wh = np.random.rand(nh,n_inputs)*1
    bh = np.zeros((nh,1))
    wo = np.random.rand(1,nh)*1
    bo = np.zeros((1,1))
    
    Vwh = np.zeros((nh, n_inputs))
    Vbh = np.zeros((nh, 1))
    Vwo = np.zeros((1, nh))
    Vbo = np.zeros((1,1))
    
    Swh = np.zeros((nh, n_inputs))
    Sbh = np.zeros((nh, 1))
    Swo = np.zeros((1, nh))
    Sbo = np.zeros((1,1))
    
    return wh, bh, wo, bo, Vwh, Vwo, Vbh, Vbo, Swh, Sbh, Swo, Sbo

#print(wh)
#print(bh)
#print(wo)
#print(bo)

def sigmoid(x):
    z=1/(1+np.exp(-x))
    return z

def forward_propgation_lossfn(m, wh, bh, wo, bo, xtrain, ytrain):
    
    Z1 = np.dot(wh, xtrain.T) + bh
    A1 = sigmoid(Z1)
    Z2 = np.dot(wo, A1) + bo
    A2 = Z2
    yest = A2.T
    
    error = ytrain-yest
    lossfn = (np.sum(error**2))/(2*m)
    
    return lossfn

def backward_proagation_gradients(m, wh, bh, wo, bo, Vwh, Vwo, Vbh, Vbo, Swh, Sbh, Swo, Sbo, xtrain, ytrain, beta1, beta2):
    sum1=0
    sum2=0
    sum3=0
    sum4=0
    for i in range(0,m,1):
        
        xi=xtrain[i,:]
        xi = np.array([xi])
#        print(xi.shape)
        Z1 = np.dot(wh, xi.T) + bh # dfg
        A1 = sigmoid(Z1)
        Z2 = np.dot(wo, A1) + bo
        A2 = Z2
        yiest=A2
        yi=ytrain[i]
        ei=yi-yiest
        
#        print(xtrain.shape)
#        print(wo.shape)
#        print(A1.shape)
#        print(ei.shape)
        
        
        sum1=sum1+(ei*(np.dot(wo,A1))*(np.dot((1-A1),xi)))  # dwh
        sum2=sum2+(ei*(np.dot(wo,A1))*(1-A1))               # dbh
        sum3=sum3+ei*A1.T                                     # dwo
        sum4=sum4+ei                                        # dbo
        
    dwh=(-sum1)/m
    Vwh = beta1*Vwh + (1-beta1)*dwh
    Swh = beta2*Swh + (1-beta2)*(dwh**2)
    
    dbh=(-sum2)/m 
    Vbh = beta1*Vbh + (1-beta1)*dbh
    Sbh = beta2*Sbh + (1-beta2)*(dbh**2)
    
    dwo=(-sum3)/m
    Vwo= beta1*Vwo + (1-beta1)*dwo
    Swo = beta2*Swo + (1-beta2)*(dwo**2)
    
    dbo=(-sum4)/m
    Vbo = beta1*Vbo + (1-beta1)*dbo
    Sbo = beta2*Sbo + (1-beta2)*(dbo**2)
    
    return Vwh, Vbh, Vwo, Vbo, Swh, Sbh, Swo, Sbo

def mlp_model(xtrain, ytrain, nh, learning_rate, epochs, beta1, beta2, t, epsilon):
    n_inputs = xtrain.shape[1]  # no. of features of the input smaples
    m = xtrain.shape[0]         # no. of training examples
    wh, bh, wo, bo, Vwh, Vwo, Vbh, Vbo, Swh, Sbh, Swo, Sbo = initialization(nh, n_inputs)
    J=[]
    for i in range(epochs):
        lossfn = forward_propgation_lossfn(m, wh, bh, wo, bo, xtrain, ytrain)
        t=t+1
        Vwh, Vbh, Vwo, Vbo, Swh, Sbh, Swo, Sbo = backward_proagation_gradients(m, wh, bh, wo, bo, Vwh, Vwo, Vbh, Vbo, Swh, Sbh, Swo, Sbo,xtrain,ytrain,beta1,beta2)
        
        Vwh_corr = Vwh/(1-beta1**t)
        Swh_corr = Swh/(1-beta2**t)
        a1 = Vwh_corr/(np.sqrt(Swh_corr) + epsilon)
        wh = wh-learning_rate*a1
        
        Vbh_corr = Vbh/(1-beta1**t)
        Sbh_corr = Sbh/(1-beta2**t)
        a2 = Vbh_corr/(np.sqrt(Sbh_corr) + epsilon)
        bh = bh-learning_rate*a2
        
        Vwo_corr = Vwo/(1-beta1**t)
        Swo_corr = Swo/(1-beta2**t)
        a3 = Vwo_corr/(np.sqrt(Swo_corr) + epsilon)
        wo = wo-learning_rate*a3
        
        Vbo_corr = Vbo/(1-beta1**t)
        Sbo_corr = Sbo/(1-beta2**t)
        a4 = Vbo_corr/(np.sqrt(Sbo_corr) + epsilon)
        bo = bo-learning_rate*a4
        
        J.append(lossfn)
    
    return wh, bh, wo, bo, J

# hyperparameters of the model

nh = 40                     # no. of neurons in the hidden layer
learning_rate = 0.4
epochs = 2000
beta1 = 0.98
beta2 = 0.999
t=0
epsilon = 1e-8
#print(ytrain.shape)
wh, bh, wo, bo, J = mlp_model(xtrain, ytrain, nh, learning_rate, epochs, beta1, beta2, t, epsilon)

#plt.plot(J)
#print(J)
# training error
Z1 = np.dot(wh, xtrain.T) + bh
A1 = sigmoid(Z1)
Z2 = np.dot(wo, A1) + bo
A2 = Z2
ypred = A2.T
print(ypred.shape)
ypred = ypred*ystd+ymean
ytrain = ytrain*ystd+ymean
print(ypred.shape)
error = ytrain-ypred

#plt.plot(ypred)
#plt.plot(ytrain)

(R, pval) = stats.pearsonr(ytrain.flatten(),ypred.flatten())
print(R)

# model testing

xtest = (xtest-xmean)/xstd
Z1t = np.dot(wh, xtest.T) + bh
A1t = sigmoid(Z1t)
Z2t = np.dot(wo, A1t) + bo
A2t = Z2t
ypredt = A2t.T
ypredt = ypredt*ystd+ymean
errort = ytest-ypredt

plt.plot(ypredt)
plt.plot(ytest)

(Rt, pvalt) = stats.pearsonr(ytest.flatten(),ypredt.flatten())
print(Rt)







 
   
        
        
    
    
    
    
    

