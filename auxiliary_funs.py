#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 07:31:30 2019

@author: John-Gioma
"""
import numpy as np
import scipy, glob, os, fnmatch, re, sklearn, time, sys, copy
import matplotlib.pyplot as plt

def pca_whitening(X,*args):
    """
    Function to compute PCA whitening matrix.
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: PCAMatrix: [M x M] matrix
    """

    if len(args) == 1:
        nica = args
    elif len(args) == 2:
        nica,EstCovNoise = args
    
    #Covariance Matrix
    Sigma   = np.cov(X);
    # Remove the contribution of noise
    Sigma -= EstCovNoise*np.eye(X.shape[0]) 

    # SVD of the Covariance
    U, S, V = np.linalg.svd( Sigma )
    
    # Whitening Matrix
    PCAWhiteningAll = np.diag(1/np.sqrt(S+1e-35))@U.T
    #Based on the principal components, choose the number of IC
#    nica    = sum(np.cov(PCAwall@X).diagonal());

    PCAWhitening    = 1.*PCAWhiteningAll[0:nica,:]
    #V = np.diag(1/np.sqrt(sv[0:nica]))@Vt[:,0:nica].T
    # Compute the whitened data
    X_white = PCAWhitening@X;
    
    # Check the covariance matrix of the whitened data
    CovX_white = np.cov(X_white)#(X_white@X_white.T)/(X_white.shape[1]-1);
    
    return U,S,V,PCAWhiteningAll,PCAWhitening,Sigma

def eigen_whitening(X,*args):
    """
    Function to compute Eigenvalue whitening matrix.
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: EVw: [M x M] matrix
    """
    if len(args) == 1:
        nica = args
    elif len(args) == 2:
        nica,EstCovNoise = args
      
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    Sigma  = np.cov(X) # [M x M]
    # Remove the contribution of noise
    Sigma -= EstCovNoise*np.eye(X.shape[0]) 
    
    # Eigenvalue Decomposition. X = E* np.diag(L) * E'
    L,E  = np.linalg.eigh( Sigma )
#    nica = sum(L/L.max()>=1e-6);
    
    EVWhiteningAll = np.dot(np.diag(1.0/np.sqrt(L + np.max([1e-35,1.001*abs(L.min())]))), E.T) # [M x M]
    EVWhitenin    = 1.*EVWhiteningAll[-nica:,:]
    
    return E,L,E.T,EVWhiteningAll,EVWhitenin,Sigma

"""
------------------------------------------------------------------------------
"""
def correlation(unknown,sources,nica):
    Cor = np.zeros((nica,sources.shape[0]))
    # Matching
    for ii in range(0,nica):
        for jj in range(0,sources.shape[0]):
            y = np.array([unknown[ii,:]/unknown[ii,:].max(),sources[jj,:]/sources[jj,:].max()])
            Cor[ii,jj] = abs(np.corrcoef(y)[0,1])
            if np.isnan(Cor[ii,jj]):
                Cor[ii,jj] = 0.
    return Cor

"""
------------------------------------------------------------------------------
"""
def mkdirs(dirpath):
    # Trz to create a directory, do nothing it such a directory already exists
    try:
        os.makedirs(dirpath)
    except:
        []    
    return
"""
------------------------------------------------------------------------------
"""

def npass_SGderivative(X,DerOrd,SGwin,SGpol):
    
    Xd = X*1.
    
    for kk in range(X.shape[0]):#range(0,X0.shape[0]):
        j = 1;
        while j<=DerOrd:
            Xd[kk,:]   = scipy.signal.savgol_filter(Xd[kk,:], SGwin, SGpol, deriv=1, mode='nearest');
            j +=1
       
    return Xd
"""
------------------------------------------------------------------------------
"""
def Hellinger(Xin,Yin,t):
    
    X = 1*Xin
    Y = 1*Yin
    # Non-negativity
    X[X<0.] = 0.
    Y[Y<0.] = 0.
    # Normalisation
    X/= np.trapz(X,t)
    Y/= np.trapz(Y,t)
    
    # Helliger Distance
    H = np.sqrt(1-np.trapz(np.sqrt(X*Y),t))
    
    return H
"""
------------------------------------------------------------------------------
"""
def duplicate(y):
    
    # Index of duplicate elements in array
    index = -1*np.ones((len(y),len(y)))
    for j in range(0,len(y)):
       index[j,:] = I-I[j] 
       index[j,j] = 1
    a = np.where(index==0)

    return a, index
