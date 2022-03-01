#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:47:54 2018

@author: John-Gioma
"""
import scipy, glob, os, fnmatch, re 
import numpy as np
#import ipdb
from numpy import matlib

def backcor(nn,y,ord,threshold,function): 

# BACKCOR   Background estimation by minimizing a non-quadratic cost function.
#
#   [EST,COEFS,IT] = BACKCOR(N,Y,ORDER,THRESHOLD,FUNCTION) computes and estimation EST
#   of the background (aka. baseline) in a spectroscopic signal Y with wavelength N.
#   The background is estimated by a polynomial with order ORDER using a cost-function
#   FUNCTION with parameter THRESHOLD. FUNCTION can have the four following values:
#       'sh'  - symmetric Huber function :  f(x) = { x^2  if abs(x) < THRESHOLD,
#                                                  { 2*THRESHOLD*abs(x)-THRESHOLD^2  otherwise.
#       'ah'  - asymmetric Huber function :  f(x) = { x^2  if x < THRESHOLD,
#                                                   { 2*THRESHOLD*x-THRESHOLD^2  otherwise.
#       'stq' - symmetric truncated quadratic :  f(x) = { x^2  if abs(x) < THRESHOLD,
#                                                       { THRESHOLD^2  otherwise.
#       'atq' - asymmetric truncated quadratic :  f(x) = { x^2  if x < THRESHOLD,
#                                                        { THRESHOLD^2  otherwise.
#   COEFS returns the ORDER+1 vector of the estimated polynomial coefficients
#   (computed with n sorted and bounded in [-1,1] and y bounded in [0,1]).
#   IT returns the number of iterations.
    
#   [EST,COEFS,IT] = BACKCOR(N,Y) does the same, but run a graphical user interface
#   to help setting ORDER, THRESHOLD and FCT.
#
# For more informations, see:
# - V. Mazet, C. Carteret, D. Brie, J. Idier, B. Humbert. Chemom. Intell. Lab. Syst. 76 (2), 2005.
# - V. Mazet, D. Brie, J. Idier. Proceedings of EUSIPCO, pp. 305-308, 2004.
# - V. Mazet. PhD Thesis, University Henri Poincaré Nancy 1, 2005.
# 
# 22-June-2004, Revised 19-June-2006, Revised 30-April-2010,
# Revised 12-November-2012 (thanks E.H.M. Ferreira!)
# Comments and questions to: vincent.mazet@unistra.fr.    
      
# Check arguments
#if nargin < 2, error('backcor:NotEnoughInputArguments','Not enough input arguments'); end;:
#    
#if nargin < 5, [z,a,it,ord,s,fct] = backcorgui(n,y); return; end; # delete this line if you do not need GUI
#if ~isequal(fct,'sh') && ~isequal(fct,'ah') && ~isequal(fct,'stq') && ~isequal(fct,'atq'),
#    error('backcor:UnknownFunction','Unknown function.');
#end;

    # Rescaling
    N = nn.size-1;
    i = nn.argsort(0);
    nn = np.sort(nn);
    y = y[i.tolist()];
    maxy = y.max();
    dely = (maxy-y.min())/2;
    nn = 2 * (nn-nn[N]) / (nn[N]-nn[0]) + 1;
    y = (y-maxy)/dely + 1;
    
    # Vandermonde matrix
    p = np.arange(0,ord+1,1);
    T = np.power(np.transpose(np.matlib.repmat(nn,ord+1,1)), 
                 np.matlib.repmat(p,N+1,1));
    
    Tinv = np.linalg.pinv(T.T@T) @ T.T;
#    Tinv = np.linalg.pinv(T);
    
    # Initialisation (least-squares estimation)
    a = Tinv@y;
    z = T@a;
    
    # Other variables
    alpha = 0.99 * 1/2;     # Scale parameter alpha
    it = 0;                 # Iteration number
    zp = np.ones((N+1,1));         # Previous estimation
  
    # LEGEND
    while np.linalg.norm(z-zp)/np.linalg.norm(zp) > 1e-6:
        
        it = it + 1;        # Iteration number
        zp = z;             # Previous estimation
        res = y - z;        # Residual
        
        # Estimate d
        if function == 'sh':
            d = (res*(2*alpha-1)) * (abs(res)<threshold) + (-alpha*2*threshold-res) * (res<=-threshold) + (alpha*2*threshold-res) * (res>=threshold);
        elif function == 'ah':
            d = (res*(2*alpha-1)) * (res<threshold) + (alpha*2*threshold-res) * (res>=threshold);
        elif function == 'stq':
            d = (res*(2*alpha-1)) * (abs(res)<threshold) - res * (abs(res)>=threshold);
        elif function == 'atq' :
            d = (res*(2*alpha-1)) * (res<threshold) - res * (res>=threshold);
        
        # Estimate z
        a = Tinv @ (y+d);   # Polynomial coefficients a
        z = T@a;            # Polynomial
    
    # Rescaling
#    j = i.argsort(0);
    z = (z[i.tolist()]-1)*dely + maxy;

    a[1]= a[1]-1;
    a = a*dely;# + maxy;

    return [z,a,it]

