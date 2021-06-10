# -*- coding: utf-8 -*-
"""
Created on Tue Apr 6

@author: gmaggioni3
@author: skocevska3
"""

"""
Import modules
"""
import numpy as np
import pandas as pd
import scipy, glob, os, fnmatch, re, sklearn, time, sys
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True,precision=4)
Pythondirectory = os.getcwd()[:-18];
sys.path.append(Pythondirectory)

import math
import auxiliary_funs
# Baseline removal
from back_corr_mazet import backcor
# Minimisation, fitting
from scipy import optimize
from scipy import signal
import multiprocessing
from itertools import combinations, permutations
from sklearn import feature_selection
import copy
from sklearn.decomposition import FastICA, PCA # Import FastICA and PCA from the class sklearn
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
from scipy.interpolate import interp1d
from sklearn.linear_model.ridge import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from scipy.signal import savgol_filter
from brokenaxes import brokenaxes
import scipy.io as io
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

"""
Set the relevant Directories
"""

MainDirectory = Pythondirectory + '/Raman/' + '/Experimental/' ;

TXTSpectradirectory = MainDirectory;

savedirectory=MainDirectory
                   
auxiliary_funs.mkdirs(savedirectory)      

os.chdir(MainDirectory)

"""
Load the library spectra
"""
data=np.load('library.npz')    
for j in range(np.size(data.files)):
    x = data.files[j]    
    
for j in range(np.size(data.files)):
    #Create a variable with the name associated to that saved in the file
    vars()[data.files[j]] = data[data.files[j]] 
    
Lnew_sources=Xlibrary[:,:]; #since the Xlibrary file contains all references

"""
Plot library L: *paper figure*
"""
plt.figure(1)
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.plot(wavelength,Lnew_sources.T[:,4],'orange',label='NO${_3}$',linewidth=3)
plt.plot(wavelength,Lnew_sources.T[:,3],'darkviolet',label='CO${_3}$',linewidth=3)
plt.plot(wavelength,Lnew_sources.T[:,2],'teal',label='NO${_2}$',linewidth=3)
plt.plot(wavelength,Lnew_sources.T[:,1],'firebrick',label='SO${_4}$',linewidth=3)
plt.plot(wavelength,Lnew_sources.T[:,5],'dimgray',label='H${_2}$O',linewidth=3)
plt.plot(wavelength,Lnew_sources.T[:,6],'hotpink',label='CH${_3}$COO',linewidth=3)
plt.plot(wavelength,Lnew_sources.T[:,0],'navy',label='PO${_4}$',linewidth=3)
plt.plot(wavelength,Lnew_sources.T[:,7],'olive',label='C${_2}$O${_4}$',linewidth=3)
plt.legend(loc="upper right", fontsize=14,frameon=False)
plt.show()
# plt.savefig('raman_library.png')

"""
Indexing
"""
CIndex = np.array([4,2,1,3]) 
CIndex_h2o = np.array([4,2,1,3,5]) #to match excel table: NO3, NO2, SO4, CO3, H2O
Lnew_h2o=Xlibrary[CIndex_h2o,:];

"""
Load the Test Spectra
"""      
data=np.load('test.npz')    
for j in range(np.size(data.files)):
    x = data.files[j]    
    
for j in range(np.size(data.files)):
    #Create a variable with the name associated to that saved in the file
    vars()[data.files[j]] = data[data.files[j]] 
    
Xsample=Xtest[:,:];

"""
Load the Pure test data spectra
"""
data=np.load('test_pure.npz')    
for j in range(np.size(data.files)):
    x = data.files[j]    
    
for j in range(np.size(data.files)):
    #Create a variable with the name associated to that saved in the file
    vars()[data.files[j]] = data[data.files[j]]
    
Xtest_pure=Xtest_pure[:,:];

"""
Load the Training spectra - BASELINED
"""
data=np.load('train.npz')    
for j in range(np.size(data.files)):
    x = data.files[j]    
    
for j in range(np.size(data.files)):
    #Create a variable with the name associated to that saved in the file
    vars()[data.files[j]] = data[data.files[j]]
    
Xtraining=Xtrain[:,:];

"""
Load Test data Concentrations
"""
excel = pd.read_excel (r'C:\Users/stefa\Dropbox (GaTech)\PhD Research\02 DATA\011 Data Collection - Repeat for BSS-PLSR paper\test_data.xlsx')
molality_h2o=excel[["molality NO3","molality NO2","molality SO4","molality CO3","molality H2O"]]

m_h2o=molality_h2o.iloc[[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22],:]
C_h2o=m_h2o.values #to get into array
C_measured_h2o=C_h2o; #for all

"""
Load Training data Concentrations
"""
excel_training = pd.read_excel (r'C:\Users/stefa\Dropbox (GaTech)\PhD Research\02 DATA\011 Data Collection - Repeat for BSS-PLSR paper\training_data.xlsx')
molality_training_h2o=excel_training[["molality NO3","molality NO2","molality SO4","molality CO3","molality H2O"]]
m_training_h2o=molality_training_h2o.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
                          26,27,28,29,30,31,32,33,34,35],:]
Ctraining_h2o=m_training_h2o.values

# Make copies of your spectra
X0 = Xsample*1 
Xe = 1*X0;
Xo = 1*X0;

# Estimate of Noise Level
EstCovNoise = 183.3505155529105; #variance for 2 spectra
SNR = (((Xe**2).mean(1)).mean()/EstCovNoise)
SNRdb=10*(math.log10(SNR))
print(SNRdb)

# Create library data

s=Lnew_sources[3] #co3
s0=Lnew_sources[1] #so4
s1=Lnew_sources[2] #no2
s3=Lnew_sources[4] #no3
s4=Lnew_sources[0,:] #po4
s5=100*Lnew_sources[5,:] #h2o
s6=Lnew_sources[6,:] #acetate
s7=Lnew_sources[7,:] #oxalate

s=s.reshape(1,-1)
s0=s0.reshape(1,-1) 
s1=s1.reshape(1,-1)
s3=s3.reshape(1,-1)
s4=s4.reshape(1,-1)
s5=s5.reshape(1,-1)
s6=s6.reshape(1,-1)
s7=s7.reshape(1,-1)

s_reshape=np.vstack((s0,s,s1,s3,s4,s5,s6,s7))

# Define what is fed into BSS
Xin=np.vstack((Xsample, Xtest_pure, Xtraining, s_reshape))
# if you remove the minor species references:
# Xin=np.vstack((Xsample, Xtest_pure, Xtraining, s,s0,s1,s3,s5))

plt.figure(2)
plt.plot(wavelength,((Xin).T))

"""
Determination of the number of Components via SVD - NEW
"""
Xnica = Xin
U, Sv, V = np.linalg.svd( Xnica, full_matrices=True)
S = np.zeros(Xnica.shape)
np.fill_diagonal(S,Sv, wrap=True)
N = np.linalg.norm(Xnica,ord=1)
E = np.zeros(Xnica.shape[0])
for nn in range(0,Xnica.shape[0]):
     Rec = U@S[:,0:nn+1]@V[0:nn+1,:]
     E[nn] = np.linalg.norm(Xnica-Rec,ord=1)/N
DE = np.append(-E[0],np.diff(E)) 
nica = np.max([(sum(E>=1e-2)+1),sum(DE<-1e-2)])
print(nica)
plt.figure(3)
plt.plot(Sv)
#nica = len(CIndex)+1

"""
BSS Removal
"""
tic = time.time()

Xd = auxiliary_funs.npass_SGderivative(Xin,1,7,2) #used in pls later
#Xd=Xin

# Compute ICA
ica = FastICA(n_components=nica,tol=1e-8,max_iter=500)
ica.fit_transform(Xd.T)  # Reconstruct signals, needs the transpose of the matrix
Aica = ica.mixing_  # Get estimated miXeg matrix

S0ica = (np.linalg.pinv(Aica)@Xin)  # Reconstruct signals, needs the transpose of the matrix

# Compute MCR
mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-4,l1_ratio=0.75),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=Xin.mean()*1e-8,c_constraints=[ConstraintNonneg()])
mcrals.fit(Xin, ST= S0ica**2 )

S0mcr = mcrals.ST_opt_;
Amcr  = mcrals.C_opt_; 
toc = time.time()

runningtime_BSS = toc-tic # How long the decomposition took


# Species Identification 
Cor = auxiliary_funs.correlation(S0mcr,Lnew_h2o,nica).T
# with cut-off value
# Ivalues = Cor.max(0);#maximum correlation from each column
# I = np.where(Ivalues<0.70)[0]; #background values

# without defining cut-off value: 
Cor=pd.DataFrame(Cor)
maxvaluesind=pd.DataFrame([])
minvaluesind=pd.DataFrame([])
minvalues=pd.DataFrame([])
for i in np.arange(0,len(pd.DataFrame(Cor).index)):
    maxval=Cor.iloc[i,:].max()
    maxind=Cor.iloc[i,:].loc[Cor.iloc[i,:]==maxval]
    maxvaluesind=pd.concat([maxvaluesind,maxind])
    
    minind=Cor.iloc[i,:].loc[Cor.iloc[i,:]!=maxval]
    minvaluesind=pd.concat([minvaluesind,minind],axis=1)
minvaluesind=minvaluesind.T
rowmin=minvaluesind.dropna(1)
mincolumns=rowmin.columns.values
I=np.array(mincolumns).astype(int);

#save
np.savetxt("Correlation.csv", Cor, delimiter=",")

#Compute the deflation matrix
Xdeflation = Amcr[0:21,I]@S0mcr[I,:]
newXdef=Xdeflation

# if you add the pure test data:
Xsample=np.vstack((Xsample,Xtest_pure))

#Compute the deflated matrix
newXhat=Xsample-newXdef;

"""
Plot overlays:
"""
plt.figure(4)
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.ylabel('Counts',fontsize=14)
plt.plot(wavelength,(Xsample[:,:]).T,'b',linewidth=1) #this is the mixture sample in blue
plt.plot(wavelength,(newXdef[:,:]).T,'y',linewidth=1) # this is the deflation (removed background) matrix in yellow
plt.plot(wavelength,(newXhat).T,'r',linewidth=1) # this is the target matrix in red
legend_elements= [Line2D([0], [0], lw=3, color='b', label='original'),
                  Line2D([0], [0], lw=3, color='y', label='subtracted'),
                  Line2D([0], [0], lw=3, color='r', label='preprocessed')]
plt.legend(handles=legend_elements, frameon=False, loc='upper right',fontsize=13)

#%% plot matching spectra

#normalize spectra
S0mcr_normalized = np.zeros((S0mcr.shape[0],wavelength.shape[0]))
X_normalized = np.zeros((S0mcr.shape[0],wavelength.shape[0]))
for ii in range(0,(S0mcr.shape[0])):
    X_normalized[ii]=np.array([S0mcr[ii,:]/S0mcr[ii,:].max()])
S0mcr_normalized=np.array(X_normalized)

Lnew_sources_normalized = np.zeros((Lnew_sources.shape[0],wavelength.shape[0]))
Y_normalized = np.zeros((Lnew_sources.shape[0],wavelength.shape[0]))
for jj in range(0,Lnew_sources.shape[0]):
    Y_normalized[jj]=np.array([Lnew_sources[jj,:]/Lnew_sources[jj,:].max()])
Lnew_sources_normalized=np.array(Y_normalized)
    
fig=plt.figure(8899,figsize=(9,10))
plt.figure(8899)
fig.text(0.5, 0.04, 'Wavenumber [$cm^{-1}$]', fontsize=14, ha='center', va='center')
fig.text(0.06, 0.5, 'Counts', ha='center', fontsize=14, va='center', rotation='vertical')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.subplot(9,1,1) #no3
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,4],'orange',label="nitrate")
plt.plot(wavelength,((S0mcr_normalized).T)[:,7],'k--',label="nitrate")
legend_elements1= [Line2D([0], [0], color='orange', lw=2, label='NO${_3}$ ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower right',fontsize=12)


plt.subplot(9,1,2) #co3
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,3],'darkviolet',label="carbonate")
plt.plot(wavelength,((S0mcr_normalized).T)[:,8],'k--',label="carbonate")
legend_elements1= [Line2D([0], [0], color='darkviolet', lw=2, label='CO${_3}$ ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower right',fontsize=12)


plt.subplot(9,1,3) #no2
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,2],'teal',label="nitrite")
plt.plot(wavelength,((S0mcr_normalized).T)[:,1],'k--',label="nitrite")
legend_elements1= [Line2D([0], [0], color='teal', lw=2, label='NO${_2}$ ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower right',fontsize=12)


plt.subplot(9,1,4) #so4
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,1],'firebrick',label="sulfate")
plt.plot(wavelength,((S0mcr_normalized).T)[:,6],'k--',label="sulfate")
legend_elements1= [Line2D([0], [0], color='firebrick', lw=2, label='SO${_4}$ ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower right',fontsize=12)


plt.subplot(9,1,5) #water
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,5],'dimgray',label="water")
plt.plot(wavelength,((S0mcr_normalized).T)[:,0],'k--',label="water")
legend_elements1= [Line2D([0], [0], color='dimgray', lw=2, label='H${_2}$O ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='H${_2}$O source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower center',fontsize=12)

plt.subplot(9,1,6) #nitrate-carbonate extra source
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,4],'orange',label="nitrate")
plt.plot(wavelength,Lnew_sources_normalized.T[:,3],'darkviolet',label="carbonate")
plt.plot(wavelength,((S0mcr_normalized).T)[:,2],'k--',label="extra source")
legend_elements1= [Line2D([0], [0], color='orange', lw=2, label='NO${_3}$ ref'),
                    Line2D([0], [0], color='darkviolet', lw=2, label='CO${_3}$ ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='center right',fontsize=10)

plt.subplot(9,1,7) #acetate
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,6],'hotpink',label="acetate")
plt.plot(wavelength,((S0mcr_normalized).T)[:,4],'k--',label="acetate")
legend_elements1= [Line2D([0], [0], color='hotpink', lw=2, label='CH${_3}$COO ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower right',fontsize=12)


plt.subplot(9,1,8) #phosphate
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.xaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,0],'navy',label="phosphate")
plt.plot(wavelength,((S0mcr_normalized).T)[:,3],'k--',label="phosphate")
legend_elements1= [Line2D([0], [0], color='navy', lw=2, label='PO${_4}$ ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower right',fontsize=12)

plt.subplot(9,1,9) #oxalate
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.plot(wavelength,Lnew_sources_normalized.T[:,7],'olive',label="oxalate")
plt.plot(wavelength,((S0mcr_normalized).T)[:,5],'k--',label="oxalate")
legend_elements1= [Line2D([0], [0], color='olive', lw=2, label='C${_2}$O${_4}$ ref'),
                    Line2D([0], [0], color='k', linestyle='--',lw=2, label='source')]
plt.legend(handles=legend_elements1, frameon=False,loc='lower right',fontsize=12)
plt.show()



#%% PLS regression
#define PLS components
nica_pls=5;
# Get derivative spectra
Xtraining2 = savgol_filter(Xtraining, 7,polyorder = 2,deriv=1)
Xsample2 = savgol_filter(Xsample, 7, polyorder = 2,deriv=1)
newXhat2 = savgol_filter(newXhat, 7, polyorder = 2,deriv=1)
                           

pls = PLSRegression(n_components=nica_pls)
pls.fit(Xtraining2, Ctraining_h2o)
y_pls_original = pls.predict(Xsample2) #using orriginal spectra
y_pls_corrected = pls.predict(newXhat2) #using corrected spectra

# Get X scores
T = pls.x_scores_
# Get X loadings
P = pls.x_loadings_
plt.figure(12) 
plt.plot(wavelength,(P[:,:]),linewidth=0.7)

#Mean squared error
MSE_pls_h2o = np.square(np.subtract(C_measured_h2o,y_pls_original)).mean() 
MSE_pls_h2o_def=np.square(np.subtract(C_measured_h2o,y_pls_corrected)).mean() 
print(MSE_pls_h2o)
print(MSE_pls_h2o_def)

#Percent error
y_true=C_measured_h2o;
y_pred=y_pls_original;
y_pred_bss=y_pls_corrected;
def mean_absolute_percentage_error(y_true, y_pred): 
    return (np.abs((y_true - y_pred) / y_true)) * 100
PE=mean_absolute_percentage_error(y_true, y_pred)
PE= np.asarray(PE)
np.savetxt("PE_exp_Raman_70corr.csv", PE, delimiter=",")

def mean_absolute_percentage_error_bss(y_true, y_pred_bss): 
    return (np.abs((y_true - y_pred_bss) / y_true)) * 100
PE_bss=mean_absolute_percentage_error_bss(y_true, y_pred_bss)
PE_bss= np.asarray(PE_bss)
np.savetxt("PE_BSS_exp_Raman_70corr.csv", PE_bss, delimiter=",")
"""
Plot spectra: *paper figure*
"""
#plot training data
plt.figure(13)
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.ylabel('Counts',fontsize=14)
plt.plot(wavelength,(Xtraining[:,:]).T,linewidth=1)

#plot test data - original
plt.figure(14)
plt.plot(wavelength,(Xsample[:,:]).T,linewidth=1) 
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.ylabel('Counts',fontsize=14)

#plot derivative of training data
plt.figure(15)
plt.plot(wavelength,(Xtraining2[:,:]).T,linewidth=1) 
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.ylabel('Counts',fontsize=14)

#plot derivative of test data - original
plt.figure(16)
plt.plot(wavelength,(Xsample2[:,:]).T,linewidth=1) 
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.ylabel('Counts',fontsize=14)

#plot derivative of test data - preprocessed
plt.figure(17)
plt.plot(wavelength,(newXhat2[:,:]).T,linewidth=1) 
plt.xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.ylabel('Counts',fontsize=14)

#%% baxes
fig = plt.figure(18)
#bax = brokenaxes(xlims=((0,1.5),(55.4,55.7)), ylims=((-4,2),(55.4,55.7)), hspace=.15)
bax = brokenaxes(xlims=((0,1.5),(55.4,55.7)), ylims=((-3,3),(55.4,55.7)), hspace=.15)
fig.text(0.5, 0.03, 'Measured concentration (mol solute/kg solvent)', fontsize=13, ha='center', va='center')
fig.text(0.03, 0.5, 'Predicted concentration (mol solute/kg solvent)', ha='center', fontsize=13, va='center', rotation='vertical')


x1=np.linspace(0,100,101) 
bax.plot(x1,x1,'k-') # identity line
bax.plot(C_measured_h2o[:,0],y_pls_original[:,0],'bo',label="original spectra NO3")
bax.plot(C_measured_h2o[:,1],y_pls_original[:,1],'b*',label="original spectra NO2")
bax.plot(C_measured_h2o[:,2],y_pls_original[:,2],'bd',label="original spectra SO4")
bax.plot(C_measured_h2o[:,3],y_pls_original[:,3],'bs',label="original spectra CO3")
bax.plot(C_measured_h2o[:,4],y_pls_original[:,4],'bP',label="original spectra H2O")

bax.plot(C_measured_h2o[:,0],y_pls_corrected[:,0],'ro',label="corrected spectra NO3") # for deflated matrix
bax.plot(C_measured_h2o[:,1],y_pls_corrected[:,1],'r*',label="corrected spectra NO2")
bax.plot(C_measured_h2o[:,2],y_pls_corrected[:,2],'rd',label="corrected spectra SO4")
bax.plot(C_measured_h2o[:,3],y_pls_corrected[:,3],'rs',label="corrected spectra CO3")
bax.plot(C_measured_h2o[:,4],y_pls_corrected[:,4],'rP',label="corrected spectra H2O")

legend_elements1= [Line2D([0], [0], color='r', lw=3, label='BSS-PLSR'),
                   Line2D([0], [0], color='b', lw=3, label='PLSR'),
                   Line2D([0], [0], marker='o', color='w', label='NO${_3}$',markerfacecolor='k',markersize=7), 
                   Line2D([0], [0], marker='*', color='w', label='NO${_2}$',markerfacecolor='k',markersize=12),
                   Line2D([0], [0], marker='d', color='w', label='SO${_4}$',markerfacecolor='k',markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='CO${_3}$',markerfacecolor='k',markersize=7),
                   Line2D([0], [0], marker='P', color='w', label='H${_2}$O',markerfacecolor='k',markersize=10)]

bax.legend(handles=legend_elements1, frameon=False,loc='lower right', bbox_to_anchor=(1.1,0.0001),fontsize=13)
# #%%
# """
# Save the results
# """            
# SimNum='Final'
# file_name = 'BSS-PLSR-Raman-Experimental' + str(SimNum)

# np.savez(SimSaveDir + file_name,
#                  L=Lnew_sources, wavelength=wavelength,Xsample=Xsample,newXhat=newXhat)
        
# scipy.io.savemat(SimSaveDir + file_name,
#                  mdict={
#                  'L': Lnew_sources, 'wavelength':wavelength, 'Xsample':Xsample, 'newXhat':newXhat})
