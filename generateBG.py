import matplotlib.pyplot as plt
try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass


from MJOLNIR import _tools
from MJOLNIR.Data import DataSet 
from lmfit import Model, Parameter, report_fit
from MJOLNIR.Data import Mask
from MJOLNIR._tools import fileListGenerator

import numpy as np
from os import path
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import pandas as pd
import copy

def generateBG(foregroundDataSet,backgroundDataSet, foregroundMask = None,dQ=0.02,dE=0.1,plotPowderAverage=False,verbose=True):
    """Generate a background dataset using the backgroundDataSet to perform powder averaged background interpolation
    in the shape of the foregroundDataSet.
    
    Args:
    
        - foregroundDataSet (DataSet): Foreground data set to be replicated by the bg
        
        - backgroundDataSet (DataSet): Masked data set to be used for background masking out intensities. 
        
    Kwargs: 
    
        - foregroundMask (Mask): Mask to apply to powder-averaged background dataset. If none use mask from foregroundDataSet (default None)
    
        - dQ (float): Bin size along Qlength (default 0.02 1/AA)
        
        - dE (float): Bin size in energy (default 0.1 meV)
        
        - plotPowderAverage (bool): Flag to control plotting of powder average, if true ax is also returned (default False)
        
        - verbose (bool): If true, print the averaging process to the command line (default True)
        
    
    """
    # Reset the mask of ds to be nothing
    bgmask = copy.deepcopy(backgroundDataSet.mask)
    backgroundDataSet.mask = [np.zeros_like(df.I,dtype=bool) for df in backgroundDataSet]
    
    # Find the maximal positions before the mask is applied
    qLength = np.array([np.max(np.linalg.norm([df.qx,df.qy],axis=0)) for df in backgroundDataSet])
    E = [[np.min(e),np.max(e)] for e in foregroundDataSet.energy.data]
    QMax = np.max(qLength) # QMin is always set to 0
    EMin = np.nanmin(E)
    EMax = np.nanmax(E)
    if verbose:
        print('Bin limites are\n','Q:',0.0,QMax,'\nE:',EMin,EMax)
        print('Copying bg dataset')
    
    backgroundDataSet.mask = bgmask
    
    # Copy the foreground data to bg data set
    newBackgroundDataSet = copy.deepcopy(foregroundDataSet)

    
    if verbose:
        print('Performing powder average')
    # Prepare for powder average of background data
    I = backgroundDataSet.I.extractData()
    Monitor = backgroundDataSet.Monitor.extractData()
    Norm = backgroundDataSet.Norm.extractData()
    
    # Position in the powder average (QLength, Energy)
    positions2D = np.array([np.linalg.norm([backgroundDataSet.qx.extractData(),
                                            backgroundDataSet.qy.extractData()],axis=0),
                            backgroundDataSet.energy.extractData()])

    # Generate the bins with suitable extensions
    QBins = np.arange(0,QMax+dQ*1.1,dQ)
    EnergyBins = np.arange(EMin-dE*1.1,EMax+dE*1.1,dE)

    
    # Perform 2D histogram
    normcounts,*powderBins = np.histogram2d(*positions2D,bins=np.array([QBins,EnergyBins],dtype=object),weights=np.ones((positions2D.shape[1])).flatten())
    intensity = np.histogram2d(*positions2D,bins=np.array([QBins,EnergyBins],dtype=object),weights=I.flatten())[0]
    MonitorCount=  np.histogram2d(*positions2D,bins=np.array([QBins,EnergyBins],dtype=object),weights=Monitor.flatten())[0]
    Normalization= np.histogram2d(*positions2D,bins=np.array([QBins,EnergyBins],dtype=object),weights=Norm.flatten())[0]

    # Calcualte the intensities
    Int = np.divide(intensity*normcounts,MonitorCount*Normalization)
    #eMean = 0.5*(EnergyBins[:-1]+EnergyBins[1:])
    #qMean = 0.5*(QBins[:-1]+QBins[1:])

    
    if plotPowderAverage: # Plot the powder average
        powderFig,powderAx = plt.subplots()
        X,Y = np.meshgrid(QBins,EnergyBins)
        powderAx.pcolormesh(X,Y,Int.T)
    
    if verbose:
        print('Finding intensities for background dataset')
    # Find intensites for individual points in the scan across data files
    for count,df in enumerate(newBackgroundDataSet):
        dfPosition = np.array([np.linalg.norm([df.qx,df.qy],axis=0),df.energy])
        # Calculate the bin index
        qIndex = np.floor((dfPosition[0]-QBins[0])/dQ).astype(int)#np.asarray([np.argmin(np.abs(qpos-qMean)) for qpos in dfPosition[0].flatten()]).reshape(*dfPosition[0].shape)
        eIndex = np.floor((dfPosition[1]-EnergyBins[0])/dE).astype(int)#np.asarray([np.argmin(np.abs(epos-eMean)) for epos in dfPosition[1].flatten()]).reshape(*dfPosition[1].shape)

        # Clamp the indicies (Might not be needed....)
        qIndex[qIndex<0]=0
        qIndex[qIndex>Int.shape[0]-1]=Int.shape[0]-1
        eIndex[eIndex<0]=0
        eIndex[eIndex>Int.shape[1]-1]=Int.shape[1]-1
        
        # Find intensity by rescaling with monitor and normalization
        df.I = Int[[qIndex],[eIndex]][0]*df.Monitor*df.Norm
        if verbose:
            print('df',count+1,'of',len(backgroundDataSet))
    # update the background data set
    newBackgroundDataSet._getData()
    # Apply the foreground mask to the background dataset
    
    if foregroundMask is None:
        newBackgroundDataSet.mask = foregroundDataSet.mask
    else:
        newBackgroundDataSet.mask = foregroundMask(newBackgroundDataSet)
    
    if plotPowderAverage:
        return newBackgroundDataSet,powderAx
    return newBackgroundDataSet
    