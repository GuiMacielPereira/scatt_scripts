
from os import error
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

from scipy import signal
from scipy import optimize

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=True)
 
resolution = Load(r"fixatures/yspace_fit/resolution.nxs", OutputWorkspace="resolution_sum")
H_JoY_Sym = Load(r"fixatures/yspace_fit/H_JoY_Sym.nxs")
name = H_JoY_Sym.name()
resName = resolution.name()

rebinPars = "-20,0.5,20"
Rebin(InputWorkspace=name, Params=rebinPars, OutputWorkspace=name)
Rebin(InputWorkspace=resName, Params=rebinPars, OutputWorkspace=resName)

simple_gaussian_fit = True

def fitProfileInYSpace(wsYSpaceSym, wsRes):
    for minimizer_sum in ('Levenberg-Marquardt','Simplex'):
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = name+minimizer_sum+'_joy_sum_fitted')
        
        if (simple_gaussian_fit):
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()'''
        # else:
        #     function='''composite=Convolution,FixResolution=true,NumDeriv=true;
        #     name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
        #     name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
        #     +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
        #     y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''

        Fit(
            Function=function, 
            InputWorkspace=name+minimizer_sum+'_joy_sum_fitted', 
            Output=name+minimizer_sum+'_joy_sum_fitted',
            Minimizer=minimizer_sum
            )
        
        ws=mtd[name+minimizer_sum+'_joy_sum_fitted_Parameters']
        print('Using the minimizer: ',minimizer_sum)
        print('Hydrogen standard deviation: ',ws.cell(3,1),' +/- ',ws.cell(3,2))

    popt = [ws.cell(0,1), ws.cell(2,1), ws.cell(1,1), ws.cell(3,1)]
    pcov = [ws.cell(0,2), ws.cell(2,2), ws.cell(1,2), ws.cell(3,2)]
    return popt, pcov

# Need to check that dataX has the same seperation for the resolution and profile array
def optimizedFitInYSpace(wsYSpaceSym, wsRes):
    res = wsRes.extractY()[0]
    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    def convolvedGaussian(x, y0, x0, A, sigma):

        histWidths = x[1:] - x[:-1]
        if ~ (np.max(histWidths)==np.min(histWidths)):
            raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

        gaussFunc = gaussian(x, y0, x0, A, sigma)
        # # Add a zero to the end to increase length by one
        # gaussFunc = np.pad(gaussFunc, (0, 1), 'constant', constant_values=(0, 0))
        # Mode=same guarantees that convolved signal remains centered 
        convGauss = signal.convolve(gaussFunc, res, mode="same") * histWidths[0]
        # # Take out maximum poit to match form of original
        # convGaussMaxMask = convGauss==np.max(convGauss)
        # convGauss = convGauss[~convGaussMaxMask]
        return convGauss
    
    def gaussian(x, y0, x0, A, sigma):
        """Gaussian centered at zero"""
        return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
        
    p0 = [0, 0, 1, 5]
    popt, pcov = optimize.curve_fit(
        convolvedGaussian, dataX, dataY, p0=p0,
        sigma=dataE
    )
    yfit = convolvedGaussian(dataX, *popt)

    print("Best fit pars from curve_fit:\n", 
    popt, "\nCovariance matrix: \n", pcov)
    print("Width: ", popt[-1], "+/-", np.sqrt(pcov[-1,-1]))

    plt.figure()
    plt.errorbar(dataX, dataY, yerr=dataE, fmt="none", label="Data")
    plt.plot(dataX, yfit, label="curve_fit")
    plt.legend()
    plt.show()

    return popt, pcov, yfit


# def fitWithMinimize(ws, wsRes):
#     res = wsRes.extractY()[0]
#     dataY = ws.extractY()[0]
#     dataX = ws.extractX()[0]
#     dataE = ws.extractE()[0]
 
#     def errorFunc(pars, res, dataX, dataY, dataE):
#         yFit = convolvedGaussian(dataX, pars, res)
#         chi2 = (dataY - yFit)**2 / dataE**2
#         return np.sum(chi2)

#     def convolvedGaussian(x, pars, res):
#         gaussFunc = gaussian(x, pars)
#         # Mode=same guarantees that convolved signal remains centered 
#         convGauss = signal.convolve(gaussFunc, res, mode="same")
#         return convGauss
    
#     def gaussian(x, pars):
#         y0, x0, A, sigma = pars
#         """Gaussian centered at zero"""
#         return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

#     pars=np.array([0,0,1,5])
#     result = optimize.minimize(
#         errorFunc,
#         pars,
#         args=(res, dataX, dataY, dataE),
#         method="SLSQP",
#         options={"disp":True}
#     )
#     print("Best fit pars using scipy's minimize: \n",
#             result["x"])
    

ws = mtd[name]
wsRes = mtd[resName]

poptOri, pcovOri = fitProfileInYSpace(ws, wsRes)
popt, pcov, yfit = optimizedFitInYSpace(ws, wsRes)
#fitWithMinimize(H_JoY_Sym, resolution)

# ---- somethjing dirty down below

def convolvedGaussian(x, popt, res):
    y0, x0, A, sigma = popt

    histWidths = x[1:] - x[:-1]
    if ~ (np.max(histWidths)==np.min(histWidths)):
        raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

    gaussFunc = gaussian(x, y0, x0, A, sigma)
    # # Add a zero to the end to increase length by one
    # gaussFunc = np.pad(gaussFunc, (0, 1), 'constant', constant_values=(0, 0))
    # Mode=same guarantees that convolved signal remains centered 
    convGauss = signal.convolve(gaussFunc, res, mode="same") * histWidths[0]
    # # Take out maximum poit to match form of original
    # convGaussMaxMask = convGauss==np.max(convGauss)
    # convGauss = convGauss[~convGaussMaxMask]
    return convGauss

def gaussian(x, y0, x0, A, sigma):
    """Gaussian centered at zero"""
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


res = wsRes.extractY()[0]
dataY = ws.extractY()[0]
dataX = ws.extractX()[0]
dataE = ws.extractE()[0]

print("Original parameters:\n",poptOri, 
    "\nOptimized parameters: \n", popt)
yfitOpt = convolvedGaussian(dataX, popt, res)
yfitOri = convolvedGaussian(dataX, poptOri, res)

plt.figure()
plt.errorbar(dataX, dataY, yerr=dataE, fmt="none", label="Data")
plt.plot(dataX, yfitOpt, label="curve_fit")
plt.plot(dataX, yfitOri, label="mantid_Fit")
plt.legend()
plt.show()
