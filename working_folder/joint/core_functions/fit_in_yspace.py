import matplotlib.pyplot as plt
import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from scipy import ndimage, signal
from pathlib import Path
from iminuit import Minuit, cost, util
from iminuit.util import make_func_code

repoPath = Path(__file__).absolute().parent  # Path to the repository



def fitInYSpaceProcedure(ic, wsFinal, ncpForEachMass):
    firstMass = ic.masses[0]
    wsResSum, wsRes = calculateMantidResolution(ic, wsFinal, firstMass)
    
    wsSubMass = subtractAllMassesExceptFirst(ic, wsFinal, ncpForEachMass)
    wsYSpace, wsQ = convertToYSpace(ic.rebinParametersForYSpaceFit, wsSubMass, firstMass) 
    wsYSpaceAvg = weightedAvg(wsYSpace)
    
    if ic.symmetrisationFlag:
        wsYSpaceAvg = symmetrizeWs(ic.symmetriseHProfileUsingAveragesFlag, wsYSpaceAvg)

    fitProfileMinuit(ic, wsYSpaceAvg, wsResSum)
    fitProfileMantidFit(ic, wsYSpaceAvg, wsResSum)
    
    printYSpaceFitResults(wsYSpaceAvg.name())

    yfitResults = ResultsYFitObject(ic, wsFinal.name())
    yfitResults.save()

    if ic.globalFitFlag:
        fitGlobalFit(wsYSpace, wsQ, wsRes, "Simplex", ic.singleGaussFitToHProfile, wsSubMass.name())


def calculateMantidResolution(ic, ws, mass):
    resName = ws.name()+"_Resolution"
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
        Rebin(InputWorkspace="tmp", Params=ic.rebinParametersForYSpaceFit, OutputWorkspace="tmp")

        if index == 0:   # Ensures that workspace has desired units
            RenameWorkspace("tmp",  resName)
        else:
            AppendSpectra(resName, "tmp", OutputWorkspace=resName)
   
    MaskDetectors(resName, WorkspaceIndexList=ic.maskedDetectorIdx)
    wsResSum = SumSpectra(InputWorkspace=resName, OutputWorkspace=resName+"_Sum")
 
    normalise_workspace(wsResSum)
    DeleteWorkspace("tmp")
    return wsResSum, mtd[resName]

    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def subtractAllMassesExceptFirst(ic, ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMassExceptFirst = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotalExceptFirst = np.sum(ncpForEachMassExceptFirst, axis=0)

    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_Mass0")
    for j in range(wsSubMass.getNumberHistograms()):
        if wsSubMass.spectrumInfo().isMasked(j):
            continue

        binWidths = wsSubMass.dataX(j)[1:] - wsSubMass.dataX(j)[:-1]
        wsSubMass.dataY(j)[:-1] -= ncpTotalExceptFirst[j] * binWidths

     # Mask spectra again, to be seen as masked from Mantid's perspective
    MaskDetectors(Workspace=wsSubMass, WorkspaceIndexList=ic.maskedDetectorIdx)  

    if np.any(np.isnan(wsSubMass.extractY())):
        raise ValueError("The workspace for the isolated first mass countains NaNs in non-masked spectra, might cause problems!")
    return wsSubMass


def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertToYSpace(rebinPars, ws0, mass):
    wsJoY, wsQ = ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, 
        OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
    wsJoY = Rebin(
        InputWorkspace=wsJoY, Params=rebinPars, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )
    wsQ = Rebin(
        InputWorkspace=wsQ, Params=rebinPars, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_Q"
        )
    
    # If workspace has nans present, normalization will put zeros on the full spectrum
    assert np.any(np.isnan(wsJoY.extractY()))==False, "Nans present before normalization."
    
    normalise_workspace(wsJoY)
    return wsJoY, wsQ


def weightedAvg(wsYSpace):
    """Returns ws with weighted avg of input ws"""
    
    dataY = wsYSpace.extractY()
    dataE = wsYSpace.extractE()

    # TODO: Revise this, some zeros might not be cut offs
    # TODO: If one column is all zeros ir puts dataY=nan and dataE=inf, will throw an error when fitting
    # Replace dataE at cut-offs by np.inf?

    dataY[dataY==0] = np.nan
    dataE[dataE==0] = np.nan

    meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))

    tempWs = SumSpectra(wsYSpace)
    newWs = CloneWorkspace(tempWs, OutputWorkspace=wsYSpace.name()+"_Weighted_Avg")
    newWs.dataY(0)[:] = meanY
    newWs.dataE(0)[:] = meanE
    DeleteWorkspace(tempWs)
    return newWs


def symmetrizeWs(avgSymFlag, avgYSpace):
    """Symmetrizes workspace,
       Needs to have symmetric binning"""

    dataX = avgYSpace.extractX()
    dataY = avgYSpace.extractY()
    dataE = avgYSpace.extractE()

    yFlip = np.flip(dataY, axis=1)
    eFlip = np.flip(dataE, axis=1)

    if avgSymFlag:
        # Inverse variance weighting
        dataYSym = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
        dataESym = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)

    # TODO: Maybe take out possibility of symmetrisation by mirror
    else:
        # Mirroring positive values from negative ones
        dataYSym = np.where(dataX>0, yFlip, dataY)
        dataESym = np.where(dataX>0, eFlip, dataE)

    Sym = CloneWorkspace(avgYSpace, OutputWorkspace=avgYSpace.name()+"_Symmetrised")
    for i in range(Sym.getNumberHistograms()):
        Sym.dataY(i)[:] = dataYSym[i]
        Sym.dataE(i)[:] = dataESym[i]
    return Sym


def fitProfileMinuit(ic, wsYSpaceSym, wsRes):
    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    resY = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]

    if ic.singleGaussFitToHProfile:
        def model(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

        funcSig = ["x", "y0", "A", "x0", "sigma"]
        initPars = {"y0":0, "A":1, "x0":0, "sigma":5}

    else:
        def model(x, A, x0, sigma1, c4, c6):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        
        funcSig = ["x", "A", "x0", "sigma1", "c4", "c6"]
        initPars = {"A":1, "x0":0, "sigma1":5, "c4":0, "c6":0}

    xDense, xDelta, resDense = chooseXDense(resX, resY, True)

    def convolvedModel(x, *pars):
        convDense = signal.convolve(model(xDense, *pars), resDense, mode="same") * xDelta
        return np.interp(x, xDense, convDense)

    convolvedModel.func_code = make_func_code(funcSig)

    # Ignore values that are zero, eg. cut-offs
    nonZeros = dataY != 0
    dataXNZ = dataX[nonZeros]
    dataYNZ = dataY[nonZeros]
    dataENZ = dataE[nonZeros]

    # Fit with Minuit
    costFun = cost.LeastSquares(dataXNZ, dataYNZ, dataENZ, convolvedModel)
    m = Minuit(costFun, **initPars)
    m.limits["A"] = (0, None)

    m.simplex()
    if ic.singleGaussFitToHProfile:
        m.migrad()
    else:
        def constrFunc(*pars):
            return convolvedModel(dataX, *pars)   # GC > 0 after convolution

        m.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

    # Explicit calculation of Hessian after the fit
    m.hesse()

    # Weighted Chi2
    chi2 = m.fval / (len(dataX)-m.nfit)

    # Propagate error to yfit
    # Takes in the best fit parameters and their covariance matrix
    # Outputs the best fit curve with std in the diagonal
    dataYFit, dataYCov = util.propagate(lambda pars: convolvedModel(dataX, *pars), m.values, m.covariance)
    dataYSigma = np.sqrt(np.diag(dataYCov))

    # Weight the confidence band
    # TODO: Shoud it be weighted?
    dataYSigma *= chi2

    Residuals = dataY - dataYFit

    # Create workspace to store best fit curve and errors on the fit
    CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
                    DataY=np.concatenate((dataY, dataYFit, Residuals)), 
                    DataE=np.concatenate((dataE, dataYSigma, np.zeros(len(dataE)))),
                    NSpec=3,
                    OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit")
    
    # Calculate correlation matrix
    corr = m.covariance.correlation()
    corr *= 100

    # Create correlation tableWorkspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_NormalizedCovarianceMatrix")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str',name="Name")
    for p in m.parameters:
        tableWS.addColumn(type='float',name=p)
    for p, arr in zip(m.parameters, corr):
        tableWS.addRow([p] + list(arr))
    

    # Create Parameters workspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_Parameters")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str', name="Name")
    tableWS.addColumn(type='float', name="Value")
    tableWS.addColumn(type='float', name="Error")
    tableWS.addColumn(type='float', name="Auto Minos Error-")
    tableWS.addColumn(type='float', name="Auto Minos Error+")
    tableWS.addColumn(type='float', name="Manual Minos Error-")
    tableWS.addColumn(type='float', name="Manual Minos Error+")


    # Extract info from fit before running any MINOS
    parameters = m.parameters
    values = m.values
    errors = m.errors
    
    bestFitVals = {}
    bestFitErrs = {}
    for p, v, e in zip(m.parameters, m.values, m.errors):
        bestFitVals[p] = v
        bestFitErrs[p] = e
    fValsMin = m.fval

    try:  # Compute errors from MINOS, fails if constraint forces result away from minimum
        if ic.forceManualMinos:
            try:
                constrFunc(*m.values)      # Check if constraint is present
                raise(RuntimeError)        # If so, jump to Manual MINOS

            except UnboundLocalError:      # Constraint not present, default to auto MINOS
                print("\nConstraint not present, using default Automatic MINOS ...\n")
                pass
        
        m.minos()
        me = m.merrors
        for p, v, e in zip(parameters, values, errors):
            tableWS.addRow([p, v, e, me[p].lower, me[p].upper, 0, 0])   
        plotAutoMinos(m)

    except RuntimeError:
        merrors = runAndPlotManualMinos(m, constrFunc, bestFitVals, bestFitErrs)     # Changes values of minuit obj m, do not use m below this point
        for p, v, e in zip(parameters, values, errors):
            tableWS.addRow([p, v, e, 0, 0, merrors[p][0], merrors[p][1]])

    tableWS.addRow(["Cost function", chi2, 0, 0, 0, 0, 0])
    return 


def chooseXDense(x, res, flag):
    # TODO: Need to sort out the best density for the convolution

    """Make high density symmetric grid for convolution"""

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if flag:
        if x.size % 2 == 0:
            dens = x.size+1  # If even change to odd
        else:
            dens = x.size    # If odd, keep being odd)
    else:
        dens = 1000

    xDense = np.linspace(np.min(x), np.max(x), dens)
    xDelta = xDense[1] - xDense[0]
    resDense = np.interp(xDense, x, res)
    return xDense, xDelta, resDense



def fitProfileMantidFit(ic, wsYSpaceSym, wsRes):
    print('\nFitting on the sum of spectra in the West domain ...\n')     
    for minimizer in ['Levenberg-Marquardt','Simplex']:
        outputName = wsYSpaceSym.name()+"_Fitted_"+minimizer
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = outputName)
        
        if ic.singleGaussFitToHProfile:
            function=f"""composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()"""
        else:
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)+c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            A=1,x0=0,sigma1=4.0,c4=0.0,c6=0.0,ties=(),constraints=(0<c4,0<c6)
            """

        Fit(
            Function=function, 
            InputWorkspace=outputName,
            Output=outputName,
            Minimizer=minimizer
            )
        # Fit produces output workspaces with results
    return 


def runAndPlotManualMinos(minuitObj, constrFunc, bestFitVals, bestFitErrs):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})  #subplot_kw={'projection':'mantid'}
    fig.canvas.set_window_title("Manual Implementation of Minos algorithm")

    merrors = {}
    for p, ax in zip(minuitObj.parameters, axs.flat):
        lerr, uerr = runMinosForPar(minuitObj, constrFunc, p, 2, ax, bestFitVals, bestFitErrs)
        merrors[p] = np.array([lerr, uerr])

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    fig.show()
    return merrors


def runMinosForPar(minuitObj, constrFunc, var:str, bound:int, ax, bestFitVals, bestFitErrs):

    # Set parameters to previously found minimum to restart procedure
    for p in bestFitVals:
        minuitObj.values[p] = bestFitVals[p]
        minuitObj.errors[p] = bestFitErrs[p]

    # Run Fitting procedures again to be on the safe side and reset to minimum
    minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
    minuitObj.hesse()

    # Extract parameters from minimum
    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]
    # Store fval of best fit
    fValsMin = minuitObj.fval      # Used to calculate error bands at the end

    # Initiate arrays
    varSpace = np.linspace(varVal - bound*varErr, varVal + bound*varErr, 30)
    fValsScipy = np.zeros(varSpace.shape)
    fValsMigrad = np.zeros(varSpace.shape)

    # Run Minos algorithm
    minuitObj.fixed[var] = True        # Variable is fixed at each iteration

    # Split variable space in two parts to start loop from minimum
    lhsRange, rhsRange = np.split(np.arange(varSpace.size), 2)
    betterRange = [rhsRange, np.flip(lhsRange)]  # First do rhs, then lhs, starting from minima
    for side in betterRange:
        # Reset values and errors to minima
        for p in bestFitVals:
            minuitObj.values[p] = bestFitVals[p]
            minuitObj.errors[p] = bestFitErrs[p]

        # Unconstrained fit
        for i in side.astype(int):
            minuitObj.values[var] = varSpace[i]      # Fix variable
            minuitObj.migrad()     
            fValsMigrad[i] = minuitObj.fval

        # Reset values and errors to minima
        for p in bestFitVals:
            minuitObj.values[p] = bestFitVals[p]
            minuitObj.errors[p] = bestFitErrs[p]

        # Constrained fit       
        for i in side.astype(int):
            minuitObj.values[var] = varSpace[i]      # Fix variable
            minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
            fValsScipy[i] = minuitObj.fval
        
    minuitObj.fixed[var] = False    # Release variable       

    # Use intenpolation to create dense array of fmin values 
    varSpaceDense = np.linspace(np.min(varSpace), np.max(varSpace), 100000)
    fValsScipyDense = np.interp(varSpaceDense, varSpace, fValsScipy)
    # Calculate points of intersection with line delta fmin val = 1
    idxErr = np.argwhere(np.diff(np.sign(fValsScipyDense - fValsMin - 1)))
    
    if idxErr.size != 2:    # Intersections not found, do not plot error range
        lerr, uerr = 0., 0.   
    else:
        lerr, uerr = varSpaceDense[idxErr].flatten() - varVal

    ax.plot(varSpaceDense, fValsScipyDense, label="fVals Constr Scipy")
    plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr)
  
    return lerr, uerr


def plotAutoMinos(minuitObj):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})  #subplot_kw={'projection':'mantid'}
    fig.canvas.set_window_title("Plot of automatic Minos algorithm")

    for p, ax in zip(minuitObj.parameters, axs.flat):
        loc, fvals, status = minuitObj.mnprofile(p, bound=2)
        

        minfval = minuitObj.fval
        minp = minuitObj.values[p]
        hessp = minuitObj.errors[p]
        lerr = minuitObj.merrors[p].lower
        uerr = minuitObj.merrors[p].upper
        plotProfile(ax, p, loc, fvals, lerr, uerr, minfval, minp, hessp)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    fig.show()   


def plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr):
    """
    Plots likelihood profilef for the Migrad fvals.
    varSpace : x axis
    fValsMigrad : y axis
    """

    ax.set_title(var+f" = {varVal:.3f} {lerr:.3f} {uerr:+.3f}")

    ax.plot(varSpace, fValsMigrad, label="fVals Migrad")

    ax.axvspan(lerr+varVal, uerr+varVal, alpha=0.2, color="red", label="Minos error")
    ax.axvspan(varVal-varErr, varVal+varErr, alpha=0.2, color="grey", label="Hessian Std error")
    
    ax.axvline(varVal, 0.03, 0.97, color="k", ls="--")
    ax.axhline(fValsMin+1, 0.03, 0.97, color="k")
    ax.axhline(fValsMin, 0.03, 0.97, color="k")


def printYSpaceFitResults(wsJoYName):
    print("\nFit in Y Space results:")

    wsFitLM = mtd[wsJoYName + "_Fitted_Levenberg-Marquardt_Parameters"]
    wsFitSimplex = mtd[wsJoYName + "_Fitted_Simplex_Parameters"]
    wsFitMinuit = mtd[wsJoYName + "_Fitted_Minuit_Parameters"]

    for tableWS in [wsFitLM, wsFitSimplex, wsFitMinuit]:
        print("\n"+" ".join(tableWS.getName().split("_")[-3:])+":")
        # print("    ".join(tableWS.keys()))
        for key in tableWS.keys():
            if key=="Name":
                print(f"{key:12s}:  "+"  ".join([f"{elem:7.8s}" for elem in tableWS.column(key)]))
            else:
                print(f"{key:12s}: "+"  ".join([f"{elem:7.4f}" for elem in tableWS.column(key)]))
    print("\n")


class ResultsYFitObject:

    def __init__(self, ic, wsFinalName):
        # Extract most relevant information from ws
        wsFinal = mtd[wsFinalName]
        wsMass0 = mtd[wsFinalName + "_Mass0"]
        if ic.symmetrisationFlag:
            wsJoYAvg = mtd[wsFinalName + "_Mass0_JoY_Weighted_Avg_Symmetrised"]
        else:
            wsJoYAvg = mtd[wsFinalName + "_Mass0_JoY_Weighted_Avg"]
        wsResSum = mtd[wsFinalName + "_Resolution_Sum"]

        self.finalRawDataY = wsFinal.extractY()
        self.finalRawDataE = wsFinal.extractE()
        self.HdataY = wsMass0.extractY()
        self.YSpaceSymSumDataY = wsJoYAvg.extractY()
        self.YSpaceSymSumDataE = wsJoYAvg.extractE()
        self.resolution = wsResSum.extractY()

        # Extract best fit parameters from workspaces
        wsFitLM = mtd[wsJoYAvg.name() + "_Fitted_Levenberg-Marquardt_Parameters"]
        wsFitSimplex = mtd[wsJoYAvg.name() + "_Fitted_Simplex_Parameters"]
        wsFitMinuit = mtd[wsJoYAvg.name() + "_Fitted_Minuit_Parameters"]

        noPars = len(wsFitLM.column("Value"))
        popt = np.zeros((3, noPars))
        perr = np.zeros((3, noPars))
        for i, ws in enumerate([wsFitMinuit, wsFitLM, wsFitSimplex]):
            popt[i] = ws.column("Value")
            perr[i] = ws.column("Error")
        self.popt = popt
        self.perr = perr

        self.savePath = ic.ySpaceFitSavePath
        self.singleGaussFitToHProfile = ic.singleGaussFitToHProfile


    def save(self):
        np.savez(self.savePath,
                 YSpaceSymSumDataY=self.YSpaceSymSumDataY,
                 YSpaceSymSumDataE=self.YSpaceSymSumDataE,
                 resolution=self.resolution, 
                 HdataY=self.HdataY,
                 finalRawDataY=self.finalRawDataY, 
                 finalRawDataE=self.finalRawDataE,
                 popt=self.popt, 
                 perr=self.perr)


# Functions for Global Fit

def fitGlobalFit(wsJoY, wsQ, wsRes, minimizer, gaussFitFlag, wsFirstMassName):
    replaceNansWithZeros(wsJoY)
    wsGlobal = artificialErrorsInUnphysicalBins(wsJoY)
    wsQInv = createOneOverQWs(wsQ)

    avgWidths = globalFitProcedure(wsGlobal, wsQInv, wsRes, minimizer, gaussFitFlag, wsFirstMassName)


def replaceNansWithZeros(ws):
    for j in range(ws.getNumberHistograms()):
        ws.dataY(j)[np.isnan(ws.dataY(j)[:])] = 0
        ws.dataE(j)[np.isnan(ws.dataE(j)[:])] = 0


def artificialErrorsInUnphysicalBins(wsJoY):
    wsGlobal = CloneWorkspace(InputWorkspace=wsJoY, OutputWorkspace=wsJoY.name()+'_Global')
    for j in range(wsGlobal.getNumberHistograms()):
        wsGlobal.dataE(j)[wsGlobal.dataE(j)[:]==0] = 0.1
    
    assert np.any(np.isnan(wsGlobal.extractE())) == False, "Nan present in input workspace need to be replaced by zeros."

    return wsGlobal


def createOneOverQWs(wsQ):
    wsInvQ = CloneWorkspace(InputWorkspace=wsQ, OutputWorkspace=wsQ.name()+"_Inverse")
    for j in range(wsInvQ.getNumberHistograms()):
        nonZeroFlag = wsInvQ.dataY(j)[:] != 0
        wsInvQ.dataY(j)[nonZeroFlag] = 1 / wsInvQ.dataY(j)[nonZeroFlag]

        ZeroIdxs = np.argwhere(wsInvQ.dataY(j)[:]==0)   # Indxs of zero elements
        if ZeroIdxs.size != 0:     # When zeros are present
            wsInvQ.dataY(j)[ZeroIdxs[0] - 1] = 0       # Put a zero before the first zero
    
    return wsInvQ


def globalFitProcedure(wsGlobal, wsQInv, wsRes, minimizer, gaussFitFlag, wsFirstMassName):
    if gaussFitFlag:
        convolution_template = """
        (composite=Convolution,$domains=({0});
        name=Resolution,Workspace={1},WorkspaceIndex={0};
            (
            name=UserFunction,Formula=
            A*exp( -(x-x0)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5,
            A=1.,x0=0.,Sigma=6.0,  ties=();
                (
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace={2},WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
                name=UserFunction,Formula=
                Sigma*1.4142/12.*exp( -(x)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
                *((8*((x)/sqrt(2.)/Sigma)^3-12*((x)/sqrt(2.)/Sigma))),
                Sigma=6.0);ties=()
                )
            )"""
    else:
        convolution_template = """
        (composite=Convolution,$domains=({0});
        name=Resolution,Workspace={1},WorkspaceIndex={0};
            (
            name=UserFunction,Formula=
            A*exp( -(x-x0)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
            *(1+c4/32*(16*((x-x0)/sqrt(2.)/Sigma)^4-48*((x-x0)/sqrt(2.)/Sigma)^2+12)),
            A=1.,x0=0.,Sigma=6.0, c4=0, ties=();
                (
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace={2},WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
                name=UserFunction,Formula=
                Sigma*1.4142/12.*exp( -(x)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
                *((8*((x)/sqrt(2.)/Sigma)^3-12*((x)/sqrt(2.)/Sigma))),
                Sigma=6.0);ties=()
                )
            )"""    

    print('\nGlobal fit in the West domain over 8 mixed banks\n')
    widths = []  
    for bank in range(8):
        dets=[bank, bank+8, bank+16, bank+24]

        convolvedFunctionsList = []
        ties = ["f0.f1.f1.f1.Sigma=f0.f1.f0.Sigma"]
        datasets = {'InputWorkspace' : wsGlobal.name(),
                    'WorkspaceIndex' : dets[0]}

        print("Detectors: ", dets)

        counter = 0
        for i in dets:

            print(f"Considering spectrum {wsGlobal.getSpectrumNumbers()[i]}")
            if wsGlobal.spectrumInfo().isMasked(i):
                print(f"Skipping masked spectrum {wsGlobal.getSpectrumNumbers()[i]}")
                continue

            thisIterationFunction = convolution_template.format(counter, wsRes.name(), wsQInv.name())
            convolvedFunctionsList.append(thisIterationFunction)

            if counter > 0:
                ties.append('f{0}.f1.f0.Sigma= f{0}.f1.f1.f1.Sigma=f0.f1.f0.Sigma'.format(counter))
                #TODO: Ask if conditional statement goes here
                #ties.append('f{0}.f1.f0.c4=f0.f1.f0.c4'.format(counter))
                #ties.append('f{0}.f1.f1.f1.c3=f0.f1.f1.f1.c3'.format(counter))

                # Attach datasets
                datasets[f"InputWorkspace_{counter}"] = wsGlobal.name()
                datasets[f"WorkspaceIndex_{counter}"] = i
            counter += 1

        multifit_func = f"composite=MultiDomainFunction; {';'.join(convolvedFunctionsList)}; ties=({','.join(ties)})"
        minimizer_string = f"{minimizer}, AbsError=0.00001, RealError=0.00001, MaxIterations=2000"

        # Unpack dictionary as arguments
        Fit(multifit_func, Minimizer=minimizer_string, Output=wsFirstMassName+f'Joy_Mixed_Banks_Bank_{str(bank)}_fit', **datasets)
        
        # Select ws with fit results
        ws=mtd[wsFirstMassName+f'Joy_Mixed_Banks_Bank_{str(bank)}_fit_Parameters']
        print(f"Bank: {str(bank)} -- sigma={ws.cell(2,1)} +/- {ws.cell(2,2)}")
        widths.append(ws.cell(2,1))

        # DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_NormalisedCovarianceMatrix')
        # DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_Workspaces') 
    print('\nAverage hydrogen standard deviation: ',np.mean(widths),' +/- ', np.std(widths))
    return widths
