from scipy import stats
import numpy as np
from .analysis_functions import arraysFromWS, histToPointData, prepareFitArgs, fitNcpToArray
from .analysis_functions import iterativeFitForDataReduction
from .fit_in_yspace import fitInYSpaceProcedure
from .procedures import runJointBackAndForwardProcedure, runIndependentIterativeProcedure
from mantid.api import AnalysisDataService, mtd
from mantid.simpleapi import CloneWorkspace, SaveNexus, Load
from pathlib import Path
import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")
currentPath = Path(__file__).parent.absolute()


def runIndependentBootstrap(singleIC, nSamples, yFitIC, checkUserIn=True, fastBootstrap=False):
    inputIC = [singleIC]
    return runBootstrap(inputIC, nSamples, yFitIC, checkUserIn, fastBootstrap)


def runJointBootstrap(bckwdIC, fwdIC, nSamples, yFitIC, checkUserIn=True, fastBootstrap=False):
    inputIC = [bckwdIC, fwdIC]
    return runBootstrap(inputIC, nSamples, yFitIC, checkUserIn, fastBootstrap)


def runBootstrap(inputIC, nSamples, yFitIC, checkUserIn, fastBootstrap):
    """
    Runs main procedure for each bootstrap replica (created from the residuals)
    inputIC can have one or two (back, forward) IC inputs.
    """

    # setICsToDefault(inputIC, yFitIC)

    # # t0 = time.time()
    # parentResults = runMainProcedure(inputIC, yFitIC)
    # # t1 = time.time()

    # parentWSnNCPs = selectParentWorkspaces(inputIC, fastBootstrap)
    # checkResiduals(parentWSnNCPs)

    parentResults, parentWSnNCP = runOriginalBeforeBootstrap(inputIC, yFitIC, fastBootstrap)

    # if checkUserIn:
    #     userIn = checkUserInput(t1-t0, nSamples)
    #     if (userIn != "y") and (userIn != "Y"): return

    # Form each bootstrap workspace and run ncp fit with MS corrections
    setOutputDirectories(inputIC, nSamples)
    parentWSNCPSavePaths = convertWSToSavePaths(parentWSnNCP)
    bootResults = initializeResults(parentResults, nSamples)
    for j in range(nSamples):
        AnalysisDataService.clear()

        bootInputWS = createBootstrapWS(parentWSNCPSavePaths)

        plugBootWSIntoIC(inputIC, bootInputWS, fastBootstrap)   

        # Run procedure for bootstrap ws
        iterResults = runMainProcedure(inputIC, yFitIC)

        storeBootIter(bootResults, j, iterResults)
        saveBootstrapResults(bootResults, inputIC, fastBootstrap)
    return bootResults


def runJackknife(inputIC, yFitIC, checkUserIn, fastBootstrap):
    """
    Runs main procedure for each bootstrap replica (created from the residuals)
    inputIC can have one or two (back, forward) IC inputs.
    """

    # setICsToDefault(inputIC, yFitIC)

    # # t0 = time.time()
    # parentResults = runMainProcedure(inputIC, yFitIC)
    # # t1 = time.time()

    # parentWSnNCPs = selectParentWorkspaces(inputIC, fastBootstrap)
    # checkResiduals(parentWSnNCPs)

    parentResults, parentWSnNCP = runOriginalBeforeBootstrap(inputIC, yFitIC, fastBootstrap)

    # if checkUserIn:
    #     userIn = checkUserInput(t1-t0, nSamples)
    #     if (userIn != "y") and (userIn != "Y"): return

    # For the Jackknife, one sample per point, so nSamples = no of points
    nSamples = parentWSnNCP[0][0].dataY(0).size
    runningJackknife = True

    setOutputDirectories(inputIC, nSamples, runningJackknife)
    parentWSNCPSavePaths = convertWSToSavePaths(parentWSnNCP)
    bootResults = initializeResults(parentResults, nSamples)
    for j in range(nSamples):
        AnalysisDataService.clear()

        bootInputWS = createBootstrapWS(parentWSNCPSavePaths, runningJackknife)

        plugBootWSIntoIC(inputIC, bootInputWS, fastBootstrap)   

        # Run procedure for bootstrap ws
        iterResults = runMainProcedure(inputIC, yFitIC)

        storeBootIter(bootResults, j, iterResults)
        saveBootstrapResults(bootResults, inputIC, fastBootstrap)
    return bootResults


def runOriginalBeforeBootstrap(inputIC, yFitIC, fastBootstrap):
    """Runs unaltered procedure and extracts parent workspaces according to fast/slow bootstrap."""
    
    setICsToDefault(inputIC, yFitIC)
    parentResults = runMainProcedure(inputIC, yFitIC)
    parentWSnNCP = selectParentWorkspaces(inputIC, fastBootstrap)
    checkResiduals(parentWSnNCP)   
    return parentResults, parentWSnNCP



# def runJackknife(inputIC, yFitIC, checkUserIn, fastJack):



def setICsToDefault(inputIC, yFitIC):
    """Disables some features of yspace fit, makes sure the default """
    # Disable global fit 
    yFitIC.globalFitFlag = False
    # Run automatic minos by default
    yFitIC.forceManualMinos = False
    # Hide plots
    yFitIC.showPlots = False

    for IC in inputIC:    # Default is not to run with bootstrap ws
        IC.bootSample = False


def setOutputDirectories(inputIC, nSamples, runJackknife=False):
    "Forms Bootstrap output paths"

     # Select script name and experiments path
    sampleName = inputIC[0].scriptName   # Name of sample currently running
    experimentsPath = currentPath/".."/".."/"experiments"

    if runJackknife:
        bootOutPath = experimentsPath / sampleName / "jackknife_data"
    else:
        bootOutPath = experimentsPath / sampleName / "bootstrap_data"
    bootOutPath.mkdir(exist_ok=True)

    quickPath = bootOutPath / "quick"
    slowPath = bootOutPath / "slow"
    quickPath.mkdir(exist_ok=True)
    slowPath.mkdir(exist_ok=True)

    jackOutPath = experimentsPath / sampleName / "jacknnife_data"

    for IC in inputIC:    # Make save paths for .npz files
        # Build Filename based on ic
        corr = ""
        if IC.MSCorrectionFlag & (IC.noOfMSIterations>1):
            corr+="_MS"
        if IC.GammaCorrectionFlag & (IC.noOfMSIterations>1):
            corr+="_GC"

        fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"
        bootName = fileName + f"_nsampl_{nSamples}"+".npz"
        bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}"+".npz"

        IC.bootQuickSavePath = quickPath / bootName
        IC.bootQuickYFitSavePath = quickPath / bootNameYFit
        IC.bootSlowSavePath = slowPath / bootName
        IC.bootSlowYFitSavePath = slowPath / bootNameYFit   


def runMainProcedure(inputIC, yFitIC):

    if len(inputIC) == 2:
        bckwdIC, fwdIC = inputIC

        wsParent, bckwdScatResP, fwdScatResP = runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=False)
        yFitResultsParent = fitInYSpaceProcedure(yFitIC, fwdIC, wsParent)

        parentResults = [bckwdScatResP, fwdScatResP, yFitResultsParent]
    
    elif len(inputIC) == 1:
        singleIC = inputIC[0]

        wsParent, singleScatResP = runIndependentIterativeProcedure(singleIC, clearWS=False)
        yFitResultsParent = fitInYSpaceProcedure(yFitIC, singleIC, wsParent)

        parentResults = [singleScatResP, yFitResultsParent]  

    else:
        raise ValueError("len(inputIC) needs to be one or two.")

    return parentResults


def checkUserInput(t, nSamples):
    print(f"\nRan original procedure, time: {t:.2f} s.")
    print(f"Estimated time for {nSamples} samples: {nSamples*t/60/60:.1f} hours.")
    userIn = input("Continue with Bootstrap? y/n: ")
    return userIn


def selectParentWorkspaces(inputIC, fastBoot):
    """
    Selects parent workspace from which the Bootstrap replicas will be created.
    If fast mode, the parent ws is the final ws after MS corrections."""
    
    parentWSnNCPs = []
    for IC in inputIC:

        if fastBoot:
            wsIter = str(IC.noOfMSIterations - 1)    # Selects last ws after MS corrections
        else:
            wsIter = "0"

        parentWS = mtd[IC.name+wsIter]
        # Extract total ncp fit to the selected workspace
        parentNCP = mtd[parentWS.name()+"_TOF_Fitted_Profiles"]

        parentWSnNCPs.append([parentWS, parentNCP])

    return parentWSnNCPs


def convertWSToSavePaths(parentWSnNCPs):
    savePaths = []
    for wsList in parentWSnNCPs:
        savePaths.append(saveWorkspacesLocally(wsList))
    return savePaths


def saveWorkspacesLocally(args):
    wsSavePaths = []
    for ws in args:

        keys = ws.name().split("_")
        saveName = "Parent"

        if "FORWARD" in keys:
            saveName += "_Front"
        elif "BACKWARD" in keys:
            saveName += "_Back"

        if "Profiles" in keys:
            saveName += "_NCP"
        
        saveName += ".nxs"
        savePath = currentPath / "bootstrap_ws" / saveName
        SaveNexus(ws, str(savePath))
        wsSavePaths.append(savePath)

    return wsSavePaths


def initializeResults(parentResults, nSamples):
    bootResultObjs = []
    for pResults in parentResults:
        try:
            resultsObj = BootScattResults(pResults, nSamples)
        except AttributeError:
            resultsObj = BootYFitResults(pResults, nSamples)

        bootResultObjs.append(resultsObj)
    return bootResultObjs


class BootScattResults:

    def __init__(self, parentResults, nSamples):
        self.parentResult = parentResults.all_spec_best_par_chi_nit[-1]
        self.bootSamples = np.zeros((nSamples, *self.parentResult.shape))

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.all_spec_best_par_chi_nit[-1]
    
    def saveResults(self, IC, fastBootstrap):
        if fastBootstrap:
            savePath = IC.bootQuickSavePath
        else:
            savePath = IC.bootSlowSavePath

        np.savez(savePath, boot_samples=self.bootSamples,
             parent_result=self.parentResult)


class BootYFitResults:

    def __init__(self, parentResults, nSamples):
        self.parentPopt = parentResults.popt
        self.parentPerr = parentResults.perr
        self.bootSamples = np.zeros((nSamples, *self.parentPopt.shape))

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.popt
    
    def saveResults(self, IC, fastBootstrap):
        if fastBootstrap:
            savePath = IC.bootQuickYFitSavePath
        else:
            savePath = IC.bootSlowYFitSavePath

        np.savez(savePath, boot_samples=self.bootSamples,
             parent_popt=self.parentPopt, parent_perr=self.parentPerr)    


def storeBootIter(bootResultObjs, j, bootIterResults):
    for bootObj, iterRes in zip(bootResultObjs, bootIterResults):
        bootObj.storeBootIterResults(j, iterRes)


def saveBootstrapResults(bootResultObjs, inputIC, fastBootstrap):
    for bootObj, IC in zip(bootResultObjs, inputIC):
        bootObj.saveResults(IC, fastBootstrap)
    bootResultObjs[-1].saveResults(inputIC[-1], fastBootstrap)


def createBootstrapWS(parentWSNCPSavePaths):
    """
    Creates bootstrap ws replica.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit
    """

    bootInputWS = []
    for (parentWSPath, totNcpWSPath) in parentWSNCPSavePaths:
        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)

        dataY = parentWS.extractY()[:, :-1]
        totNcp = totNcpWS.extractY()[:, :]     

        residuals = dataY - totNcp

        bootRes = bootstrapResidualsSample(residuals)
        bootDataY = totNcp + bootRes

        wsBoot = CloneWorkspace(parentWS, OutputWorkspace=parentWS.name()+"_Bootstrap")
        for i, row in enumerate(bootDataY):
            wsBoot.dataY(i)[:-1] = row     # Last column will be ignored in ncp fit anyway

        assert np.all(wsBoot.extractY()[:, :-1] == bootDataY), "Bootstrap data not being correctly passed onto ws."

        bootInputWS.append(wsBoot)
    return bootInputWS


def createJackknifeWS(parentWSNCPSavePaths, j):
    bootInputWS = []
    for (parentWSPath, totNcpWSPath) in parentWSNCPSavePaths:
        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)
        
        #TODO: Need to find a way of cutting a column out a ws
        
        # dataY = parentWS.extractY()[:, :-1]
        # totNcp = totNcpWS.extractY()[:, :]     

        # residuals = dataY - totNcp

        # bootRes = bootstrapResidualsSample(residuals)
        # bootDataY = totNcp + bootRes

        # wsBoot = CloneWorkspace(parentWS, OutputWorkspace=parentWS.name()+"_Bootstrap")
        # for i, row in enumerate(bootDataY):
        #     wsBoot.dataY(i)[:-1] = row     # Last column will be ignored in ncp fit anyway

        # assert np.all(wsBoot.extractY()[:, :-1] == bootDataY), "Bootstrap data not being correctly passed onto ws."

        bootInputWS.append(wsBoot)
    return bootInputWS

#TODO: Figure out how to include this check in the code
def checkResiduals(parentWSnNCP):
    """
    Calculates the self-correlation of residuals for each spectrum.
    When correlation is detected, plots the spectrum.
    """
    for (parentWS, parentNCP) in parentWSnNCP:
        dataY = parentWS.extractY()[:, :-1]
        totNcp = parentNCP.extractY()[:, :]     
        residuals = dataY - totNcp 

        lag = 1     # For lag-plot of self-correlation
        print(f"Correlation Coefficients for lag={lag}:\n")
        for rowRes in residuals:
            corrCoef = stats.pearsonr(rowRes[:-lag], rowRes[lag:])  
            print(corrCoef[0])
            if np.abs(corrCoef[0]) >= 0.5:
                plt.scatter(rowRes[:-lag], rowRes[lag:], label=f"r={corrCoef}")
                plt.legend()
                plt.show()


def loadWorkspacesFromPath(*args):
    wsList = []
    for path in args:
        saveName = path.name.split(".")[0]
        ws = Load(str(path), OutputWorkspace=saveName)
        wsList.append(ws)

    return wsList


def bootstrapResidualsSample(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]
    return bootRes



def plugBootWSIntoIC(inputIC, bootInputWS, fastBootstrap):
    """
    Changes initial conditions to take in the bootstrap workspace.
    If bootstrap is on fast mode, change MS iterations to 1.
    """
    for IC, wsBoot in zip(inputIC, bootInputWS):
        IC.bootSample = True
        IC.bootWS = wsBoot

        if fastBootstrap:
            IC.noOfMSIterations = 1


