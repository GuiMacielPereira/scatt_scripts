from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runJointBackAndForwardProcedure, runIndependentIterativeProcedure
from vesuvio_analysis.core_functions.ICHelpers import buildFinalWSName, noOfHistsFromTOFBinning
from mantid.api import AnalysisDataService, mtd
from mantid.simpleapi import CloneWorkspace, SaveNexus, Load, SumSpectra
from scipy import stats
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")
currentPath = Path(__file__).parent.absolute()


def runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC):

    checkValidInput(bootIC)

    checkOutputDirExists(bckwdIC, fwdIC, bootIC)            # Checks to see if those directories exits already
    askUserConfirmation(bckwdIC, fwdIC, bootIC)
    AnalysisDataService.clear()

    if bootIC.bootstrapType=="JACKKNIFE":
        return JackknifeProcedure(bckwdIC, fwdIC, bootIC, yFitIC)
    
    return bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC)


def checkValidInput(bootIC):
    boot = bootIC.bootstrapType
    assert (boot=="JACKKNIFE") | (boot=="BOOT_GAUSS_ERRS") | (boot=="BOOT_RESIDUALS"), \
        "bootstrapType not recognized. Options: 'JACKKNIFE', 'BOOT_GAUSS_ERRS', 'BOOT_RESIDUALS'"


def checkOutputDirExists(bckwdIC, fwdIC, bootIC):
    if bootIC.runningTest:
        return

    proc = bootIC.procedure
    if (proc=="BACKWARD") | (proc=="JOINT"):
        checkOutDirIC(bckwdIC)
    if (proc=="FORWARD") | (proc=="JOINT"):
        checkOutDirIC(fwdIC)
    return 


def checkOutDirIC(IC):
    if IC.bootSavePath.is_file() or IC.bootYFitSavePath.is_file():
        print(f"\nOutput data files were detected:" \
            f"\n{IC.bootSavePath.name}\n{IC.bootYFitSavePath.name}" \
            f"\nAborting Run of Bootstrap to prevent overwriting data." \
            f"\nTo avoid this issue you can change the number of samples to run.")
        raise ValueError("Output data directories already exist. Aborted Bootstrap.")
    return


def JackknifeProcedure(bckwdIC, fwdIC, bootIC, yFitIC):
    assert bootIC.procedure != None

    proc = bootIC.procedure
    if (proc=="FORWARD") | (proc=="BACKWARD"):
        return bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC)

    elif proc=="JOINT":  # Do the Jackknife procedure separately
        
        # Run original procedure to change fwdIC from running backward
        runOriginalBeforeBootstrap(bckwdIC, fwdIC, bootIC, yFitIC) 

        bootIC.procedure="BACKWARD"
        bootIC.fitInYSpace="BACKWARD"
        bckwdJackRes = bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC)

        bootIC.procedure="FORWARD"
        bootIC.fitInYSpace="FORWARD"
        fwdJackRes = bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC)

        return {**bckwdJackRes, **fwdJackRes}    # For consistency
    else: raise ValueError ("Bootstrap procedure not recognized.")


def bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC):
    """
    Main algorithm for the Bootstrap.
    Allows for Jackknife or Bootstrap depending on bool flag set in bootIC.
    Chooses fast or slow (correct) version of bootstrap depending on flag set in bootIC.
    Performs either independent or joint procedure depending of len(inputIC).
    """
    if bootIC.bootstrapType=="JACKKNIFE": assert bootIC.procedure!='JOINT', "'JOINT' mode should not have reached Jackknife here."

    AnalysisDataService.clear()
 
    parentResults, parentWSnNCPs = runOriginalBeforeBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)
    corrCoefs = autoCorrResiduals(parentWSnNCPs)

    nSamples = chooseNSamples(bootIC, parentWSnNCPs)

    bootResults = initializeResults(parentResults, nSamples, corrCoefs)
    saveBootstrapLogs(bootResults, bckwdIC, fwdIC)
    parentWSNCPSavePaths = convertWSToSavePaths(parentWSnNCPs)

    iStart, iEnd = chooseLoopRange(bootIC, nSamples)

    # Form each bootstrap workspace and run ncp fit with MS corrections
    for i in range(iStart, iEnd):
        AnalysisDataService.clear()
        plt.close("all")    # Not sure if previous step clears plt figures, so introduced this step to be safe

        try:
            sampleInputWS, parentWS = createSampleWS(parentWSNCPSavePaths, i, bootIC)   # Creates ith sample
        except JackMaskCol: continue    # If Jackknife column already masked, skip to next column

        formSampleIC(bckwdIC, fwdIC, bootIC, sampleInputWS, parentWS)  
        try:
            iterResults = runMainProcedure(bckwdIC, fwdIC, bootIC, yFitIC)   # Conversion to YSpace with masked column
        except AssertionError: continue     # If the procedure fails, skip to next iteration
        
        storeBootIter(bootResults, i, iterResults)   # Stores results for each iteration
        saveBootstrapResults(bootResults, bckwdIC, fwdIC)
    return bootResults


def askUserConfirmation(bckwdIC, fwdIC, bootIC):
    """Estimates running time for all samples and asks the user to confirm the run."""
    
    if not(bootIC.userConfirmation):   # Skip user confirmation 
        return

    tDict = storeRunnningTime(fwdIC, bckwdIC, bootIC)   # Run times file path stores in bootIC

    proc = bootIC.procedure
    runTime = 0
    if (proc=="BACKWARD") | (proc=="JOINT"):
        runTime += calcRunTime(bckwdIC, tDict["tBackNoMS"], tDict["tBackPerMS"], bootIC)

    if (proc=="FORWARD") | (proc=="JOINT"):
        runTime += calcRunTime(fwdIC, tDict["tFowNoMS"], tDict["tFowPerMS"], bootIC)

    userInput = input(f"\n\nEstimated time for Bootstrap procedure: {runTime/60:.1f} hours.\nProceed? (y/n): ")
    if (userInput == "y") or (userInput == "Y"):
        return
    else:
        raise KeyboardInterrupt ("Bootstrap procedure interrupted.")


def storeRunnningTime(fwdIC, bckwdIC, bootIC):
    """Used to write run times to txt file."""

    savePath = bootIC.runTimesPath

    if not(savePath.is_file()):
        with open(savePath, "w") as txtFile:
            txtFile.write("This file stores run times to estimate Bootstrap total run time.")
            txtFile.write("\nTime in minutes.\n\n")
    
    resDict = {}
    with open(savePath, "r") as txtFile:
        for line in txtFile:
            if line[0]=="{":   # If line contains dictionary
                resDict = eval(line)

    if len(resDict)<4:
        ans = input("Did not find necessary information to estimate runtime. Will run a short routine to store an estimate. Please wait until this is finished. Press any key to continue.")
        resDict = buildRunTimes(fwdIC, bckwdIC)

        with open(savePath, "a") as txtFile:
            print(resDict, file=txtFile)

    return resDict


def buildRunTimes(fwdIC, bckwdIC):
    resDict = {}
    for IC, mode in zip([bckwdIC, fwdIC], ["Back", "Fow"]):
        oriMS = IC.noOfMSIterations
        for NIter, key in zip([0, 1], ["NoMS", "PerMS"]):
            IC.noOfMSIterations = NIter
            t0 = time.time()
            runIndependentIterativeProcedure(IC)
            t1 = time.time()
            resDict["t"+mode+key] = (t1-t0) / 60
        # Restore starting value
        IC.noOfMSIterations = oriMS

        # Correct times of only MS by subtacting time spend on fitting ncps
        resDict["t"+mode+"PerMS"] -= 2 * resDict["t"+mode+"NoMS"]   

    return resDict


def calcRunTime(IC, tNoMS, tPerMS, bootIC):
    if bootIC.skipMSIterations:
        timePerSample = tNoMS
    else:
        timePerSample = tNoMS + (IC.noOfMSIterations) * (tNoMS+tPerMS)

    nSamples = bootIC.nSamples 
    if bootIC.bootstrapType=="JACKKNIFE":
        nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

    return  nSamples * timePerSample
    

def chooseLoopRange(bootIC, nSamples):
    iStart = 0
    iEnd = nSamples
    if bootIC.bootstrapType=="JACKKNIFE" and bootIC.runningTest:
        iStart = int(nSamples/2)
        iEnd = iStart + 3   
    return iStart, iEnd


def runOriginalBeforeBootstrap(bckwdIC, fwdIC, bootIC, yFitIC):
    """Runs unaltered procedure to store parent results and select parent ws"""

    setICsToDefault(bckwdIC, fwdIC, yFitIC)
    parentResults = runMainProcedure(bckwdIC, fwdIC, bootIC, yFitIC)
    parentWSnNCPs = selectParentWorkspaces(bckwdIC, fwdIC, bootIC)

    return parentResults, parentWSnNCPs


def chooseNSamples(bootIC, parentWSnNCPs: dict):
    """
    Returns number of samples to run.
    If Jackknife is running, no of samples is the number of bins in the workspace."""

    nSamples = bootIC.nSamples
    if bootIC.bootstrapType=="JACKKNIFE":
        assert len(parentWSnNCPs) == 2, "Running Jackknife, supports only one IC at a time."
        if bootIC.procedure=="FORWARD": key = "fwdNCP"
        elif bootIC.procedure=="BACKWARD": key = "bckwdNCP"

        nSamples = parentWSnNCPs[key].blocksize()   # Number of cols from ncp workspace, accounts for missing last col or not
    return nSamples


def setICsToDefault(bckwdIC, fwdIC, yFitIC):
    """Disables some features of yspace fit, makes sure the default """

    # Disable Minos
    if yFitIC.runMinos: yFitIC.runMinos = False

    # Disable global fit 
    if yFitIC.globalFit: yFitIC.globalFit = False

    # Don't show plots
    if yFitIC.showPlots: yFitIC.showPlots = False

    if bckwdIC.runningSampleWS: bckwdIC.runningSampleWS = False
    if fwdIC.runningSampleWS: fwdIC.runningSampleWS = False
    return


def runMainProcedure(bckwdIC, fwdIC, bootIC, yFitIC):
    """Decides main procedure to run based on the initial conditions offered as inputs."""

    resultsDict = {}

    if (bootIC.procedure=="FORWARD") | (bootIC.procedure=="BACKWARD"):

        for mode, IC, key in zip(["FORWARD", "BACKWARD"], [fwdIC, bckwdIC], ["fwd", "bckwd"]):

            if bootIC.procedure==mode:
                wsFinal, bckwdScatRes = runIndependentIterativeProcedure(IC, clearWS=False)
                resultsDict[key+"Scat"] = bckwdScatRes

                if bootIC.bootstrapType=="JACKKNIFE": yFitIC.maskTypeProcedure = 'NAN'  # Enable NAN averaging in y-space fit

                bckwdYFitRes = fitInYSpaceProcedure(yFitIC, IC, wsFinal)
                resultsDict[key+"YFit"] = bckwdYFitRes

    
    elif bootIC.procedure=="JOINT":

        ws, bckwdScatRes, fwdScatRes = runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=False)
        resultsDict["bckwdScat"] = bckwdScatRes
        resultsDict["fwdScat"] = fwdScatRes

        for mode, IC, key in zip(["FORWARD", "BACKWARD"], [fwdIC, bckwdIC], ["fwd", "bckwd"]):

            if (bootIC.fitInYSpace==mode) | (bootIC.fitInYSpace=="JOINT"):
                wsName = buildFinalWSName(IC.scriptName, mode, IC)  
                fwdYFitRes = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
                resultsDict[key+"YFit"] = fwdYFitRes
    else:
        raise ValueError("Bootstrap procedure not recognized.")

    return resultsDict


def selectParentWorkspaces(bckwdIC, fwdIC, bootIC):
    """
    Selects parent workspace from which the Bootstrap replicas will be created.
    If fast mode, the parent ws is the final ws after MS corrections.
    """
    parentWSnNCPsDict = {}
    
    for mode, IC, key in zip(["FORWARD", "BACKWARD"], [fwdIC, bckwdIC], ["fwd", "bckwd"]):

        if (bootIC.procedure==mode) | (bootIC.procedure=="JOINT"):

            wsIter = str(IC.noOfMSIterations) if bootIC.skipMSIterations else "0"   # In case of skipping MS, select very last corrected ws
            
            parentWS = mtd[IC.name+wsIter]
            parentNCP = mtd[parentWS.name()+"_TOF_Fitted_Profiles"]

            parentWSnNCPsDict[key+"WS"] = parentWS
            parentWSnNCPsDict[key+"NCP"] = parentNCP

    return parentWSnNCPsDict 


def autoCorrResiduals(parentWSnNCP: dict):
    """
    Calculates the self-correlation of residuals for each spectrum.
    """
    corrCoefs = {}
    for mode in ["bckwd", "fwd"]:

        try:    # Look for workspaces in dictionary, skip if not present
            parentWS = parentWSnNCP[mode+"WS"]
            parentNCP = parentWSnNCP[mode+"NCP"]
        except KeyError: continue

        totNcp = parentNCP.extractY()[:, :]     
        dataY = parentWS.extractY()[:, :totNcp.shape[1]]    # Missing last column or not
        residuals = dataY - totNcp 

        lag = 1     # For lag-plot of self-correlation
        corr = np.zeros((len(residuals), 2))
        for i, rowRes in enumerate(residuals):
            corr[i] = stats.pearsonr(rowRes[:-lag], rowRes[lag:]) 

        corrCoefs[mode+"Scat"] = corr
    return corrCoefs
    

def initializeResults(parentResults: dict, nSamples, corrCoefs):
    """
    Initializes a list with objects to store output data.
    [BootBackResults, BootFrontResults, BootYSpaceResults]
    """
    bootResultObjs = {}

    for key in ["fwd", "bckwd"]:

        if key+"Scat" in parentResults:
            bootResultObjs[key+"Scat"] = BootScattResults(parentResults[key+"Scat"], nSamples, corrCoefs[key+"Scat"])

        if key+"YFit" in parentResults:
            bootResultObjs[key+"YFit"] = BootYFitResults(parentResults[key+"YFit"], nSamples)
    return bootResultObjs


class BootScattResults:

    def __init__(self, parentResults, nSamples, corr):
        self.parentResult = parentResults.all_spec_best_par_chi_nit[-1]
        self.bootSamples = np.full((nSamples, *self.parentResult.shape), np.nan)
        self.corrResiduals = corr

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.all_spec_best_par_chi_nit[-1]
    
    def saveResults(self, IC):
        np.savez(IC.bootSavePath, boot_samples=self.bootSamples,
             parent_result=self.parentResult, corr_residuals=self.corrResiduals)

    def saveLog(self, IC):
        with open(IC.logFilePath, "a") as logFile:
            logFile.write("\n"+IC.bootSavePathLog)


class BootYFitResults:

    def __init__(self, parentResults, nSamples):
        self.parentPopt = parentResults.popt
        self.parentPerr = parentResults.perr
        self.bootSamples = np.full((nSamples, *self.parentPopt.shape), np.nan)

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.popt
    
    def saveResults(self, IC):
        np.savez(IC.bootYFitSavePath, boot_samples=self.bootSamples,
             parent_popt=self.parentPopt, parent_perr=self.parentPerr)    

    def saveLog(self, IC):
        with open(IC.logFilePath, "a") as logFile:
            logFile.write("\n"+IC.bootYFitSavePathLog)


def storeBootIter(bootResultObjs: dict, j: int, bootIterResults: dict):
    for key in bootResultObjs:
        bootResultObjs[key].storeBootIterResults(j, bootIterResults[key])
    return


def saveBootstrapResults(bootResultObjs: dict, bckwdIC, fwdIC):
    for key, IC in zip(["bckwd", "fwd"], [bckwdIC, fwdIC]):
        for res in ["Scat", "YFit"]:
            if key+res in bootResultObjs:
                bootResultObjs[key+res].saveResults(IC)
    return


def saveBootstrapLogs(bootResultObjs: dict, bckwdIC, fwdIC):
    for key, IC in zip(["bckwd", "fwd"], [bckwdIC, fwdIC]):
        for res in ["Scat", "YFit"]:
            if key+res in bootResultObjs:
                bootResultObjs[key+res].saveLog(IC)
    return


def convertWSToSavePaths(parentWSnNCPs: dict):
    savePaths = {}
    for key in parentWSnNCPs:
        savePaths[key] = saveWorkspacesLocally(parentWSnNCPs[key])
    return savePaths


def saveWorkspacesLocally(ws):
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
    return savePath 


def createSampleWS(parentWSNCPSavePaths: dict, j: int, bootIC):

    boot = bootIC.bootstrapType
    if boot=="JACKKNIFE":
        return createJackknifeWS(parentWSNCPSavePaths, j)
    elif boot=="BOOT_RESIDUALS":
        return createBootstrapWS(parentWSNCPSavePaths)
    elif boot=="BOOT_GAUSS_ERRS":
        return createBootstrapWS(parentWSNCPSavePaths, drawGauss=True)


def createBootstrapWS(parentWSNCPSavePaths:dict, drawGauss=False):
    """
    Creates bootstrap ws replica.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit
    """

    bootInputWS = {}
    parentInputWS = {} 
    for key in ["bckwd", "fwd"]:
        try:
            parentWSPath = parentWSNCPSavePaths[key+"WS"]
            totNcpWSPath = parentWSNCPSavePaths[key+"NCP"]
        except KeyError: continue

        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)

        totNcp = totNcpWS.extractY()[:, :]     
        dataY = parentWS.extractY()[:, :totNcp.shape[1]]    # Missing last col or not
        dataE = parentWS.extractE()[:, :totNcp.shape[1]]

        # Filter out masked columns
        maskCols = np.all(dataY==0, axis=0)
        dataY, totNcp, dataE = dataY[:, ~maskCols], totNcp[:, ~maskCols], dataE[:, ~maskCols]

        # Draw DataY from Gaussian distribution
        if drawGauss:
            # TODO: This is actualy the implementation of parametric bootstrap!
            # The correct implementation is dataY + normal(0, 1) * dataE
            # Is this the same thing?
            bootDataY = np.random.normal(dataY, dataE)  # Mean at dataY, width dataE

        else:   # Default, resample residuals
            residuals = dataY - totNcp
            bootRes = bootstrapResidualsSample(residuals)
            bootDataY = totNcp + bootRes

        # Add masked columns as in parent workspace
        fullBootDataY = np.zeros((len(bootDataY), len(maskCols)))
        fullBootDataY[:, ~maskCols] = bootDataY     # Set non-masked values

        # Pass dataY onto workspace
        wsBoot = CloneWorkspace(parentWS, OutputWorkspace=parentWS.name()+"_Bootstrap")
        for i, row in enumerate(fullBootDataY):
            wsBoot.dataY(i)[:len(row)] = row     # Last column will be ignored or not
            if drawGauss:
                wsBoot.dataE(i)[:] = np.zeros(wsBoot.readE(i).size)

        assert ~np.all(wsBoot.extractY() == parentWS.extractY()), "Bootstrap data not being correctly passed onto ws."

        bootInputWS[key+"WS"] = wsBoot
        parentInputWS[key+"WS"] = parentWS
        parentInputWS[key+"NCP"] = totNcpWS
    return bootInputWS, parentInputWS


def bootstrapResidualsSample(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]
    return bootRes


def createJackknifeWS(parentWSNCPSavePaths: list, j: int):
    """
    Creates jackknife ws replicas.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit
    """

    jackInputWS = {}
    parentInputWS = {} 
    # Jackknife does not have 'JOINT' option
    # Careful with this step if in future Jackknife allows for 'JOINT' internally
    assert len(parentWSNCPSavePaths)==2, "Jackknife can only allow either forward or backward at a time."
    for key in ["bckwd", "fwd"]:    # Only one iteration is selected at a time
        try:
            parentWSPath = parentWSNCPSavePaths[key+"WS"]
            totNcpWSPath = parentWSNCPSavePaths[key+"NCP"]
        except KeyError: continue

        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)

        dataY = parentWS.extractY()

        jackDataY = dataY.copy()

        # Skip Jackknife procedure on columns that are already masked
        if np.all(jackDataY[:, j]==0): raise JackMaskCol

        jackDataY[:, j] = 0   # Masks j collumn with zeros
        # DataE is not masked intentionally, to preserve errors that are used in the normalization of averaged NaN profile
        
        wsJack = CloneWorkspace(parentWS, OutputWorkspace=parentWS.name()+"_Jackknife")
        for i, yRow in enumerate(jackDataY):
            wsJack.dataY(i)[:] = yRow     # Last column will be ignored in ncp fit anyway

        assert np.all(wsJack.extractY() == jackDataY), "Bootstrap data not being correctly passed onto ws."

        jackInputWS[key+"WS"] = wsJack
        parentInputWS[key+"WS"] = parentWS
        parentInputWS[key+"NCP"] = totNcpWS
    return jackInputWS, parentInputWS


class JackMaskCol(Exception):
    """
    Custom exception used only to flag and skip a Jackknife iteration
    for a column that is already masked.
    """
    pass


def loadWorkspacesFromPath(*savePaths):
    wsList = []
    for path in savePaths:
        saveName = path.name.split(".")[0]
        ws = Load(str(path), OutputWorkspace=saveName)
        SumSpectra(ws, OutputWorkspace=ws.name()+"_Sum")
        wsList.append(ws)

    return wsList


def formSampleIC(bckwdIC, fwdIC, bootIC, sampleInputWS:dict, parentWS:dict):
    """Adds atributes to initial conditions to start procedure with sample ws."""

    for mode, IC, key in zip(["FORWARD", "BACKWARD"], [fwdIC, bckwdIC], ["fwd", "bckwd"]):

        if (bootIC.procedure==mode) | (bootIC.procedure=="JOINT"):
            IC.runningSampleWS = True

            if bootIC.skipMSIterations: 
                IC.noOfMSIterations = 0

            IC.sampleWS = sampleInputWS[key+"WS"]
            IC.parentWS = parentWS[key+"WS"]

