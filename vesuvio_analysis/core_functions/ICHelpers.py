
from random import sample
from mantid.simpleapi import LoadVesuvio, SaveNexus
from pathlib import Path
import numpy as np
import json
currentPath = Path(__file__).absolute().parent
experimentsPath = currentPath / ".."/ ".." / "experiments"


def completeICFromInputs(IC, scriptName, wsIC):
    """Assigns new methods to the initial conditions class from the inputs of that class"""

    assert IC.lastSpec > IC.firstSpec, "Last spectrum needs to be bigger than first spectrum"
    assert ((IC.lastSpec<135) & (IC.firstSpec<135)) | ((IC.lastSpec>=135) & (IC.firstSpec>=135)), "First and last spec need to be both in Back or Front scattering."

    if IC.lastSpec <= 134:
        IC.modeRunning = "BACKWARD"
    elif IC.firstSpec >= 135:
        IC.modeRunning = "FORWARD"
    else:
        raise ValueError("Invalid first and last spectra input.")

    IC.name = scriptName+"_"+IC.modeRunning+"_"

    IC.masses = IC.masses.astype(float)
    IC.noOfMasses = len(IC.masses)

    IC.maskedSpecNo = IC.maskedSpecAllNo[(IC.maskedSpecAllNo>=IC.firstSpec) & (IC.maskedSpecAllNo<=IC.lastSpec)]
    IC.maskedDetectorIdx = IC.maskedSpecNo - IC.firstSpec

    # Extract some attributes from wsIC
    IC.mode = wsIC.mode
    # IC.subEmptyFromRaw = wsIC.subEmptyFromRaw
    # IC.scaleEmpty = wsIC.scaleEmpty
    # IC.scaleRaw = wsIC.scaleRaw
    
    # When attribute InstrParsPath is not present, set it equal to path from wsIC
    try:    
        r = IC.InstrParsPath    # If present, leave it unaltered
    except AttributeError:
        IC.InstrParsPath = wsIC.ipfile

    # Sort out input and output paths
    rawPath, emptyPath = inputDirsForSample(wsIC, scriptName)

    if (not rawPath.is_file()) or (not emptyPath.is_file()):

        rawPath.parent.mkdir(parents=True, exist_ok=True)
        assert rawPath.parent == emptyPath.parent, "Raw and Empty workspaces not set up to be saved under the same directory"
        print(f"\nWorkspaces not found, will save new workspaces in: {rawPath.parent.name}")

        saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath)
    
    IC.userWsRawPath = rawPath
    IC.userWsEmptyPath = emptyPath

    setOutputDirsForSample(IC, scriptName)
    
    # Do not run bootstrap sample, by default
    IC.runningSampleWS = False

    # Store script name
    IC.scriptName = scriptName

    # Default not running preliminary procedure to estimate HToMass0Ratio
    IC.runningPreliminary = False
    
    # Set directories for figures
    figSavePath = experimentsPath / scriptName /"figures" 
    figSavePath.mkdir(exist_ok=True)
    IC.figSavePath = figSavePath

    # Create default of not running original version with histogram data
    try:
        t = IC.runHistData
    except AttributeError:
        IC.runHistData = False

    # Norm voigt except when comparing with tests
    try:
        d = IC.normVoigt
    except AttributeError:
        IC.normVoigt = True

    return 


def inputDirsForSample(wsIC, sampleName):
    inputWSPath = experimentsPath / sampleName / "input_ws"
    inputWSPath.mkdir(parents=True, exist_ok=True)

    runningMode = identifyRunningMode(wsIC)

    rawWSName, emptyWSName = nameRawEmptyWS(sampleName, runningMode)

    newWSDir = defaultNewWSDirectory(inputWSPath, runningMode)

    rawPath = newWSDir / rawWSName
    emptyPath = newWSDir / emptyWSName

    currLoadWSDict = convertLoadWSICToDict(wsIC)

    for filePath in inputWSPath.rglob('*' + runningMode + '.json'):
        storedDict = json.load(open(filePath))

        if currLoadWSDict == storedDict:   # Ignores order

            storedWSDir = filePath.parent

            rawPath = storedWSDir / rawWSName
            emptyPath = storedWSDir / emptyWSName
            print(f"\nFound {runningMode} workspaces with matching inputs in: {str(storedWSDir.name)}")

    return rawPath, emptyPath


def identifyRunningMode(wsIC) -> str:
    if int(wsIC.spectra.split("-")[1]) < 135:
        runningMode = "backward"
    elif int(wsIC.spectra.split("-")[0]) >= 135:
        runningMode = "forward"
    else:
        raise ValueError("Problem in loading workspaces: invalid range of spectra.")
    return runningMode


def nameRawEmptyWS(sampleName, runningMode):
    rawWSName = sampleName + "_raw_" + runningMode + ".nxs"
    emptyWSName = sampleName + "_empty_" + runningMode + ".nxs"
    return rawWSName, emptyWSName


def defaultNewWSDirectory(inputWSPath, runningMode):
    wsDirs = inputWSPath.glob(f'{runningMode}*/')
    versionNums = [float(dir.name.split('_')[-1]) for dir in wsDirs]
    versionNums = [0.0] if not versionNums else versionNums    # Take care of empty list
    newDirName = runningMode + '_' + str(max(versionNums) + 1)
    newWSDir = inputWSPath / newDirName
    return newWSDir


def setOutputDirsForSample(IC, sampleName):
    outputPath = experimentsPath / sampleName / "output_npz_for_testing"
    outputPath.mkdir(parents=True, exist_ok=True)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_MS"
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"+".npz"
    fileNameYSpace = fileName + "_ySpaceFit"+".npz"

    IC.resultsSavePath = outputPath / fileName
    IC.ySpaceFitSavePath = outputPath / fileNameYSpace
    return


def saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath):
    
    print(f"\nLoading and storing workspace sample runs: {wsIC.runs}\n")

    rawVesuvio = LoadVesuvio(
        Filename=wsIC.runs,
        SpectrumList=wsIC.spectra,
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile),
        OutputWorkspace=rawPath.name
        )

    SaveNexus(rawVesuvio, str(rawPath))
    print(f"\nRaw workspace stored locally under {rawPath.parent.name}\n")

    emptyVesuvio = LoadVesuvio(
        Filename=wsIC.empty_runs,
        SpectrumList=wsIC.spectra,
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile),
        OutputWorkspace=emptyPath.name
        )

    SaveNexus(emptyVesuvio, str(emptyPath))
    print(f"\nRaw workspace stored locally under {emptyPath.parent.name}\n")

    wsLogNameFile = rawPath.name.replace('_raw_', '_').replace('.nxs', '.json')
    saveJsonFile(rawPath.parent, wsLogNameFile, wsIC)
    return


def saveJsonFile(parentDir, fileName, wsIC):
    savePath = parentDir / fileName
    currLoadWSDict = convertLoadWSICToDict(wsIC)
    json.dump(currLoadWSDict, open(savePath, 'w'))
    return


def completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC):
    if not bootIC.runBootstrap:
        return

    try:    # Assume it is not running a test if atribute is not found
        reading = bootIC.runningTest
    except AttributeError:
        bootIC.runningTest = False

    setBootstrapDirs(bckwdIC, fwdIC, bootIC, yFitIC)
    return


def setBootstrapDirs(bckwdIC, fwdIC, bootIC, yFitIC):
    """Form bootstrap output data paths"""

    # Select script name and experiments path
    sampleName = bckwdIC.scriptName   # Name of sample currently running
    experimentsPath = currentPath/".."/".."/"experiments"
    
    # Used to store running times required to estimate Bootstrap total run time.
    bootIC.runTimesPath = experimentsPath / sampleName / "running_times.txt"

    # Make bootstrap and jackknife data directories
    if bootIC.bootstrapType=="JACKKNIFE":
        bootPath = experimentsPath / sampleName / "jackknife_data"
    else:
        bootPath = experimentsPath / sampleName / "bootstrap_data"
    bootPath.mkdir(exist_ok=True)

    # Folders for skipped and unskipped MS
    if bootIC.skipMSIterations:
        dataPath = bootPath / "skip_MS_corrections"
    else:
        dataPath = bootPath / "with_MS_corrections"
    dataPath.mkdir(exist_ok=True)

    # Create text file for logs
    logFilePath = dataPath / "data_files_log.txt"
    if not(logFilePath.is_file()):
        with open(logFilePath, "w") as txtFile:
            txtFile.write(header_string())

    for IC in [bckwdIC, fwdIC]:    # Make save paths for .npz files
        bootName, bootNameYFit = genBootFilesName(IC, bootIC)

        IC.bootSavePath = dataPath / bootName          # works because modeRunning has same strings as procedure
        IC.bootYFitSavePath = dataPath / bootNameYFit

        IC.logFilePath = logFilePath
        IC.bootSavePathLog = logString(bootName, IC, yFitIC, bootIC, isYFit=False)
        IC.bootYFitSavePathLog = logString(bootNameYFit, IC, yFitIC, bootIC, isYFit=True)
    return 


def genBootFilesName (IC, bootIC):
    """Generates save file name for either BACKWARD or FORWARD class"""

    nSamples = bootIC.nSamples
    if bootIC.bootstrapType=="JACKKNIFE": 
        nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_MS"
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"
    bootName = fileName + f"_nsampl_{nSamples}"+".npz"
    bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}"+".npz"
    return bootName, bootNameYFit


def header_string():
    return """
    This file contains some information about each data file in the folder.
    ncp data file: boot type | procedure | tof binning | masked tof range.
    yspace fit data file: boot type | procedure | symmetrisation | rebin pars | fit model | mask type
    """

def logString(bootDataName, IC, yFitIC, bootIC, isYFit):
    if isYFit:
        log = (bootDataName+" : "+bootIC.bootstrapType+
        " | "+str(bootIC.fitInYSpace)+
        " | "+str(yFitIC.symmetrisationFlag)+
        " | "+yFitIC.rebinParametersForYSpaceFit+
        " | "+yFitIC.fitModel+
        " | "+str(yFitIC.maskTypeProcedure))
    else:
        log = (bootDataName+" : "+bootIC.bootstrapType+
        " | "+str(bootIC.procedure)+
        " | "+IC.tofBinning+
        " | "+str(IC.maskTOFRange))
    return log


def noOfHistsFromTOFBinning(IC):
    start, spacing, end = [int(float(s)) for s in IC.tofBinning.split(",")]  # Convert first to float and then to int because of decimal points
    return int((end-start)/spacing) - 1 # To account for last column being ignored


def buildFinalWSName(scriptName: str, procedure: str, IC):
    # Format of corrected ws from last iteration
    name = scriptName + "_" + procedure + "_" + str(IC.noOfMSIterations)
    return name 

def completeYFitIC(yFitIC, sampleName):
    # Set directories for figures

    figSavePath = experimentsPath / sampleName /  "figures" 
    figSavePath.mkdir(exist_ok=True)
    yFitIC.figSavePath = figSavePath
    return

def convertLoadWSICToDict(wsIC):
    load_ws_params = {}
    for attr in ["runs", "empty_runs", "spectra", "mode", "ipfile" ]:
        load_ws_params[attr] = str(getattr(wsIC, attr))      # str -> str, PosixPath -> str
    return load_ws_params

