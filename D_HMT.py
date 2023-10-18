
import time
import numpy as np
from pathlib import Path
from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.core_functions.bootstrap_analysis import runAnalysisOfStoredBootstrap
from vesuvio_analysis.core_functions.run_script import runScript

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to experiments/sample
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs='36517-36556'           
    empty_runs='34038-34045'           
    spectra='3-134'                    
    mode = 'DoubleDifference'
    ipfile=ipFilesPath / "ip2018_3.par" 


class LoadVesuvioFrontParameters:
    runs='36517-36556'                   
    empty_runs='34038-34045'                 
    spectra='135-182'                       
    mode='SingleDifference'
    ipfile=ipFilesPath / "ip2018_3.par"


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  


class BackwardInitialConditions(GeneralInitialConditions):

    subEmptyFromRaw = True      
    scaleEmpty = 1    
    scaleRaw = 1  

    HToMassIdxRatio = None   # Set to None when H is not present or not known
    massIdx = 0

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([2.015, 12, 14, 27])

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers   
            1, 6, 0.,     
            1, 12, 0.,    
            1, 12, 0.,   
            1, 12.5, 0.    
        ])
    bounds = np.array([
            [0, np.nan], [3.53, 20], [-3, 1],
            [0, np.nan], [8.62, 20], [-3, 1],
            [0, np.nan], [9.31, 20], [-3, 1],
            [0, np.nan], [12.93, 25], [-3, 1]
        ])
    constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] },{'type': 'eq', 'fun': lambda par:  par[3] - 0.7234*par[6] })

    noOfMSIterations = 4     
    firstSpec = 3    #3
    lastSpec = 134    #134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    tofBinning='50,1.,420'           
    maskTOFRange = None        # TOF Range for the resonance peak

    transmission_guess =  0.92      
    multiple_scattering_order, number_of_events = 2, 1.e5


class ForwardInitialConditions(GeneralInitialConditions):

    subEmptyFromRaw = False        
    scaleEmpty = 1      
    scaleRaw = 1

    masses = np.array([2.015, 12, 14, 27]) 
  
    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers  
        0.4569, 6.5532, 0.,     
        0.166, 12.1585, 0.,    
        0.2295, 13.4784, 0.,   
        0.1476, 17.0095, 0. 
    ])
    bounds = np.array([
        [0, np.nan], [5, 8], [-3, 1],
        [0, np.nan], [12.1585, 12.1585], [-3, 1],
        [0, np.nan], [13.4784, 13.4784], [-3, 1],
        [0, np.nan], [17.0095, 17.0095], [-3, 1]
    ])
    constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] },{'type': 'eq', 'fun': lambda par:  par[3] - 0.7234*par[6] })
    
    noOfMSIterations = 4  
    firstSpec = 135   #135
    lastSpec = 182  #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([180])

    tofBinning="110,1.,430"         
    maskTOFRange = None        # TOF Range for the resonance peak
 
    transmission_guess =  0.92      
    multiple_scattering_order, number_of_events = 2, 1.e5


class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = False
    rebinParametersForYSpaceFit = "-25, 0.5, 25"    # Needs to be symetric
    fitModel = "GC_C4_C6"     # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6', 'DOUBLE_WELL', 'ANSIO_GAUSSIAN'
    runMinos = True
    globalFit = False 
    nGlobalFitGroups = 4   
    maskTypeProcedure = None   # Options: 'NCP', 'NAN', None


class UserScriptControls:
    runRoutine = False 

    # Choose main procedure to run
    procedure = "BACKWARD"   # Options: "BACKWARD", "FORWARD", "JOINT"
    # Choose on which ws to perform the fit in y space
    fitInYSpace = "BACKWARD"    # Options: None, "BACKWARD", "FORWARD", "JOINT"


class BootstrapInitialConditions:
    runBootstrap = False

    procedure = "JOINT"
    fitInYSpace = "JOINT" #"FORWARD"

    bootstrapType = "BOOT_RESIDUALS"  # Options: "JACKKNIFE", "BOOT_RESIDUALS", "BOOT_GAUSS_ERRS" 
    nSamples = 600 
    skipMSIterations = False
    runningTest = False
    userConfirmation = True


class BootstrapAnalysis:
    runAnalysis = False

    # Choose whether to filter averages as done in original procedure
    filterAvg = True                 # True discards some unreasonable values of widths and intensities
    
    # Flags below control the plots to show
    plotRawWidthsIntensities = False 
    plotMeanWidthsIntensities = True
    plotMeansEvolution = False 
    plot2DHists = False
    plotYFitHists = True


start_time = time.time()

wsBackIC = LoadVesuvioBackParameters
wsFrontIC = LoadVesuvioFrontParameters  
bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yFitIC = YSpaceFitInitialConditions
bootIC = BootstrapInitialConditions
userCtr = UserScriptControls

runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

analysisIC = BootstrapAnalysis

runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr)
