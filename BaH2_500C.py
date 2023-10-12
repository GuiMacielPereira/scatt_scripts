import time
import numpy as np
from pathlib import Path
from vesuvio_analysis.core_functions.bootstrap_analysis import runAnalysisOfStoredBootstrap
from vesuvio_analysis.core_functions.run_script import runScript

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs = '38531-38542'      
    empty_runs = '38373-38387'   # 77K         # The numbers of the empty runs to be subtracted
    spectra = '3-134'                          # Spectra to be analysed
    mode = "DoubleDifference"
    ipfile = ipFilesPath / 'ip2018_3.par'  

    subEmptyFromRaw = True         # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 0.9        
    scaleRaw = 1


class LoadVesuvioFrontParameters:
    runs = '38543-38564'        # 100K        # The numbers of the runs to be analysed
    empty_runs = '38373-38387'  # 100K        # The numbers of the empty runs to be subtracted
    spectra = '135-182'                        # Spectra to be analysed
    mode = "SingleDifference" 
    ipfile = ipFilesPath / "ip2018_3.par"

    subEmptyFromRaw = True     # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 0.9       
    scaleRaw = 1


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):

    HToMassIdxRatio = 21  # Set to None when either unknown or H not present
    massIdx = 0

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([16, 27, 28, 93, 137.3])

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers   
            1,  9,   0.,    
            1,  13,  0., 
            1,  13,  0.,   
            1,  20,  0.,
            1,  20,  0.
        ])
    bounds = np.array([
            [0, np.nan], [7, 14], [-3, 1],
            [0, np.nan], [12, 14], [-3, 1],
            [0, np.nan], [10, 20], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1]
        ])
    constraints = ()

    noOfMSIterations = 1     #4
    firstSpec = 3    #3
    lastSpec = 134   #134

    maskedSpecAllNo = np.array([18, 34, 42, 62])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    tofBinning="110,1.,420"                    # Binning of ToF spectra
    maskTOFRange = None     # TOF Range for the resonance peak

    transmission_guess =  0.9        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5


class ForwardInitialConditions(GeneralInitialConditions):

    masses = np.array([1.0079, 16, 27, 28, 93, 137.3]) 

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers  
            10, 5, 0., 
            1,  9,   0.,    
            1,  13,  0., 
            1,  13,  0.,   
            1,  20,  0.,
            1,  20,  0.   
    ])
    bounds = np.array([
            [0, np.nan], [3, 6], [-1.5, 0.5],
            [0, np.nan], [7, 14], [-3, 1],
            [0, np.nan], [12, 14], [-3, 1],
            [0, np.nan], [10, 20], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1]
    ])
    constraints = ()

    noOfMSIterations = 0   #4
    firstSpec = 135   #135
    lastSpec = 147   #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([171, 172, 173, 174])

    tofBinning = "110,1.,420"                 # Binning of ToF spectra
    maskTOFRange ="150, 163"     # TOF Range for the resonance peak

    transmission_guess =  0.9        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5


class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"    # Needs to be symetric
    fitModel = "SINGLE_GAUSSIAN" #"DOUBLE_WELL"   # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6', 'DOUBLE_WELL', 'DOUBLE_WELL_ANSIO'
    runMinos = False
    globalFit = False
    nGlobalFitGroups = 4         # Number or string "ALL"
    maskTypeProcedure = "NAN" #"NAN" # "NCP"    # Options: 'NCP', 'NAN', None    



class UserScriptControls:
    runRoutine = True

    # Choose main procedure to run
    procedure = "FORWARD"   # Options: None, "BACKWARD", "FORWARD", "JOINT"
    # Choose on which ws to perform the fit in y space
    fitInYSpace = "FORWARD"   # Options: None, "BACKWARD", "FORWARD", "JOINT"


class BootstrapInitialConditions:
    runBootstrap = False

    procedure = "FORWARD"
    fitInYSpace = "FORWARD"   
    
    bootstrapType = "BOOT_GAUSS_ERRS"    # Options: "JACKKNIFE", "BOOT_RESIDUALS", "BOOT_GAUSS_ERRS"
    nSamples = 1
    skipMSIterations = False
    userConfirmation = True
    runningTest = False


class BootstrapAnalysis:
    # Flag below controls whether or not analysis is run
    runAnalysis = False

    # Choose whether to filter averages as done in original procedure
    filterAvg = False       # True discards some unreasonable values of widths and intensities
    
    # Flags below control the plots to show
    plotRawWidthsIntensities = True
    plotMeanWidthsIntensities = True
    plotMeansEvolution = False
    plot2DHists = False
    plotYFitHists = True


# Initialize classes and run script below
# Not for useers

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
