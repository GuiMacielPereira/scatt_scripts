
import time
import numpy as np
from pathlib import Path
from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.core_functions.bootstrap_analysis import runAnalysisOfStoredBootstrap
from vesuvio_analysis.core_functions.run_script import runScript

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs = "43066-43076"             # The numbers of the runs to be analysed
    empty_runs = "41876-41923"       # The numbers of the empty runs to be subtracted
    spectra = '3-134'                # Spectra to be analysed
    mode = 'DoubleDifference'           
    ipfile = ipFilesPath / "ip2019.par"    # Name of ip file in ip_files folder

    subEmptyFromRaw = True    # Subtracts Empty WS from Raw WS 
    scaleEmpty = 1       # Scaling factor 
    scaleRaw = 1         # Scaling factor

class LoadVesuvioFrontParameters:    # Same as previous class but for forward ws
    runs = '43066-43076'         
    empty_runs = '43868-43911'   
    spectra = '144-182'         
    mode = 'SingleDifference'  
    ipfile = ipFilesPath / "ip2018_3.par"

    subEmptyFromRaw = False     
    scaleEmpty = 1      
    scaleRaw = 1

class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001     # Sample slab parameters, expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):

    # Ratio of H peak to chosen mass
    HToMassIdxRatio = 19.0620008206   # Set to None either when H not present or ratio not known 
    massIdx = 0   # Idx of mass to take the ratio with, idx is relative to backward scattering masses

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers   
            1, 12, 0.,    
            1, 12, 0.,   
            1, 12.5, 0.    
        ])
    bounds = np.array([
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [11, 14], [-3, 1]
        ])
    constraints = ()

    noOfMSIterations = 1     # Number of MS corrections, 0 is no correction
    firstSpec = 3    #3
    lastSpec = 134   #134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    tofBinning='275.,1.,420'   
    maskTOFRange = None              # TOF Range for the resonance peak

    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5    # Used in MS correction


class ForwardInitialConditions(GeneralInitialConditions):    # Same structure as above

    masses = np.array([1.0079, 12, 16, 27]) 

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers  
        0.902, 4.7, 0, 
        0.047, 14.594, 0.,    
        0.020, 8.841, 0.,   
        0.031, 13.896, 0.    
    ])
    bounds = np.array([
        [0, np.nan], [3, 6], [-3, 1],
        [0, np.nan], [14.594, 14.594], [-3, 1],
        [0, np.nan], [8.841, 8.841], [-3, 1],
        [0, np.nan], [13.896, 13.896], [-3, 1]
    ])
    constraints = ()

    noOfMSIterations = 0      
    firstSpec = 144   #144
    lastSpec = 182   #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    tofBinning="110,1,430"       
    maskTOFRange = None              # TOF Range for the resonance peak
 
    transmission_guess = 0.742   # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 1, 1.e5    # Used in MS correction

class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"    # Needs to be symetric
    fitModel = "MULTIVARIATE_GAUSSIAN"     # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6', 'DOUBLE_WELL', 'ANSIO_GAUSSIAN'
    runMinos = True
    globalFit = True                 # Performs global fit with Minuit by default
    nGlobalFitGroups = 4             # Number or string "ALL"
    maskTypeProcedure = "NAN"         # Options: 'NCP', 'NAN', None


class UserScriptControls:
    runRoutine = True
    
    # Choose main procedure to run
    procedure = "FORWARD"  # Options: None, "BACKWARD", "FORWARD", "JOINT"
    # Choose on which ws to perform the fit in y space
    fitInYSpace = "FORWARD"    # Options: None, "BACKWARD", "FORWARD", "JOINT"


class BootstrapInitialConditions:
    runBootstrap = False

    procedure = "BACKWARD"
    fitInYSpace = None #"FORWARD"

    bootstrapType = "BOOT_RESIDUALS"  # Options: "JACKKNIFE", "BOOT_RESIDUALS", "BOOT_GAUSS_ERRS" 
    nSamples = 650                  # Used if running Bootstrap, otherwise code ignores it
    skipMSIterations = False        # Each replica runs with no MS or Gamma corrections
    userConfirmation = True         # Asks user to confirm procedure, will probably be deleted in the future


class BootstrapAnalysis:
    runAnalysis = False    # Controls whether or not analysis is run

    # Choose whether to filter averages as done in original procedure
    filterAvg = True      # True discards some unreasonable values of widths and intensities
    
    # Flags below control the plots to show
    plotRawWidthsIntensities = False
    plotMeanWidthsIntensities = True
    plotMeansEvolution = False
    plot2DHists = False
    plotYFitHists = True


# Initialize classes and run script below
#  ------------- Not for users ----------------

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
