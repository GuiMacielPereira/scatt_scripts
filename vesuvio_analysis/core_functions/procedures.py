
from .analysis_functions import iterativeFitForDataReduction, switchFirstTwoAxis
from mantid.api import AnalysisDataService, mtd
import numpy as np


def runIndependentIterativeProcedure(IC, clearWS=True):
    """
    Runs the iterative fitting of NCP, cleaning any previously stored workspaces.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays
    """

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()
        
    wsFinal, ncpFitResultsObject = iterativeFitForDataReduction(IC)
    return wsFinal, ncpFitResultsObject


def runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=True):
    assert bckwdIC.modeRunning == "BACKWARD", "Missing backward IC, args usage: (bckwdIC, fwdIC)"
    assert fwdIC.modeRunning == "FORWARD", "Missing forward IC, args usage: (bckwdIC, fwdIC)"

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()

    if isHPresent(fwdIC.masses) and (bckwdIC.HToMass0Ratio==None):
        wsFinal, bckwdScatResults, fwdScatResults = runHPresentAndHRatioNotKnown(bckwdIC, fwdIC)

    else:
        assert (isHPresent(fwdIC.masses) != (bckwdIC.HToMass0Ratio==None)), "When H is not present, HToMass0Ratio has to be set to None"
        
        wsFinal, bckwdScatResults, fwdScatResults = runJoint(bckwdIC, fwdIC)

    return wsFinal, bckwdScatResults, fwdScatResults


def runHPresentAndHRatioNotKnown(bckwdIC, fwdIC):
    """
    Used when H is present and H to first mass ratio is not known.
    Preliminary forward scattering is run to get rough estimate of H to first mass ratio.
    Runs iterative procedure with alternating back and forward scattering.
    """

    assert bckwdIC.runningSampleWS == False, "Procedure not suitable for Bootstrap."

    nIter = askUserNoOfIterations()
 
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    for i in range(int(nIter)):    # Loop until convergence is achieved

        # ---- New addition -----
        bckwdIC.HRatioMassIdx = np.argmax(fwdScatResults.all_mean_intensities[-1, 1:]) # Idx will be already corrected for back masses
        print(f"\n\n---------\nIdx for H ratio: {bckwdIC.HRatioMassIdx}\n-------\n\n")
        #----- End of addition -------

        AnalysisDataService.clear()    # Clears all Workspaces

        bckwdIC.HToMass0Ratio = calculateHToMass0Ratio(fwdScatResults, bckwdIC.HRatioMassIdx)
        wsFinal, bckwdScatResults, fwdScatResults = runJoint(bckwdIC, fwdIC)

    print(f"\n\n---------\nIdx for H ratio: {bckwdIC.HRatioMassIdx}\n-------\n\n")
    print(f"\n\nFinal estimate for HToMass0Ratio: {calculateHToMass0Ratio(fwdScatResults, bckwdIC.HRatioMassIdx):.3f}\n")
    return wsFinal, bckwdScatResults, fwdScatResults


def askUserNoOfIterations():
    print("\nH was detected but HToMass0Ratio was not provided.")
    print("\nSugested preliminary procedure:\n\nrun_forward\nfor n:\n    estimate_HToMass0Ratio\n    run_backward\n    run_forward")
    userInput = input("\n\nDo you wish to run preliminary procedure to estimate HToMass0Ratio? (y/n)") 
    if not((userInput=="y") or (userInput=="Y")): raise KeyboardInterrupt("Preliminary procedure interrupted.")
    
    nIter = int(input("\nHow many iterations do you wish to run? n="))
    return nIter
 

def calculateHToMass0Ratio(fwdScatResults, HRatioMassIdx):
    fwdMeanIntensityRatios = fwdScatResults.all_mean_intensities[-1] 
    # return fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]
    return fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[HRatioMassIdx+1]


def runJoint(bckwdIC, fwdIC):
    wsFinal, bckwdScatResults = iterativeFitForDataReduction(bckwdIC)
    setInitFwdParsFromBackResults(bckwdScatResults, bckwdIC.HToMass0Ratio, fwdIC, bckwdIC.HRatioMassIdx)
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, bckwdScatResults, fwdScatResults   


def setInitFwdParsFromBackResults(bckwdScatResults, HToMass0Ratio, fwdIC, HRatioMassIdx):
    """
    Used to pass mean widths and intensities from back scattering onto intial conditions of forward scattering.
    Checks if H is present and adjust the passing accordingly:
    If H present, use HToMass0Ratio to recalculate intensities and fix only non-H widths.
    If H not present, widths and intensities are directly mapped and all widhts except first are fixed. 
    """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = bckwdScatResults.all_mean_widths[-1]
    backMeanIntensityRatios = bckwdScatResults.all_mean_intensities[-1] 

    if isHPresent(fwdIC.masses):

        assert len(backMeanWidths) == fwdIC.noOfMasses-1, "H Mass present, no of masses in front needs to be bigger than back by 1."

        # Use H ratio to calculate intensity ratios 
        # HIntensity = HToMass0Ratio * backMeanIntensityRatios[0]
        HIntensity = HToMass0Ratio * backMeanIntensityRatios[HRatioMassIdx]
        initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
        initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)

        # Set calculated intensity ratios to forward scattering 
        fwdIC.initPars[0::3] = initialFwdIntensityRatios
        # Set forward widths from backscattering
        fwdIC.initPars[4::3] = backMeanWidths
        # Fix all widths except for H, i.e. the first one
        fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))

    else:   # H mass not present anywhere

        assert len(backMeanWidths) == fwdIC.noOfMasses, "H Mass not present, no of masses needs to be the same for front and back scattering."

        # Set widths and intensity ratios
        fwdIC.initPars[1::3] = backMeanWidths       
        fwdIC.initPars[0::3] = backMeanIntensityRatios  

        if len(backMeanWidths) > 1:           # In the case of single mass, width is not fixed
            # Fix all widhts except first
            fwdIC.bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1,2))   

    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")
    return


def isHPresent(masses) -> bool:

    Hmask = np.abs(masses-1)/1 < 0.1        # H mass whithin 10% of 1 au

    if np.any(Hmask):    # H present

        print("\nH mass detected.\n")
        assert len(Hmask) > 1, "When H is only mass present, run independent forward procedure, not joint."
        assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
        assert sum(Hmask) == 1, "More than one mass very close to H were detected."
        return True
    else:
        return False


