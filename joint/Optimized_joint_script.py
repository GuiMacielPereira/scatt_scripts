import multiprocessing
import numpy as np
#import mantid
from mantid.simpleapi import *
from scipy import optimize
from scipy import ndimage
import time
from pathlib import Path
import multiprocessing as mp
from multiprocessing import freeze_support
from functools import partial
from concurrent.futures import ProcessPoolExecutor 

# Format print output of arrays
np.set_printoptions(suppress=True, precision=4, linewidth=150)
repoPath = Path(__file__).absolute().parent  # Path to the repository


'''
The InitialConditions class allows the user to change the starting parameters.
The script includes everything required for forward and backscattering.
The user can choose to perform back or forward scattering individually or to connect the mean widths from 
backscattering onto the initial parameters for forward scattering.

The fit procedure in the time-of-flight domain is  based on the scipy.minimize.optimize() tool,
used with the SLSQP minimizer, that can handle both boundaries and constraints for fitting parameters.
'''


class InitialConditions:

    # Initialize object to use self methods
    def __init__(self):
        return None

    # Masked detectors for front and back scattering 
    maskedSpecNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133, 173, 174, 179])

    # Multiscaterring Correction Parameters
    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters
  
    # Choose type of scattering, when both True, the mean widths from back are used in ic of front
    backScatteringProcedure = False
    forwardScatteringProcedure = True

    # Paths to save results for back and forward scattering
    pathForTesting = repoPath / "tests" / "cleaning"  
    forwardScatteringSavePath = pathForTesting / "current_forward.npz" 
    backScatteringSavePath = pathForTesting / "current_backward.npz"

    withMultiprocessing = False

    def setBackscatteringInitialConditions(self):
        # Parameters to load Raw and Empty Workspaces
        self.userWsRawPath = r"./input_ws/starch_80_RD_raw_backward.nxs"
        self.userWsEmptyPath = r"./input_ws/starch_80_RD_empty_backward.nxs"

        self.name = "starch_80_RD_backward_"
        self.runs='43066-43076'  # 77K             # The numbers of the runs to be analysed
        self.empty_runs='41876-41923'   # 77K             # The numbers of the empty runs to be subtracted
        self.spectra='3-134'                            # Spectra to be analysed
        self.tof_binning='275.,1.,420'                    # Binning of ToF spectra
        self.mode='DoubleDifference'
        self.ipfile=r'./ip2019.par' 

        # Masses, instrument parameters and initial fitting parameters
        self.masses = np.array([12, 16, 27])
        self.noOfMasses = len(self.masses)
        # Hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5
        self.hydrogen_to_mass0_ratio = 19.0620008206
        self.InstrParsPath = repoPath / 'ip2018_3.par'

        self.initPars = np.array([ 
        # Intensities, NCP widths, NCP centers   
            1, 12, 0.,    
            1, 12, 0.,   
            1, 12.5, 0.    
        ])
        self.bounds = np.array([
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [11, 14], [-3, 1]
        ])
        self.constraints = ()

        self.noOfMSIterations = 1     #4
        self.firstSpec = 3    #3
        self.lastSpec = 134    #134

        # Boolean Flags to control script
        self.loadWsFromUserPathFlag = True
        self.scaleParsFlag = False
        self.MSCorrectionFlag = True       # Original True
        self.GammaCorrectionFlag = False

        self.firstSpecIdx = 0
        self.lastSpecIdx = self.lastSpec - self.firstSpec

        # Consider only the masked spectra between first and last spectrum
        self.maskedSpecNo = self.maskedSpecNo[
            (self.maskedSpecNo >= self.firstSpec) & (self.maskedSpecNo <= self.lastSpec)
        ]
        self.maskedDetectorIdx = self.maskedSpecNo - self.firstSpec

        # Set scaling factors for the fitting parameters, default is ones
        self.scalingFactors = np.ones(self.initPars.shape)
        if self.scaleParsFlag:        # Scale fitting parameters using initial values
                self.initPars[2::3] = np.ones((1, self.noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
                self.scalingFactors = 1 / self.initPars



    def setForwardScatteringInitialConditions(self):
        self.userWsRawPath = r"./input_ws/starch_80_RD_raw_forward.nxs"
        self.userWsEmptyPath = r"./input_ws/starch_80_RD_raw_forward.nxs"

        self.name = "starch_80_RD_forward_"
        self.runs='43066-43076'         # 100K             # The numbers of the runs to be analysed
        self.empty_runs='43868-43911'   # 100K             # The numbers of the empty runs to be subtracted
        self.spectra='144-182'                               # Spectra to be analysed
        self.tof_binning="110,1.,430"                    # Binning of ToF spectra
        self.mode='SingleDifference'
        self.ipfile=r'./ip2018_3.par'

        self.masses = np.array([1.0079, 12, 16, 27]) # Changed this recently from shape (4, 1, 1)
        self.noOfMasses = len(self.masses)
        self.hydrogen_to_mass0_ratio = 0
        self.InstrParsPath = repoPath / 'ip2018_3.par'

        self.initPars = np.array([ 
        # Intensities, NCP widths, NCP centers  
            1, 4.7, 0, 
            1, 12.71, 0.,    
            1, 8.76, 0.,   
            1, 13.897, 0.    
        ])
        self.bounds = np.array([
            [0, np.nan], [3, 6], [-3, 1],
            [0, np.nan], [12.71, 12.71], [-3, 1],
            [0, np.nan], [8.76, 8.76], [-3, 1],
            [0, np.nan], [13.897, 13.897], [-3, 1]
        ])
        self.constraints = ()

        self.noOfMSIterations = 1     #4
        self.firstSpec = 144   #144
        self.lastSpec = 182    #182

        # Boolean Flags to control script
        self.loadWsFromUserPathFlag = True
        self.scaleParsFlag = False
        self.MSCorrectionFlag = True         # Original True
        self.GammaCorrectionFlag = True        # Original True

        # Parameters to control fit in Y-Space
        self.symmetriseHProfileUsingAveragesFlag = False
        self.useScipyCurveFitToHProfileFlag = False
        self.rebinParametersForYSpaceFit = "-20, 0.5, 20"
        self.singleGaussFitToHProfile = True 

        self.firstSpecIdx = 0
        self.lastSpecIdx = self.lastSpec - self.firstSpec

        # Consider only the masked spectra between first and last spectrum
        self.maskedSpecNo = self.maskedSpecNo[
            (self.maskedSpecNo >= self.firstSpec) & (self.maskedSpecNo <= self.lastSpec)
        ]
        self.maskedDetectorIdx = self.maskedSpecNo - self.firstSpec

        # Set scaling factors for the fitting parameters, default is ones
        self.scalingFactors = np.ones(self.initPars.shape)
        if self.scaleParsFlag:        # Scale fitting parameters using initial values
                self.initPars[2::3] = np.ones((1, self.noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
                self.scalingFactors = 1 / self.initPars



# This is the only variable with global behaviour, all functions are defined to use attributes of ic
ic = InitialConditions() 


def main():
    if ic.backScatteringProcedure:
        ic.setBackscatteringInitialConditions()
        wsFinal, backScatteringResults = iterativeFitForDataReduction()
        backScatteringResults.save(ic.backScatteringSavePath)

    if ic.forwardScatteringProcedure:
        ic.setForwardScatteringInitialConditions()

        try:  
            backMeanWidths = backScatteringResults.resultsList[0][-1]
            ic.initPars[4::3] = backMeanWidths
            ic.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))
            print("\nChanged ic according to mean widhts from backscattering:\n",
                "Forward scattering initial fitting parameters:\n", ic.initPars,
                "\nForward scattering initial fitting bounds:\n", ic.bounds)
        except UnboundLocalError:
            print("Using the unchanged ic for forward scattering ...")
            pass

        wsFinal, forwardScatteringResults = iterativeFitForDataReduction()
        fitInYSpaceProcedure(wsFinal, forwardScatteringResults)
        forwardScatteringResults.save(ic.forwardScatteringSavePath)


"""
All the functions required to run main() are listed below, in order of appearance
"""


def iterativeFitForDataReduction():

    wsToBeFittedUncropped = loadVesuvioDataWorkspaces()
    wsToBeFitted = cropCloneAndMaskWorkspace(wsToBeFittedUncropped)
    CloneWorkspace(InputWorkspace=wsToBeFitted, OutputWorkspace=ic.name)
    createSlabGeometry()

    # Initialize arrays to store script results
    thisScriptResults = resultsObject(wsToBeFitted)

    for iteration in range(ic.noOfMSIterations):
        # Workspace from previous iteration
        wsToBeFitted = mtd[ic.name+str(iteration)]

        print("\nFitting spectra ", ic.firstSpec, " - ", ic.lastSpec, "\n.............")
        fittedNcpResults = fitNcpToWorkspace(wsToBeFitted)
        thisScriptResults.append(iteration, fittedNcpResults)

        if (iteration < ic.noOfMSIterations - 1):  

            meanWidths, meanIntensityRatios = fittedNcpResults[:2]
            CloneWorkspace(InputWorkspace=ic.name, OutputWorkspace="tmpNameWs")

            if ic.MSCorrectionFlag:
                createWorkspacesForMSCorrection(meanWidths, meanIntensityRatios)
                Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=ic.name+"_MulScattering",
                      OutputWorkspace="tmpNameWs")

            if ic.GammaCorrectionFlag:  
                createWorkspacesForGammaCorrection(meanWidths, meanIntensityRatios)
                Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=ic.name+"_gamma_background", 
                      OutputWorkspace="tmpNameWs")

            RenameWorkspace(InputWorkspace="tmpNameWs", OutputWorkspace=ic.name+str(iteration+1))
    wsFinal = mtd[ic.name+str(ic.noOfMSIterations - 1)]
    return wsFinal, thisScriptResults


def loadVesuvioDataWorkspaces():
    """Loads raw and empty workspaces from either LoadVesuvio or user specified path"""
    if ic.loadWsFromUserPathFlag:
        wsToBeFitted =  loadRawAndEmptyWsFromUserPath()
    else:
        wsToBeFitted = loadRawAndEmptyWsFromLoadVesuvio()
    return wsToBeFitted


def loadRawAndEmptyWsFromUserPath():

    print('\n', 'Loading the sample runs: ', ic.runs, '\n')
    Load(Filename=ic.userWsRawPath, OutputWorkspace=ic.name+"raw")
    Rebin(InputWorkspace=ic.name+'raw', Params=ic.tof_binning,
          OutputWorkspace=ic.name+'raw')
    SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
    wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name)

    if ic.mode=="DoubleDifference":
        print('\n', 'Loading the empty runs: ', ic.empty_runs, '\n')
        Load(Filename=ic.userWsEmptyPath, OutputWorkspace=ic.name+"empty")
        Rebin(InputWorkspace=ic.name+'empty', Params=ic.tof_binning,
            OutputWorkspace=ic.name+'empty')
        wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty',
                            OutputWorkspace=ic.name)

    print(wsToBeFitted.name())
    return wsToBeFitted


def loadRawAndEmptyWsFromLoadVesuvio():
    
    print('\n', 'Loading the sample runs: ', ic.runs, '\n')
    LoadVesuvio(Filename=ic.runs, SpectrumList=ic.spectra, Mode=ic.mode,
                InstrumentParFile=ic.ipfile, OutputWorkspace=ic.name+'raw')
    Rebin(InputWorkspace=ic.name+'raw', Params=ic.tof_binning,
          OutputWorkspace=ic.name+'raw')
    SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
    wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name)

    if ic.mode=="DoubleDifference":
        print('\n', 'Loading the empty runs: ', ic.empty_runs, '\n')
        LoadVesuvio(Filename=ic.empty_runs, SpectrumList=ic.spectra, Mode=ic.mode,
                    InstrumentParFile=ic.ipfile, OutputWorkspace=ic.name+'empty')
        Rebin(InputWorkspace=ic.name+'empty', Params=ic.tof_binning,
            OutputWorkspace=ic.name+'empty')
        wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty', 
                            OutputWorkspace=ic.name)
    return wsToBeFitted


def cropCloneAndMaskWorkspace(ws):
    """Returns cloned and cropped workspace with modified name"""
    ws = CropWorkspace(
        InputWorkspace=ws.name(), 
        StartWorkspaceIndex=ic.firstSpecIdx, EndWorkspaceIndex=ic.lastSpecIdx, 
        OutputWorkspace=ws.name()
        )
    wsToBeFitted = CloneWorkspace(
        InputWorkspace=ws.name(), OutputWorkspace=ws.name()+"0"
        )
    MaskDetectors(Workspace=wsToBeFitted, WorkspaceIndexList=ic.maskedDetectorIdx)
    return wsToBeFitted


def createSlabGeometry():
    half_height, half_width, half_thick = 0.5*ic.vertical_width, 0.5*ic.horizontal_width, 0.5*ic.thickness
    xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"
    CreateSampleShape(ic.name, xml_str)

def fitNcpHelper(args, ic):
    """Function to unpack arguments from mapping"""
    return partial(fitNcpToSingleSpec, ic=ic)(*args)


def fitNcpToWorkspace(ws):
    """Firstly calculates matrices for all spectrums,
    then iterates over each spectrum
    """
    wsDataY = ws.extractY()       #DataY unaltered
    dataY, dataX, dataE = loadWorkspaceIntoArrays(ws)                     
    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(dataX)
    
    if ic.withMultiprocessing:
        # pool = mp.Pool(processes=mp.cpu_count())
        # fitPars = np.array(pool.starmap(
        #     partial(fitNcpToSingleSpec, ic=ic), 
        #     zip(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)
        #     ))
        # pool.close()

        with ProcessPoolExecutor() as executor:        # with statement calls shutdown() automatically
            fitPars = np.array(list(executor.map(
                partial(fitNcpHelper, ic=ic), 
                zip(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)
                )))
        
    else:
        fitPars = np.array(list(map(
            partial(fitNcpToSingleSpec, ic=ic), 
            dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays
            )))       

    fitParsObj = FitParameters(fitPars)
    fitParsObj.printPars()
    meanWidths, meanIntensityRatios = fitParsObj.getMeanWidthsAndIntensities()

    # Create ncpTotal workspaces
    mainPars = fitParsObj.mainPars
    ncpForEachMass = np.array(list(map(
        buildNcpFromSpec, 
        mainPars , ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays
    )))
    ncpTotal = np.sum(ncpForEachMass, axis=1)  
    createNcpWorkspaces(ncpForEachMass, ncpTotal, ws)  

    return [meanWidths, meanIntensityRatios, fitPars, ncpTotal, wsDataY, ncpForEachMass]


def loadWorkspaceIntoArrays(ws):
    """Output: dataY, dataX and dataE as arrays and converted to point data"""
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    histWidths = dataX[:, 1:] - dataX[:, :-1]
    dataY = dataY[:, :-1] / histWidths
    dataE = dataE[:, :-1] / histWidths
    dataX = (dataX[:, 1:] + dataX[:, :-1]) / 2
    return dataY, dataX, dataE


def prepareFitArgs(dataX):
    instrPars = loadInstrParsFileIntoArray(ic.InstrParsPath, ic.firstSpec, ic.lastSpec)       
    resolutionPars = loadResolutionPars(instrPars)                                   

    v0, E0, delta_E, delta_Q = calculateKinematicsArrays(dataX, instrPars)   
    kinematicArrays = np.array([v0, E0, delta_E, delta_Q])
    ySpacesForEachMass = convertDataXToYSpacesForEachMass(dataX, ic.masses, delta_Q, delta_E)        
    
    kinematicArrays = reshapeArrayPerSpectrum(kinematicArrays)
    ySpacesForEachMass = reshapeArrayPerSpectrum(ySpacesForEachMass)
    return resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass


def loadInstrParsFileIntoArray(InstrParsPath, firstSpec, lastSpec):
    """Loads instrument parameters into array, from the file in the specified path"""
    data = np.loadtxt(InstrParsPath, dtype=str)[1:].astype(float)

    spectra = data[:, 0]
    select_rows = np.where((spectra >= firstSpec) & (spectra <= lastSpec))
    instrPars = data[select_rows]
    return instrPars


def loadResolutionPars(instrPars):
    """Resolution of parameters to propagate into TOF resolution
       Output: matrix with each parameter in each column"""
    spectrums = instrPars[:, 0] 
    L = len(spectrums)
    #for spec no below 135, back scattering detectors, mode is double difference
    #for spec no 135 or above, front scattering detectors, mode is single difference
    dE1 = np.where(spectrums < 135, 88.7, 73)       #meV, STD
    dE1_lorz = np.where(spectrums < 135, 40.3, 24)  #meV, HFHM
    dTOF = np.repeat(0.37, L)      #us
    dTheta = np.repeat(0.016, L)   #rad
    dL0 = np.repeat(0.021, L)      #meters
    dL1 = np.repeat(0.023, L)      #meters
    
    resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose() 
    return resolutionPars 


def calculateKinematicsArrays(dataX, instrPars):          
    """Kinematics quantities calculated from TOF data"""   

    mN, Ef, en_to_vel, vf, hbar = loadConstants()    
    det, plick, angle, T0, L0, L1 = np.hsplit(instrPars, 6)     #each is of len(dataX)
    t_us = dataX - T0                                           #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  np.square( v0 / en_to_vel )            #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 - Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt(E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q              #shape(no of spectrums, no of bins)


def reshapeArrayPerSpectrum(A):
    """Exchanges the first two indices of an array A,
    ao rearranges array to match iteration per spectrum of main fitting map()
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertDataXToYSpacesForEachMass(dataX, masses, delta_Q, delta_E):
    "Calculates y spaces from TOF data, each row corresponds to one mass" 
    
    #prepare arrays to broadcast
    dataX = dataX[np.newaxis, :, :]
    delta_Q = delta_Q[np.newaxis, :, :]
    delta_E = delta_E[np.newaxis, :, :]  

    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    masses = masses.reshape(ic.noOfMasses, 1, 1)

    energyRecoil = np.square( hbar * delta_Q ) / 2. / masses              
    ySpacesForEachMass = masses / hbar**2 /delta_Q * (delta_E - energyRecoil)    #y-scaling  
    return ySpacesForEachMass


def fitNcpToSingleSpec(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Fits the NCP and returns the best fit parameters for one spectrum"""

    if np.all(dataY == 0) | np.all(np.isnan(dataY)): 
        return np.full(len(ic.initPars)+3, np.nan)
    
    scaledPars = ic.initPars * ic.scalingFactors
    scaledBounds = ic.bounds * ic.scalingFactors[:, np.newaxis]

    result = optimize.minimize(
        errorFunction, 
        scaledPars, 
        args=(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic),
        method='SLSQP', 
        bounds = scaledBounds, 
        constraints=ic.constraints
        )

    fitScaledPars = result["x"]
    fitPars = fitScaledPars / ic.scalingFactors

    noDegreesOfFreedom = len(dataY) - len(fitPars)
    specFitPars = np.append(instrPars[0], fitPars)
    return np.append(specFitPars, [result["fun"] / noDegreesOfFreedom, result["nit"]])


def errorFunction(scaledPars, dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Error function to be minimized, operates in TOF space"""

    unscaledPars = scaledPars / ic.scalingFactors
    ncpForEachMass, ncpTotal = calculateNcpSpec(unscaledPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic)

    if np.all(dataE == 0) | np.all(np.isnan(dataE)):
        # This condition is not usually satisfied but in the exceptional case that it is,
        # we can use a statistical weight to make sure the chi2 used is not too small for the 
        # optimization algorithm. This is not used in the original script
        chi2 = (ncpTotal - dataY)**2 / dataY**2
    else:
        chi2 =  (ncpTotal - dataY)**2 / dataE**2    
    return np.sum(chi2)


def calculateNcpSpec(unscaledPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):    
    """Creates a synthetic C(t) to be fitted to TOF values of a single spectrum, from J(y) and resolution functions
       Shapes: datax (1, n), ySpacesForEachMass (4, n), res (4, 2), deltaQ (1, n), E0 (1,n),
       where n is no of bins"""
    
    masses, intensities, widths, centers = prepareArraysFromPars(ic, unscaledPars) 
    v0, E0, deltaE, deltaQ = kinematicArrays
    
    gaussRes, lorzRes = caculateResolutionForEachMass(
        masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays
        )
    totalGaussWidth = np.sqrt(widths**2 + gaussRes**2)                 
    
    JOfY = pseudoVoigt(ySpacesForEachMass - centers, totalGaussWidth, lorzRes)  
    
    FSE =  - numericalThirdDerivative(ySpacesForEachMass, JOfY) * widths**4 / deltaQ * 0.72 
    
    ncpForEachMass = intensities * (JOfY + FSE) * E0 * E0**(-0.92) * masses / deltaQ   
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    return ncpForEachMass, ncpTotal


def prepareArraysFromPars(ic, initPars):
    """Extracts the intensities, widths and centers from the fitting parameters
        Reshapes all of the arrays to collumns, for the calculation of the ncp,"""

    shapeOfArrays = (ic.noOfMasses, 1)
    masses = ic.masses.reshape(shapeOfArrays)    
    intensities = initPars[::3].reshape(shapeOfArrays)
    widths = initPars[1::3].reshape(shapeOfArrays)
    centers = initPars[2::3].reshape(shapeOfArrays)  
    return masses, intensities, widths, centers 


def caculateResolutionForEachMass(masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays):    
    """Calculates the gaussian and lorentzian resolution
    output: two column vectors, each row corresponds to each mass"""
    
    v0, E0, delta_E, delta_Q = kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays)
    
    gaussianResWidth = calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    lorentzianResWidth = calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    return gaussianResWidth, lorentzianResWidth


def kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays):
    """v0, E0, deltaE, deltaQ at the peak of the ncpTotal for each mass"""

    shapeOfArrays = (len(ySpacesForEachMass), 1)
    proximityToYCenters = np.abs(ySpacesForEachMass - centers)
    yClosestToCenters = proximityToYCenters.min(axis=1).reshape(shapeOfArrays)
    yCentersMask = proximityToYCenters == yClosestToCenters

    v0, E0, deltaE, deltaQ = kinematicArrays

    # Expand arrays to match shape of yCentersMask
    v0 = v0 * np.ones(shapeOfArrays)
    E0 = E0 * np.ones(shapeOfArrays)
    deltaE = deltaE * np.ones(shapeOfArrays)
    deltaQ = deltaQ * np.ones(shapeOfArrays)

    v0 = v0[yCentersMask].reshape(shapeOfArrays)
    E0 = E0[yCentersMask].reshape(shapeOfArrays)
    deltaE = deltaE[yCentersMask].reshape(shapeOfArrays)
    deltaQ = deltaQ[yCentersMask].reshape(shapeOfArrays)
    return v0, E0, deltaE, deltaQ


def calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars):
    # Currently the function that takes the most time in the fitting
    if masses.shape != (len(masses), 1):
        raise ValueError("The shape of the masses array needs to be a collumn!")

    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi/180

    dWdE1 = 1. + (E0 / Ef)**1.5 * (L1 / L0)
    dWdTOF = 2. * E0 * v0 / L0
    dWdL1 = 2. * E0**1.5 / Ef**0.5 / L0
    dWdL0 = 2. * E0 / L0

    dW2 = dWdE1**2*dE1**2 + dWdTOF**2*dTOF**2 + dWdL1**2*dL1**2 + dWdL0**2*dL0**2
    # conversion from meV^2 to A^-2, dydW = (M/q)^2
    dW2 *= (masses / hbar**2 / delta_Q)**2

    dQdE1 = 1. - (E0 / Ef)**1.5 * L1/L0 - np.cos(angle) * ((E0 / Ef)**0.5 - L1/L0 * E0/Ef)
    dQdTOF = 2.*E0 * v0/L0
    dQdL1 = 2.*E0**1.5 / L0 / Ef**0.5
    dQdL0 = 2.*E0 / L0
    dQdTheta = 2. * np.sqrt(E0 * Ef) * np.sin(angle)

    dQ2 = dQdE1**2*dE1**2 + (dQdTOF**2*dTOF**2 + dQdL1**2*dL1**2 + dQdL0 **
                             2*dL0**2)*np.abs(Ef/E0*np.cos(angle)-1) + dQdTheta**2*dTheta**2
    dQ2 *= (mN / hbar**2 / delta_Q)**2

    # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
    gaussianResWidth = np.sqrt(dW2 + dQ2)
    return gaussianResWidth


def calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars):
    if masses.shape != (len(masses), 1):
        raise ValueError("The shape of the masses array needs to be a collumn!")
        
    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi / 180

    dWdE1_lor = (1. + (E0/Ef)**1.5 * (L1/L0))**2
    # conversion from meV^2 to A^-2
    dWdE1_lor *= (masses / hbar**2 / delta_Q)**2

    dQdE1_lor = (1. - (E0/Ef)**1.5 * L1/L0 - np.cos(angle)
                 * ((E0/Ef)**0.5 + L1/L0 * E0/Ef))**2
    dQdE1_lor *= (mN / hbar**2 / delta_Q)**2

    lorentzianResWidth = np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz   # in A-1
    return lorentzianResWidth


def loadConstants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf = np.sqrt(Ef) * en_to_vel  # m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar


def pseudoVoigt(x, sigma, gamma):
    """Convolution between Gaussian with std sigma and Lorentzian with HWHM gamma"""
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2)
    eta = 1.36603 * fl/f - 0.47719 * (fl/f)**2 + 0.11116 * (fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f / 2.
    pseudo_voigt = eta * \
        lorentizian(x, gamma_v) + (1.-eta) * gaussian(x, sigma_v)
    norm = np.sum(pseudo_voigt)*(x[1]-x[0])
    return pseudo_voigt  # /np.abs(norm)


def gaussian(x, sigma):
    """Gaussian function centered at zero"""
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian


def lorentizian(x, gamma):
    """Lorentzian centered at zero"""
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian


def numericalThirdDerivative(x, fun):
    k6 = (- fun[:, 12:] + fun[:, :-12]) * 1
    k5 = (+ fun[:, 11:-1] - fun[:, 1:-11]) * 24
    k4 = (- fun[:, 10:-2] + fun[:, 2:-10]) * 192
    k3 = (+ fun[:,  9:-3] - fun[:, 3:-9]) * 488
    k2 = (+ fun[:,  8:-4] - fun[:, 4:-8]) * 387
    k1 = (- fun[:,  7:-5] + fun[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3

    derivative = np.zeros(fun.shape)
    derivative[:, 6:-6] = dev
    # Padded with zeros left and right to return array with same shape
    return derivative


class FitParameters:
    """Stores the fitted parameters from map and defines methods to extract information"""

    def __init__(self, fitPars):
        self.spec = fitPars[:, 0][:, np.newaxis]
        self.chi2 = fitPars[:, -2][:, np.newaxis]
        self.nit = fitPars[:, -1][:, np.newaxis]

        mainPars = fitPars[:, 1:-2]
        self.intensities = mainPars[:, 0::3]
        self.widths = mainPars[:, 1::3]
        self.centers = mainPars[:, 2::3]
        self.mainPars = mainPars


    def printPars(self):
        print("[Spec Intensities----Widths----Centers Chi2 Nit]:\n\n", 
              np.hstack((self.spec, self.intensities, self.widths, self.centers, self.chi2, self.nit)))


    def getMeanWidthsAndIntensities(self):
        noOfMasses = ic.noOfMasses
        widths = self.widths.T
        intensities = self.intensities.T

        meanWidths = np.nanmean(widths, axis=1).reshape(noOfMasses, 1)  
        stdWidths = np.nanstd(widths, axis=1).reshape(noOfMasses, 1)

        # Subtraction row by row
        widthDeviation = np.abs(widths - meanWidths)
        # Where True, replace by nan
        betterWidths = np.where(widthDeviation > stdWidths, np.nan, widths)
        betterIntensities = np.where(widthDeviation > stdWidths, np.nan, intensities)

        meanWidths = np.nanmean(betterWidths, axis=1)  
        stdWidths = np.nanstd(betterWidths, axis=1)

        # Not nansum(), to propagate nan
        normalization = np.sum(betterIntensities, axis=0)
        intensityRatios = betterIntensities / normalization

        meanIntensityRatios = np.nanmean(intensityRatios, axis=1)
        stdIntensityRatios = np.nanstd(intensityRatios, axis=1)

        print("\nMasses: ", ic.masses.reshape(1, noOfMasses),
            "\nMean Widths: ", meanWidths,
            "\nMean Intensity Ratios: ", meanIntensityRatios)
        return meanWidths, meanIntensityRatios


def buildNcpFromSpec(initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):
    """input: all row shape
       output: row shape with the ncpTotal for each mass"""

    if np.all(np.isnan(initPars)):
        return np.full(ySpacesForEachMass.shape, np.nan)
    
    ncpForEachMass, ncpTotal = calculateNcpSpec(initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic)        
    return ncpForEachMass


def createNcpWorkspaces(ncpForEachMass, ncpTotal, ws):
    """Creates workspaces from ncp array data"""

    # Need to rearrage array of yspaces into seperate arrays for each mass
    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    dataX = ws.extractX()
    dataE = np.zeros(dataX.shape)

    # Original script does not have this step to multiply by histogram widths
    histWidths = dataX[:, 1:] - dataX[:, :-1]
    dataX = dataX[:, :-1]       # Cut last column to match ncpTotal length
    dataY = ncpTotal * histWidths
    dataE = dataE[:, :-1] * histWidths

    CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(),
                     Nspec=len(dataX), OutputWorkspace=ws.name()+"_tof_fitted_profiles")

    for i, ncp_m in enumerate(ncpForEachMass):
        CreateWorkspace(DataX=dataX.flatten(), DataY=ncp_m.flatten(), Nspec=len(dataX),
                        OutputWorkspace=ws.name()+"_tof_fitted_profile_"+str(i+1))


def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def createWorkspacesForMSCorrection(meanWidths, meanIntensityRatios):
    """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""

    sampleProperties = calcMSCorrectionSampleProperties(meanWidths, meanIntensityRatios)
    print("\n The sample properties for Multiple Scattering correction are:\n ", 
            sampleProperties)
    createMulScatWorkspaces(ic.name, sampleProperties)


def calcMSCorrectionSampleProperties(meanWidths, meanIntensityRatios):
    masses = ic.masses.flatten()

    if ic.hydrogen_peak:
        # ADDITION OF THE HYDROGEN intensities AS PROPORTIONAL TO A FITTED NCP (OXYGEN HERE)
        masses = np.append(masses, 1.0079)
        meanWidths = np.append(meanWidths, 5.0)
        meanIntensityRatios = np.append(
            meanIntensityRatios, ic.hydrogen_to_mass0_ratio * meanIntensityRatios[0]
            )
        meanIntensityRatios /= np.sum(meanIntensityRatios)

    MSProperties = np.zeros(3*len(masses))
    MSProperties[::3] = masses
    MSProperties[1::3] = meanIntensityRatios
    MSProperties[2::3] = meanWidths
    sampleProperties = list(MSProperties)   
    return sampleProperties


def createMulScatWorkspaces(wsName, sampleProperties):
    """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""

    print("Evaluating the Multiple Scattering Correction.")
    # selects only the masses, every 3 numbers
    MS_masses = sampleProperties[::3]
    # same as above, but starts at first intensities
    MS_amplitudes = sampleProperties[1::3]

    dens, trans = VesuvioThickness(
        Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=ic.transmission_guess, Thickness=0.1
        )

    _TotScattering, _MulScattering = VesuvioCalculateMS(
        wsName, 
        NoOfMasses=len(MS_masses), 
        SampleDensity=dens.cell(9, 1),
        AtomicProperties=sampleProperties, 
        BeamRadius=2.5,
        NumScatters=ic.multiple_scattering_order,
        NumEventsPerRun=int(ic.number_of_events)
        )

    data_normalisation = Integration(wsName)
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering", "_TotScattering"):
        Divide(LHSWorkspace=workspace, RHSWorkspace=simulation_normalisation, 
               OutputWorkspace=workspace)
        Multiply(LHSWorkspace=workspace, RHSWorkspace=data_normalisation, 
                 OutputWorkspace=workspace)
        RenameWorkspace(InputWorkspace=workspace,
                        OutputWorkspace=str(wsName)+workspace)
    DeleteWorkspaces(
        [data_normalisation, simulation_normalisation, trans, dens]
        )
  # The only remaining workspaces are the _MulScattering and _TotScattering


def createWorkspacesForGammaCorrection(meanWidths, meanIntensityRatios):
    """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

    # I do not know why, but setting these instrument parameters is required
    SetInstrumentParameter(ic.name, ParameterName='hwhm_lorentz', 
                            ParameterType='Number', Value='24.0')
    SetInstrumentParameter(ic.name, ParameterName='sigma_gauss', 
                            ParameterType='Number', Value='73.0')

    profiles = calcGammaCorrectionProfiles(ic.masses, meanWidths, meanIntensityRatios)
    background, corrected = VesuvioCalculateGammaBackground(
        InputWorkspace=ic.name, ComptonFunction=profiles
        )
    RenameWorkspace(InputWorkspace= background, OutputWorkspace = ic.name+"_gamma_background")
    Scale(InputWorkspace = ic.name+"_gamma_background", OutputWorkspace = ic.name+"_gamma_background", 
        Factor=0.9, Operation="Multiply")

    DeleteWorkspace(corrected)


def calcGammaCorrectionProfiles(masses, meanWidths, meanIntensityRatios):
    masses = masses.flatten()
    profiles = ""
    for mass, width, intensity in zip(masses, meanWidths, meanIntensityRatios):
        profiles += "name=GaussianComptonProfile,Mass="   \
                    + str(mass) + ",Width=" + str(width)  \
                    + ",Intensity=" + str(intensity) + ';'
    print("\n The sample properties for Gamma Correction are:\n",
            profiles)
    return profiles


def fitInYSpaceProcedure(wsFinal, thisScriptResults):
    ncpForEachMass = thisScriptResults.resultsList[5][-1]  # Select last iteration
    wsYSpaceSymSum, wsRes = isolateHProfileInYSpace(wsFinal, ncpForEachMass)
    popt, perr = fitTheHProfileInYSpace(wsYSpaceSymSum, wsRes)
    wsH = mtd[wsFinal.name()+"_H"]

    thisScriptResults.storeResultsOfYSpaceFit(wsFinal, wsH, wsYSpaceSymSum, wsRes, popt, perr)


def isolateHProfileInYSpace(wsFinal, ncpForEachMass):
    massH = 1.0079
    wsRes = calculateMantidResolution(wsFinal, massH)  

    wsSubMass = subtractAllMassesExceptFirst(wsFinal, ncpForEachMass)
    averagedSpectraYSpace = averageJOfYOverAllSpectra(wsSubMass, massH) 
    return averagedSpectraYSpace, wsRes


def calculateMantidResolution(ws, mass):
    rebinPars=ic.rebinParametersForYSpaceFit
    for index in range(ws.getNumberHistograms()):
        if np.all(ws.dataY(index)[:] == 0):  # Ignore masked spectra
            pass
        else:
            VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
            Rebin(InputWorkspace="tmp", Params=rebinPars, OutputWorkspace="tmp")

            if index == 0:   # Ensures that workspace has desired units
                RenameWorkspace("tmp","resolution")
            else:
                AppendSpectra("resolution", "tmp", OutputWorkspace= "resolution")

    try:
        SumSpectra(InputWorkspace="resolution",OutputWorkspace="resolution")
    except ValueError:
        raise ValueError ("All the rows from the workspace to be fitted are Nan!")

    normalise_workspace("resolution")
    DeleteWorkspace("tmp")
    return mtd["resolution"]

    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def subtractAllMassesExceptFirst(ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMass = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotal = np.sum(ncpForEachMass, axis=0)

    dataY, dataX = ws.extractY(), ws.extractX() 
    
    dataY[:, :-1] -= ncpTotal * (dataX[:, 1:] - dataX[:, :-1])
    # Although original cuts two last columns and this optimized cuts only last one,
    # I have tested it and this effect is not significant

    # Pass the data onto a Workspace, clone to preserve properties
    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_H")
    for i in range(wsSubMass.getNumberHistograms()):  # Keeps the faulty last column
        wsSubMass.dataY(i)[:] = dataY[i, :]

     # Safeguard against possible NaNs
    MaskDetectors(Workspace=wsSubMass, SpectraList=ic.maskedSpecNo)    

    if np.any(np.isnan(mtd[ws.name()+"_H"].extractY())):
        raise ValueError("The workspace for the isolated H data countains NaNs, \
                            might cause problems!")
    return wsSubMass


def averageJOfYOverAllSpectra(ws0, mass):
    wsYSpace = convertToYSpace(ws0, mass)
    wsYSpaceSym = SymetriseWorkspace(wsYSpace)
    wsOnes = replaceNonZeroNanValuesByOnesInWs(wsYSpaceSym)
    wsOnesSum = SumSpectra(wsOnes)

    wsYSpaceSymSum = SumSpectra(wsYSpaceSym)
    averagedSpectraYSpace = Divide(
         LHSWorkspace=wsYSpaceSymSum, RHSWorkspace=wsOnesSum,
         OutputWorkspace=ws0.name()+"_JOfY_symetrized_averaged"
    )
    return averagedSpectraYSpace


def convertToYSpace(ws0, mass):
    ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, 
        OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
    rebinPars=ic.rebinParametersForYSpaceFit
    Rebin(
        InputWorkspace=ws0.name()+"_JoY", Params=rebinPars, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )
    normalise_workspace(ws0.name()+"_JoY")
    return mtd[ws0.name()+"_JoY"]


def SymetriseWorkspace(wsYSpace):
    dataY = wsYSpace.extractY() 
    dataE = wsYSpace.extractE()
    dataX = wsYSpace.extractX()

    if ic.symmetriseHProfileUsingAveragesFlag:
        dataY = symetriseArrayUsingAverages(dataY)
        dataE = symetriseArrayUsingAverages(dataE)
    else:
        dataY = symetriseArrayNegMirrorsPos(dataX, dataY)
        dataE = symetriseArrayNegMirrorsPos(dataX, dataE)

    wsYSym = CloneWorkspace(wsYSpace)
    for i in range(wsYSpace.getNumberHistograms()):
        wsYSym.dataY(i)[:] = dataY[i, :]
        wsYSym.dataE(i)[:] = dataE[i, :]
    return wsYSym


def symetriseArrayUsingAverages(dataY):
    # Code below works as long as dataX is symetric
    # Need to account for kinematic cut-offs
    dataY = np.where(dataY==0, np.flip(dataY, axis=1), dataY)
    # With zeros being masked, can perform the average
    dataY = (dataY + np.flip(dataY, axis=1)) / 2
    return dataY


def symetriseArrayNegMirrorsPos(dataX, dataY):
    dataY = np.where(dataX<0, np.flip(dataY, axis=1), dataY)
    return dataY


def replaceNonZeroNanValuesByOnesInWs(wsYSym):
    dataY = wsYSym.extractY()
    dataE = wsYSym.extractE()
    
    dataY[np.isnan(dataY)] = 0   # Safeguard agaist nans
    nonZerosMask = ~(dataY==0)
    dataYones = np.where(nonZerosMask, 1, 0)
    dataE = np.full(dataE.shape, 0.000001)  # Value from original script

    # Build Workspaces, couldn't find a method for this in Mantid
    wsOnes = CloneWorkspace(wsYSym)
    for i in range(wsYSym.getNumberHistograms()):
        wsOnes.dataY(i)[:] = dataYones[i, :]
        wsOnes.dataE(i)[:] = dataE[i, :]
    return wsOnes


def fitTheHProfileInYSpace(wsYSpaceSym, wsRes):
    if ic.useScipyCurveFitToHProfileFlag:
        popt, pcov = fitProfileCurveFit(wsYSpaceSym, wsRes)
        print("popt:\n", popt)
        print("pcov:\n", pcov)
        perr = np.sqrt(np.diag(pcov))
    else:
        popt, perr = fitProfileMantidFit(wsYSpaceSym, wsRes)
    
    return popt, perr


def fitProfileCurveFit(wsYSpaceSym, wsRes):
    res = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]

    # Interpolate Resolution to get single peak at zero
    start, interval, end = [float(i) for i in ic.rebinParametersForYSpaceFit.split(",")]
    resNewX = np.arange(start, end, interval)
    res = np.interp(resNewX, resX, res)

    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    if ic.singleGaussFitToHProfile:
        def convolvedGaussian(x, y0, x0, A, sigma):
            histWidths = x[1:] - x[:-1]
            if ~ (np.max(histWidths)==np.min(histWidths)):
                raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

            gaussFunc = gaussianFit(x, y0, x0, A, sigma)
            convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]  
            return convGauss
        p0 = [0, 0, 1, 5]

    else:
        def convolvedGaussian(x, y0, x0, A, sigma):
            histWidths = x[1:] - x[:-1]
            if ~ (np.max(histWidths)==np.min(histWidths)):
                raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

            gaussFunc = gaussianFit(x, y0, x0, A, 4.76) + gaussianFit(x, 0, x0, 0.054*A, sigma)
            convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]
            return convGauss
        p0 = [0, 0, 0.7143, 5]

    popt, pcov = optimize.curve_fit(
        convolvedGaussian, 
        dataX, 
        dataY, 
        p0=p0,
        sigma=dataE
    )
    yfit = convolvedGaussian(dataX, *popt)
    Residuals = dataY - yfit
    
    # Create Workspace with the fit results
    # TODO add DataE 
    CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
                    DataY=np.concatenate((dataY, yfit, Residuals)), 
                    NSpec=3,
                    OutputWorkspace="CurveFitResults")
    return popt, pcov


def gaussianFit(x, y0, x0, A, sigma):
    """Gaussian centered at zero"""
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


def fitProfileMantidFit(wsYSpaceSym, wsRes):

    if ic.singleGaussFitToHProfile:
        popt, perr = np.zeros((2, 5)), np.zeros((2, 5))
    else:
        popt, perr = np.zeros((2, 6)), np.zeros((2, 6))


    print('\n','Fit on the sum of spectra in the West domain','\n')     
    for i, minimizer_sum in enumerate(['Levenberg-Marquardt','Simplex']):
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = ic.name+minimizer_sum+'_joy_sum_fitted')
        
        if ic.singleGaussFitToHProfile:
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution,WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()'''
        else:
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution,WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
            +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
            y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''

        Fit(
            Function=function, 
            InputWorkspace=ic.name+minimizer_sum+'_joy_sum_fitted',
            Output=ic.name+minimizer_sum+'_joy_sum_fitted',
            Minimizer=minimizer_sum
            )
        
        ws=mtd[ic.name+minimizer_sum+'_joy_sum_fitted_Parameters']
        popt[i] = ws.column("Value")
        perr[i] = ws.column("Error")

        # print('Using the minimizer: ',minimizer_sum)
        # print('Hydrogen standard deviation: ',ws.cell(3,1),' +/- ',ws.cell(3,2))   # Selects the wrong parameter in the case of the double gaussian
    print("\nFitting ------ \npopt:\n", popt, "\nperr:\n", perr)
    return popt, perr


class resultsObject: 
    """Used to store results of the script"""

    def __init__(self, wsToBeFitted):
        """Initializes arrays full of zeros"""

        noOfSpec = wsToBeFitted.getNumberHistograms()
        lenOfSpec = wsToBeFitted.blocksize()

        all_fit_workspaces = np.zeros((ic.noOfMSIterations, noOfSpec, lenOfSpec))
        all_spec_best_par_chi_nit = np.zeros((ic.noOfMSIterations, noOfSpec, ic.noOfMasses*3+3))
        all_tot_ncp = np.zeros((ic.noOfMSIterations, noOfSpec, lenOfSpec - 1))
        all_ncp_for_each_mass = np.zeros((ic.noOfMSIterations, noOfSpec, ic.noOfMasses, lenOfSpec - 1))
        all_mean_widths = np.zeros((ic.noOfMSIterations, ic.noOfMasses))
        all_mean_intensities = np.zeros(all_mean_widths.shape)

        self.resultsList = [all_mean_widths, all_mean_intensities,
                            all_spec_best_par_chi_nit, all_tot_ncp, 
                            all_fit_workspaces, all_ncp_for_each_mass]


    def append(self, mulscatIter, resultsToAppend):
        """Append results at a given MS iteration"""
        for i, currentMSArray in enumerate(resultsToAppend):
            self.resultsList[i][mulscatIter] = currentMSArray

    
    # Set default of yspace fit parameters to zero
    YSpaceSymSumDataY = 0
    YSpaceSymSumDataE = 0
    resolution = 0
    finalRawDataY = 0
    finalRawDataE = 0
    HdataY = 0
    popt = 0
    perr = 0

    def storeResultsOfYSpaceFit(self, wsFinal, wsH, wsYSpaceSymSum, wsRes, popt, perr):
        self.finalRawDataY = wsFinal.extractY()
        self.finalRawDataE = wsFinal.extractE()
        self.HdataY = wsH.extractY()
        self.YSpaceSymSumDataY = wsYSpaceSymSum.extractY()
        self.YSpaceSymSumDataE = wsYSpaceSymSum.extractE()
        self.resolution = wsRes.extractY()
        self.popt = popt
        self.perr = perr


    def save(self, savePath):
        """Saves all of the arrays stored in this object"""

        all_mean_widths, all_mean_intensities, \
        all_spec_best_par_chi_nit, all_tot_ncp, all_fit_workspaces, \
        all_ncp_for_each_mass = self.resultsList
        np.savez(savePath,
                 all_fit_workspaces=all_fit_workspaces,
                 all_spec_best_par_chi_nit=all_spec_best_par_chi_nit,
                 all_mean_widths=all_mean_widths,
                 all_mean_intensities=all_mean_intensities,
                 all_tot_ncp=all_tot_ncp,
                 all_ncp_for_each_mass=all_ncp_for_each_mass,
                 YSpaceSymSumDataY=self.YSpaceSymSumDataY,
                 YSpaceSymSumDataE=self.YSpaceSymSumDataE,
                 resolution=self.resolution, HdataY=self.HdataY,
                 finalRawDataY=self.finalRawDataY, finalRawDataE=self.finalRawDataE,
                 popt=self.popt, perr=self.perr)


if __name__=="__main__":
    freeze_support()
    start_time = time.time()
    main()
    end_time = time.time()
    print("running time: ", end_time-start_time, " seconds")
