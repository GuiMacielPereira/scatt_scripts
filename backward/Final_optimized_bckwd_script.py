import numpy as np
import mantid
from mantid.simpleapi import *
from scipy import optimize
import time
from pathlib import Path

start_time = time.time()
# format print output of arrays
np.set_printoptions(suppress=True, precision=4, linewidth=150)

#######################################################################################################################################
#######################################################                      ##########################################################
#######################################################     USER SECTION     ##########################################################
#######################################################                      ##########################################################
#######################################################################################################################################
'''
The user section is composed of an initialisation section, an iterative analysis/reduction section
of the spectra in the time-of-flight domain, and a final section where the analysis of the corrected
hydrogen neutron Compton profile is possible in the Y-space domain.

The fit procedure in the time-of-flight domain is  based on the scipy.minimize.optimize() tool,
used with the SLSQP minimizer, that can handle both boundaries and constraints for fitting parameters.
'''


def loadRawAndEmptyWorkspaces(loadVesuvioWs):
    """Loads raw and empty workspaces from either LoadVesuvio or user specified path"""
    if loadVesuvioWs:
        loadRawAndEmptyWsVesuvio()
    else:
        userRawPath = r"./input_ws/CePtGe12_100K_DD_raw.nxs"
        userEmptyPath = r"./input_ws/CePtGe12_100K_DD_empty.nxs"
        loadRawAndEmptyWsFromUserPath(userRawPath, userEmptyPath)


def loadRawAndEmptyWsFromUserPath(userRawPath, userEmptyPath):
    tof_binning='275.,1.,420'                     # Binning of ToF spectra
    runs='44462-44463'         # 100K             # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K             # The numbers of the empty runs to be subtracted

    print('\n', 'Loading the sample runs: ', runs, '\n')
    Load(Filename=userRawPath, OutputWorkspace=name+"raw")
    Rebin(InputWorkspace=name+'raw', Params=tof_binning,
          OutputWorkspace=name+'raw')
    SumSpectra(InputWorkspace=name+'raw', OutputWorkspace=name+'raw'+'_sum')

    print('\n', 'Loading the empty runs: ', empty_runs, '\n')
    Load(Filename=userEmptyPath, OutputWorkspace=name+"empty")
    Rebin(InputWorkspace=name+'empty', Params=tof_binning,
          OutputWorkspace=name+'empty')
    Minus(LHSWorkspace=name+'raw', RHSWorkspace=name +
          'empty', OutputWorkspace=name)


def loadRawAndEmptyWsVesuvio():
    runs='44462-44463'         # 100K             # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K             # The numbers of the empty runs to be subtracted
    spectra='3-134'                               # Spectra to be analysed
    tof_binning='275.,1.,420'                     # Binning of ToF spectra
    mode='DoubleDifference'
    ipfile='ip2018.par'
    
    print('\n', 'Loading the sample runs: ', runs, '\n')
    LoadVesuvio(Filename=runs, SpectrumList=spectra, Mode=mode,
                InstrumentParFile=ipfile, OutputWorkspace=name+'raw')
    Rebin(InputWorkspace=name+'raw', Params=tof_binning,
          OutputWorkspace=name+'raw')
    SumSpectra(InputWorkspace=name+'raw', OutputWorkspace=name+'raw'+'_sum')

    print('\n', 'Loading the empty runs: ', empty_runs, '\n')
    LoadVesuvio(Filename=empty_runs, SpectrumList=spectra, Mode=mode,
                InstrumentParFile=ipfile, OutputWorkspace=name+'empty')
    Rebin(InputWorkspace=name+'empty', Params=tof_binning,
          OutputWorkspace=name+'empty')
    Minus(LHSWorkspace=name+'raw', RHSWorkspace=name +
          'empty', OutputWorkspace=name)


def convertFirstAndLastSpecToIdx(firstSpec, lastSpec):
    """Used because idexes remain consistent between different workspaces, 
    which might not be the case for spec numbers"""
    spec_offset = mtd[name].getSpectrum(0).getSpectrumNo()      #use the main ws as the reference point
    firstIdx = firstSpec - spec_offset
    lastIdx = lastSpec - spec_offset
    return firstIdx, lastIdx


def loadMaskedDetectors(firstSpec, lastSpec):
    detectors_masked = np.array([18,34,42,43,59,60,62,118,119,133])   
    detectors_masked = detectors_masked[(detectors_masked >= firstSpec) & (detectors_masked <= lastSpec)] 
    return detectors_masked


def loadMSPars():
    transmission_guess = 0.98  # experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # hydrogen multiple scattering
    hydrogen_peak = False
    hydrogen_to_mass0_ratio = 0
    # hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5
    return [hydrogen_peak, hydrogen_to_mass0_ratio, transmission_guess, multiple_scattering_order, number_of_events]

def createSlabGeometry(ws_name,vertical_width, horizontal_width, thickness):  #Don't know what it does
    half_height, half_width, half_thick = 0.5*vertical_width, 0.5*horizontal_width, 0.5*thickness
    xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"
    CreateSampleShape(ws_name, xml_str)
    return


def chooseWorkspaceToBeFitted(synthetic_workspace):
    if synthetic_workspace:
        syntheticResultsPath = r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild.npz"
        wsToBeFitted = loadSyntheticNcpWorkspace(syntheticResultsPath)
    else:
        wsToBeFitted = cropAndCloneMainWorkspace()
    return wsToBeFitted


def loadSyntheticNcpWorkspace(syntheticResultsPath):
    """Loads a synthetic ncpTotal workspace from previous fit results path"""
    results = np.load(syntheticResultsPath)
    # Ncp of first iteration
    dataY = results["all_tot_ncp"][0, firstIdx : lastIdx+1]
    # Cut last collumn to match ncpTotal length
    dataX = mtd[name].extractX()[firstIdx : lastIdx+1, : -1]

    wsToBeFitted = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(),
                                   Nspec=len(dataX), OutputWorkspace=name+"0")
    print(dataY.shape, dataX.shape)
    return wsToBeFitted


def cropAndCloneMainWorkspace():
    """Returns cloned and cropped workspace with modified name"""
    CropWorkspace(InputWorkspace=name, StartWorkspaceIndex=firstIdx,
                  EndWorkspaceIndex=lastIdx, OutputWorkspace=name)
    wsToBeFitted = CloneWorkspace(
        InputWorkspace=name, OutputWorkspace=name+"0")
    return wsToBeFitted

# Elements                                                             Cerium     Platinum     Germanium    Aluminium
# Classical value of Standard deviation of the momentum distribution:  16.9966    20.0573      12.2352      7.4615         inv A
# Debye value of Standard deviation of the momentum distribution:      18.22      22.5         15.4         9.93           inv A

#Parameters:   intensities,   NCP Width,    NCP centre
par     =      ( 1,           18.22,       0.       )     #Cerium
bounds  =      ((0, None),   (17,20),    (-30., 30.))     
par    +=      ( 1,           22.5,        0.       )     #Platinum
bounds +=      ((0, None),   (20,25),    (-30., 30.))
par    +=      ( 1,           15.4,        0.       )     #Germanium
bounds +=      ((0, None),   (12.2,18),  (-10., 10.))     
par    +=      ( 1,           9.93,        0.       )     #Aluminium
bounds +=      ((0, None),   (9.8,10),   (-10., 10.))     

# intensities Constraints
# CePt4Ge12 in Al can
#  Ce cross section * stoichiometry = 2.94 * 1 = 2.94    barn
#  Pt cross section * stoichiometry = 11.71 * 4 = 46.84  barn
#  Ge cross section * stoichiometry = 8.6 * 12 = 103.2   barn

constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 2.94/46.84*par[3]},
               {'type': 'eq', 'fun': lambda par:  par[0] - 2.94/103.2*par[6]})

name = 'CePtGe12_100K_DD_'
masses = np.array([140.1, 195.1, 72.6, 27]).reshape(4, 1, 1)  #Will change to shape(4, 1) in the future

repoPath = Path(__file__).absolute().parent  # Path to the repository
InstrParsPath = repoPath / "ip2018.par"

loadVesuvioWs = False
loadRawAndEmptyWorkspaces(loadVesuvioWs)

noOfMSIterations = 1
firstSpec, lastSpec = 3, 134  # 3, 134
firstIdx, lastIdx = convertFirstAndLastSpecToIdx(firstSpec, lastSpec)
detectors_masked = loadMaskedDetectors(firstSpec, lastSpec)

mulscatPars = loadMSPars()
vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # expressed in meters
createSlabGeometry(name, vertical_width, horizontal_width, thickness)

synthetic_workspace = False
wsToBeFitted = chooseWorkspaceToBeFitted(synthetic_workspace)

scaledataY = False

savePath = repoPath / "script_runs" / "opt_spec3-134_iter4_ncp_nightlybuild_cleanest"


def main():
    # Initialize arrays to store script results
    thisScriptResults = resultsObject()

    for iteration in range(noOfMSIterations):
        # Workspace from previous iteration
        wsToBeFitted = mtd[name+str(iteration)]
        # This line is probably not necessary
        MaskDetectors(Workspace=wsToBeFitted, SpectraList=detectors_masked)

        fittedNcpResults = fitNcpToWorkspace(wsToBeFitted)

        thisScriptResults.append(iteration, fittedNcpResults)
        if (iteration < noOfMSIterations - 1):  # evaluate MS correction except if last iteration
            meanWidths, meanIntensityRatios = fittedNcpResults[:2]

            createWorkspacesForMSCorrection(
                meanWidths, meanIntensityRatios, mulscatPars)
            Minus(LHSWorkspace=name, RHSWorkspace=name+"_MulScattering",
                  OutputWorkspace=name+str(iteration+1))
    thisScriptResults.save(savePath)


######################################################################################################################################
#####################################################                          #######################################################
#####################################################   DEVELOPMENT SECTION    #######################################################
#####################################################                          #######################################################
######################################################################################################################################
""""
All the functions required to run main() are listed below, in order of appearance
"""


class resultsObject:
    """Used to store results of the script"""

    def __init__(self):
        """Initializes arrays full of zeros"""

        noOfSpec = wsToBeFitted.getNumberHistograms()
        lenOfSpec = wsToBeFitted.blocksize()
        noOfMasses = len(masses)

        all_fit_workspaces = np.zeros((noOfMSIterations, noOfSpec, lenOfSpec))
        all_spec_best_par_chi_nit = np.zeros(
            (noOfMSIterations, noOfSpec, noOfMasses*3+3))
        all_tot_ncp = np.zeros((noOfMSIterations, noOfSpec, lenOfSpec - 1))
        all_mean_widths = np.zeros((noOfMSIterations, noOfMasses))
        all_mean_intensities = np.zeros(all_mean_widths.shape)

        resultsList = [all_mean_widths, all_mean_intensities,
                       all_spec_best_par_chi_nit, all_tot_ncp, all_fit_workspaces]
        self.resultsList = resultsList

    def append(self, mulscatIter, resultsToAppend):
        """Append results at a given MS iteration"""
        for i, currentMSArray in enumerate(resultsToAppend):
            self.resultsList[i][mulscatIter] = currentMSArray

    def save(self, savePath):
        all_mean_widths, all_mean_intensities, all_spec_best_par_chi_nit, all_tot_ncp, all_fit_workspaces = self.resultsList
        np.savez(savePath,
                 all_fit_workspaces=all_fit_workspaces,
                 all_spec_best_par_chi_nit=all_spec_best_par_chi_nit,
                 all_mean_widths=all_mean_widths,
                 all_mean_intensities=all_mean_intensities,
                 all_tot_ncp=all_tot_ncp)


def fitNcpToWorkspace(ws):
    """Firstly calculates matrices for all spectrums,
    then iterates over each spectrum
    """
    wsDataY = ws.extractY()       #DataY unaltered
    dataY, dataX, dataE = loadWorkspaceIntoArrays(ws)                     
    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(dataX)
    
    scalingFactor = 1
    if scaledataY:
        # No justification for factor of 100 yet
        scalingFactor = 100 / np.sum(dataY, axis=1).reshape(len(dataY), 1)
    dataY *= scalingFactor
    
    #----------Fit all spectrums----------
    fitParsChiNit = map(fitNcpToSingleSpec, dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)
    fitParsChiNit = np.array(list(fitParsChiNit))    
    fitParsChiNit[:, :-2:3] /= scalingFactor
    
    specs = instrPars[:, 0, np.newaxis]                        
    specFitParsChiNit = np.append(specs, fitParsChiNit, axis=1)    
    print("[Spec------------------ Fit Pars---------------------Chi2 Nit]:\n\n", specFitParsChiNit)    
    fitPars  = np.array(specFitParsChiNit)[:, 1:-2] 

    #----------Create ncpTotal workspaces---------
    ncpForEachMass = map(buildNcpFromSpec, fitPars , ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)
    ncpForEachMass = np.array(list(ncpForEachMass))                             
    ncpForEachMass, ncpTotal = createNcpWorkspaces(ncpForEachMass, ws)  

    #--------Retrieve relevant quantities---------
    intensities, widths, positions = fitPars [:, 0::3].T, fitPars [:, 1::3].T, fitPars [:, 2::3].T     
    meanWidths, meanIntensityRatios = calculateMeanWidthsAndIntensities(widths, intensities)
    return [meanWidths, meanIntensityRatios, specFitParsChiNit, ncpTotal, wsDataY]


def loadWorkspaceIntoArrays(ws):
    """Output: dataY, dataX and dataE as arrays and converted to point data"""
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    hist_widths = dataX[:, 1:] - dataX[:, :-1]
    dataY = dataY[:, :-1] / hist_widths
    dataE = dataE[:, :-1] / hist_widths
    dataX = (dataX[:, 1:] + dataX[:, :-1]) / 2
    return dataY, dataX, dataE


def loadInstrParsFileIntoArray(InstrParsPath, firstSpec, lastSpec):
    """Loads instrument parameters into array, from the file in the specified path"""
    data = np.loadtxt(InstrParsPath, dtype=str)[1:].astype(float)

    spectra = data[:, 0]
    select_rows = np.where((spectra >= firstSpec) & (spectra <= lastSpec))
    instrPars = data[select_rows]
    print("instrPars first column: \n", instrPars[:, 0])  
    return instrPars


def loadResolutionPars(instrPars):
    """Resolution of parameters to propagate into TOF resolution
       Output: matrix with each parameter in each column"""
    spectrums = instrPars[:, 0] 
    L = len(spectrums)
    #for spec no below 135, back scattering detectors, in case of double difference
    #for spec no 135 or above, front scattering detectors, in case of single difference
    dE1 = np.where(spectrums < 135, 88.7, 73)   #meV, STD
    dE1_lorz = np.where(spectrums < 135, 40.3, 24)  #meV, HFHM
    dTOF = np.repeat(0.37, L)      #us
    dTheta = np.repeat(0.016, L)   #rad
    dL0 = np.repeat(0.021, L)      #meters
    dL1 = np.repeat(0.023, L)      #meters
    
    resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose()  #store all parameters in a matrix
    return resolutionPars 

def calculateKinematicsArrays(dataX, instrPars):          
    """Kinematics quantities calculated from TOF data"""   
    mN, Ef, en_to_vel, vf, hbar = loadConstants()    
    det, plick, angle, T0, L0, L1 = np.hsplit(instrPars, 6)     #each is of len(dataX)
    t_us = dataX - T0                                         #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  np.square( v0 / en_to_vel )            #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 - Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt(E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q              #shape(no_spectrums, len_spec)

def reshapeArrayPerSpectrum(A):
    """Exchanges the first two indices of an array A,
    ao rearranges array to match iteration per spectrum of main fitting map()
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]

def convertDataXToySpacesForEachMass(dataX, masses, delta_Q, delta_E):
    "Calculates y spaces from TOF data, each row corresponds to one mass"   
    dataX, delta_Q, delta_E = dataX[np.newaxis, :, :],  delta_Q[np.newaxis, :, :], delta_E[np.newaxis, :, :]   #prepare arrays to broadcast
    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    noOfMasses = len(masses)
    masses = masses.reshape(noOfMasses, 1, 1)

    energyRecoil = np.square( hbar * delta_Q ) / 2. / masses              
    ySpacesForEachMass = masses / hbar**2 /delta_Q * (delta_E - energyRecoil)    #y-scaling  
    return ySpacesForEachMass

def fitNcpToSingleSpec(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):
    """Fits the NCP and returns the best fit parameters for one spectrum"""

    if np.all(dataY == 0):  # If all zeros, then parameters are all nan, so they are ignored later down the line
        return np.full(len(par)+2, np.nan)
    
    result = optimize.minimize(errorFunction, 
                               par[:], 
                               args=(masses, dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays),
                               method='SLSQP', 
                               bounds = bounds, 
                               constraints=constraints)

    noDegreesOfFreedom = len(dataY) - len(par)
    return np.append(result["x"], [result["fun"] / noDegreesOfFreedom, result["nit"]])

def errorFunction(par, masses, dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):
    """Error function to be minimized, operates in TOF space"""
    
    ncpForEachMass, ncpTotal = calculateNcpSpec(par, masses, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)
    
    if (np.sum(dataE) > 0):    #don't understand this conditional statement
        chi2 =  ((ncpTotal - dataY)**2)/(dataE)**2    #weighted fit
    else:
        chi2 = (ncpTotal - dataY)**2
    return np.sum(chi2)

def calculateNcpSpec(par, masses, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):    
    """Creates a synthetic C(t) to be fitted to TOF values, from J(y) and resolution functions
       shapes: par (1, 12), masses (4,1,1), datax (1, n), ySpacesForEachMass (4, n), res (4, 2), deltaQ (1, n), E0 (1,n)"""
    
    shapeOfArrays = (len(masses), 1)
    masses = masses.reshape(shapeOfArrays)    
    intensitiesForEachMass = par[::3].reshape(shapeOfArrays)
    widthsForEachMass = par[1::3].reshape(shapeOfArrays)
    centersForEachMass = par[2::3].reshape(shapeOfArrays)
    
    v0, E0, deltaE, deltaQ = kinematicArrays
    
    gaussRes, lorzRes = caculateResolutionForEachMass(masses, ySpacesForEachMass, centersForEachMass, resolutionPars, instrPars, kinematicArrays)
    
    totalGaussWidth = np.sqrt(widthsForEachMass**2 + gaussRes**2)                 
    
    JOfY = pseudoVoigt(ySpacesForEachMass - centersForEachMass, totalGaussWidth, lorzRes)  
    
    FSE =  - numericalThirdDerivative(ySpacesForEachMass, JOfY) * widthsForEachMass**4 / deltaQ * 0.72 
    
    ncpForEachMass = intensitiesForEachMass * (JOfY + FSE) * E0 * E0**(-0.92) * masses / deltaQ   
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    return ncpForEachMass, ncpTotal

def caculateResolutionForEachMass(masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays):    
    """Calculates the gaussian and lorentzian resolution
    output: two column vectors, each row corresponds to each mass"""
    
    v0, E0, delta_E, delta_Q = kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays)
    
    gaussianResWidth = calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    lorentzianResWidth = calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    return gaussianResWidth, lorentzianResWidth

def kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays):
    """v0, E0, deltaE, deltaQ at the center of the ncpTotal for each mass"""
    noOfMasses = len(ySpacesForEachMass)
    shapeOfArrays = (noOfMasses, 1)

    proximityToYCenters = np.abs(ySpacesForEachMass - centers)
    yClosestToCenters = proximityToYCenters.min(axis=1).reshape(shapeOfArrays)
    yCentersMask = proximityToYCenters == yClosestToCenters

    v0, E0, deltaE, deltaQ = kinematicArrays

    # def selectValuesAtYCenter(kineArray):
    #     kineArray = kineArray * np.ones(shapeOfArrays)
    #     return kineArray[yCentersMask].reshape(shapeOfArrays)
    
    # v0, E0, deltaE, deltaQ = [selectValuesAtYCenter(A) for A in (v0, E0, deltaE, deltaQ)]

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

    dQdE1 = 1. - (E0 / Ef)**1.5 * L1/L0 - np.cos(angle) * \
        ((E0 / Ef)**0.5 - L1/L0 * E0/Ef)
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
    # need to pad with zeros left and right to return array with same shape
    derivative[:, 6:-6] = dev
    return derivative

def buildNcpFromSpec(par, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):
    """input: all row shape
       output: row shape with the ncpTotal for each mass"""

    if np.all(np.isnan(par)):
        return np.full(ySpacesForEachMass.shape, np.nan)
    
    ncpForEachMass, ncpTotal = calculateNcpSpec(par, masses, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)        
    return ncpForEachMass


def createNcpWorkspaces(ncpForEachMass, ws):
    """Transforms the data straight from the map and creates matrices of the fitted ncpTotal and respective workspaces"""

    ncpTotal = np.sum(ncpForEachMass, axis=1)  
    # Need to rearrage array of yspaces into seperate arrays for each mass
    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    dataX = ws.extractX()

    hist_widths = dataX[:, 1:] - dataX[:, :-1]
    dataX = dataX[:, :-1]  # Cut last column to match ncpTotal length
    dataY = ncpTotal * hist_widths

    CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), Nspec=len(dataX), 
                    OutputWorkspace=ws.name()+"_tof_fitted_profiles")

    for i, ncp_m in enumerate(ncpForEachMass):
        CreateWorkspace(DataX=dataX.flatten(), DataY=ncp_m.flatten(), Nspec=len(dataX),
                        OutputWorkspace=ws.name()+"_tof_fitted_profile_"+str(i+1))
    return ncpForEachMass, ncpTotal


def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


# spectra and verbose not used
def calculateMeanWidthsAndIntensities(widths, intensities):
    """calculates the mean widths and intensities of the Compton profile J(y) for each mass"""
    noOfMasses = len(masses)

    # Reshape arrays for broadcasting
    meanWidths = np.nanmean(widths, axis=1).reshape(noOfMasses, 1)  
    stdWidths = np.nanstd(widths, axis=1).reshape(noOfMasses, 1)

    # subtraction row by row
    width_deviation = np.abs(widths - meanWidths)
    # where True, replace by nan
    better_widths = np.where(width_deviation > stdWidths, np.nan, widths)
    better_intensities = np.where(width_deviation > stdWidths, np.nan, intensities)

    meanWidths = np.nanmean(better_widths, axis=1)  
    stdWidths = np.nanstd(better_widths, axis=1)

    # Not nansum(), to propagate nan
    normalization_sum = np.sum(better_intensities, axis=0)
    better_intensities /= normalization_sum

    meanIntensityRatios = np.nanmean(better_intensities, axis=1)
    meanIntensityRatios_std = np.nanstd(better_intensities, axis=1)

    print("\nMasses: ", masses.reshape(1, 4)[:],
          "\nMean Widths: ", meanWidths[:],
          "\nMean Intensity Ratios: ", meanIntensityRatios[:])
    return meanWidths, meanIntensityRatios


def createWorkspacesForMSCorrection(meanWidths, meanIntensityRatios, mulscatPars):
    """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""
    sample_properties = calculateSampleProperties(
        masses, meanWidths, meanIntensityRatios, "MultipleScattering", mulscatPars)
    createMulScatWorkspaces(name, sample_properties, mulscatPars)


def calculateSampleProperties(masses, meanWidths, meanIntensityRatios, mode, mulscatPars):
    """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
    or VesuvioCalculateMS"""
    masses = masses.reshape(4)
    hydrogen_peak, hydrogen_to_mass0_ratio = mulscatPars[:2]

    if mode == "GammaBackground":  # Not used for backscattering
        profiles = ""
        for m, mass in enumerate(masses):
            width, intensity = str(meanWidths[m]), str(meanIntensityRatios[m])
            profiles += "name=GaussianComptonProfile,Mass=" + \
                str(mass)+",Width="+width + \
                ",intensitiesForEachMass="+intensity+';'
        sample_properties = profiles

    elif mode == "MultipleScattering":
        if hydrogen_peak:
            # ADDITION OF THE HYDROGEN intensitiesForEachMass AS PROPORTIONAL TO A FITTED NCP (OXYGEN HERE)
            masses = np.append(masses, 1.0079)
            meanWidths = np.append(meanWidths, 5.0)
            meanIntensityRatios = np.append(
                meanIntensityRatios, hydrogen_to_mass0_ratio * meanIntensityRatios[0])
            meanIntensityRatios /= np.sum(meanIntensityRatios)

        MS_properties = np.zeros(3*len(masses))
        MS_properties[::3] = masses
        MS_properties[1::3] = meanIntensityRatios
        MS_properties[2::3] = meanWidths
        sample_properties = list(MS_properties)
    else:
        print("\n Mode entered not valid")
    print("\n The sample properties for ", mode, " are: ", sample_properties)
    return sample_properties


def createMulScatWorkspaces(ws_name, sample_properties, mulscatPars):
    """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""

    print("Evaluating the Multiple Scattering Correction.")
    transmission_guess, multiple_scattering_order, number_of_events = mulscatPars[2:]
    # selects only the masses, every 3 numbers
    MS_masses = sample_properties[::3]
    # same as above, but starts at first intensitiesForEachMass
    MS_amplitudes = sample_properties[1::3]

    dens, trans = VesuvioThickness(
        Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=transmission_guess, Thickness=0.1)

    _TotScattering, _MulScattering = VesuvioCalculateMS(
        ws_name, NoOfMasses=len(MS_masses), SampleDensity=dens.cell(9, 1),
        AtomicProperties=sample_properties, BeamRadius=2.5,
        NumScatters=multiple_scattering_order,
        NumEventsPerRun=int(number_of_events))

    data_normalisation = Integration(ws_name)
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering", "_TotScattering"):
        Divide(LHSWorkspace=workspace,
               RHSWorkspace=simulation_normalisation, OutputWorkspace=workspace)
        Multiply(LHSWorkspace=workspace,
                 RHSWorkspace=data_normalisation, OutputWorkspace=workspace)
        RenameWorkspace(InputWorkspace=workspace,
                        OutputWorkspace=str(ws_name)+workspace)
    DeleteWorkspaces(
        [data_normalisation, simulation_normalisation, trans, dens])
  # The only remaining workspaces are the _MulScattering and _TotScattering

# -------------- Other functions not used yet on main()--------------


# I tested this function but not throughouly, so could have missed something
def subtractAllMassesExceptFirst(ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    # Select all masses other than the first one
    ncpForEachMass = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    dataY, dataX, dataE = ws.extractY(), ws.extractX(), ws.extractE()

    # The original uses data_x, ie the mean points of the histograms, not dataX!
    dataY[:, :-1] -= ncpTotal * (dataX[:, 1:] - dataX[:, :-1])
    # But this makes more sense to calculate histogram widths, we can preserve one more data point
    first_ws = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(), Nspec=len(dataX))
    return first_ws


def convertWsToYSpaceAndSymetrise(ws_name, mass):
    """input: TOF workspace
       output: workspace in y-space for given mass with dataY symetrised"""

    ws_y, ws_q = ConvertToYSpace(InputWorkspace=ws_name, Mass=mass,
                                 OutputWorkspace=ws_name+"_JoY", QWorkspace=ws_name+"_Q")
    max_Y = np.ceil(2.5*mass+27)  
    # First bin boundary, width, last bin boundary, 120 bins over range
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    ws_y = Rebin(InputWorkspace=ws_y, Params=rebin_parameters,
                 FullBinsOnly=True, OutputWorkspace=ws_name+"_JoY")

    matrix_Y = ws_y.extractY()
    # safeguarding against nans as well
    matrix_Y[(matrix_Y != 0) & (matrix_Y != np.nan)] = 1
    no_y = np.nansum(matrix_Y, axis=0)

    ws_y = SumSpectra(InputWorkspace=ws_y, OutputWorkspace=ws_name+"_JoY")
    tmp = CloneWorkspace(InputWorkspace=ws_y)
    tmp.dataY(0)[:] = no_y
    tmp.dataE(0)[:] = np.zeros(tmp.blocksize())

    # use of Divide and not nanmean, err are prop automatically
    ws = Divide(LHSWorkspace=ws_y, RHSWorkspace=tmp,
                OutputWorkspace=ws_name+"_JoY")
    ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2 
    ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2 
    normalise_workspace(ws)
    return max_Y


def calculate_mantid_resolutions(ws_name, mass):
    # Only for loop in this script because the fuction VesuvioResolution takes in one spectra at a time
    # Haven't really tested this one becuase it's not modified
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/240)+","+str(max_Y)
    ws = mtd[ws_name]
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws, WorkspaceIndex=index,
                          Mass=mass, OutputWorkspaceySpacesForEachMass="tmp")
        tmp = Rebin("tmp", rebin_parameters)
        if index == 0:
            RenameWorkspace("tmp", "resolution")
        else:
            AppendSpectra("resolution", "tmp", OutputWorkspace="resolution")
    SumSpectra(InputWorkspace="resolution", OutputWorkspace="resolution")
    normalise_workspace("resolution")
    DeleteWorkspace("tmp")


def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name, RHSWorkspace="tmp_norm", OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")

main()
end_time = time.time()
print("running time: ", end_time-start_time, " seconds")
