import numpy as np
from mantid.simpleapi import *
from scipy import optimize

# Format print output of arrays
np.set_printoptions(suppress=True, precision=4, linewidth=100, threshold=sys.maxsize)


def iterativeFitForDataReduction(ic):
    createTableInitialParameters(ic)

    initialWs = loadRawAndEmptyWsFromUserPath(ic)  # Do this before alternative bootstrap to extract name()   

    if ic.runningSampleWS:
        initialWs = RenameWorkspace(InputWorkspace=ic.sampleWS, OutputWorkspace=initialWs.name())

    cropedWs = cropAndMaskWorkspace(ic, initialWs)

    wsToBeFitted = CloneWorkspace(InputWorkspace=cropedWs, OutputWorkspace=cropedWs.name()+"0")
   
    createSlabGeometry(ic)  # TODO: Move this function inside the MS correction, not used otherwise

    for iteration in range(ic.noOfMSIterations):
        # Workspace from previous iteration
        wsToBeFitted = mtd[ic.name+str(iteration)]
        SumSpectra(InputWorkspace=wsToBeFitted, OutputWorkspace=wsToBeFitted.name()+"_Sum")

        fitNcpToWorkspace(ic, wsToBeFitted)
        
        meanWidths, meanIntensityRatios = createMeansAndStdTableWS(wsToBeFitted.name(), ic)
   
        # When last iteration, skip MS and GC
        if iteration == ic.noOfMSIterations - 1:
          break 

        CloneWorkspace(InputWorkspace=ic.name, OutputWorkspace="tmpNameWs")

        if ic.MSCorrectionFlag:
            wsMS = createWorkspacesForMSCorrection(ic, meanWidths, meanIntensityRatios)
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=wsMS, OutputWorkspace="tmpNameWs")

        if ic.GammaCorrectionFlag:  
            wsGC = createWorkspacesForGammaCorrection(ic, meanWidths, meanIntensityRatios)
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=wsGC, OutputWorkspace="tmpNameWs")

        RenameWorkspace(InputWorkspace="tmpNameWs", OutputWorkspace=ic.name+str(iteration+1))

        if ic.runningSampleWS and ic.runningJackknife:
            maskColumnWithZeros(ic.name, ic.name+str(iteration+1))

    wsFinal = mtd[ic.name+str(ic.noOfMSIterations - 1)]
    fittingResults = resultsObject(ic)
    fittingResults.save()
    return wsFinal, fittingResults


def maskColumnWithZeros(maskedWSName, wsToBeMaskedName):

    maskedWS = mtd[maskedWSName]
    wsToBeMasked = mtd[wsToBeMaskedName]

    dataY = maskedWS.extractY()
    dataE = maskedWS.extractE()

    zeroCol = np.all(dataE==0, axis=0)
    assert np.all(zeroCol == np.all(dataY==0, axis=0)), "Jackknife column needs to be masked in dataY and dataE"
    # zeroIdx = np.argwhere(zeroCol)

    for i in range(wsToBeMasked.getNumberHistograms()):
        wsToBeMasked.dataY(i)[zeroCol] = 0
        wsToBeMasked.dataE(i)[zeroCol] = 0
    
    # Check if this was successful



def createTableInitialParameters(ic):
    print("\nRUNNING ", ic.modeRunning, " SCATTERING.\n")
    if ic.modeRunning == "BACKWARD":
        print("\nH to first mass ratio: ", ic.HToMass0Ratio, "\n")

    meansTableWS = CreateEmptyTableWorkspace(OutputWorkspace=ic.name+"_Initial_Parameters")
    meansTableWS.addColumn(type='float', name="Mass")
    meansTableWS.addColumn(type='float', name="Initial Widths")
    meansTableWS.addColumn(type='str', name="Bounds Widths")
    meansTableWS.addColumn(type='float', name="Initial Intensities")
    meansTableWS.addColumn(type='str', name="Bounds Intensities")
    meansTableWS.addColumn(type='float', name="Initial Centers")
    meansTableWS.addColumn(type='str', name="Bounds Centers")

    print("\nCreated Table with Initial Parameters:")
    for m, iw, bw, ii, bi, inc, bc in zip(ic.masses.astype(float), ic.initPars[1::3], ic.bounds[1::3], ic.initPars[0::3], ic.bounds[0::3], ic.initPars[2::3], ic.bounds[2::3]):
        meansTableWS.addRow([m, iw, str(bw), ii, str(bi), inc, str(bc)])
        print("\nMass: ", m)
        print(f"{'Initial Intensity:':>20s} {ii:<8.3f} Bounds: {bi}")
        print(f"{'Initial Width:':>20s} {iw:<8.3f} Bounds: {bw}")
        print(f"{'Initial Center:':>20s} {inc:<8.3f} Bounds: {bc}")
    print("\n")    


def loadRawAndEmptyWsFromUserPath(ic):

    print('\nLoading local workspaces ...\n')
    Load(Filename=str(ic.userWsRawPath), OutputWorkspace=ic.name+"raw")
    Rebin(InputWorkspace=ic.name+'raw', Params=ic.tof_binning,
          OutputWorkspace=ic.name+'raw')
    SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
    wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+"uncroped_unmasked")

    # if ic.mode=="DoubleDifference":
    if ic.subEmptyFromRaw:
        Load(Filename=str(ic.userWsEmptyPath), OutputWorkspace=ic.name+"empty")
        Rebin(InputWorkspace=ic.name+'empty', Params=ic.tof_binning,
            OutputWorkspace=ic.name+'empty')

        if (type(ic.scaleEmpty)==float) | (type(ic.scaleEmpty)==int):
            Scale(InputWorkspace=ic.name+'empty', OutputWorkspace=ic.name+'empty', Factor=str(ic.scaleEmpty))
        elif ic.scaleEmpty == None:
            pass
        else:
            raise ValueError("Scaling factor fot empty workspace not recognized.")

        SumSpectra(InputWorkspace=ic.name+'empty', OutputWorkspace=ic.name+'empty'+'_sum')
        
        wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty',
                            OutputWorkspace=ic.name+"uncroped_unmasked")
    return wsToBeFitted


def cropAndMaskWorkspace(ic, ws):
    """Returns cloned and cropped workspace with modified name"""
    # Read initial Spectrum number
    wsFirstSpec = ws.getSpectrumNumbers()[0]
    assert ic.firstSpec >= wsFirstSpec, "Can't crop workspace, firstSpec < first spectrum in workspace."
    
    initialIdx = ic.firstSpec - wsFirstSpec
    lastIdx = ic.lastSpec - wsFirstSpec
    
    newWsName = ws.name().split("uncroped")[0]  # Retrieve original name
    cropedWs = CropWorkspace(
        InputWorkspace=ws, 
        StartWorkspaceIndex=initialIdx, EndWorkspaceIndex=lastIdx, 
        OutputWorkspace=newWsName
        )
    MaskDetectors(Workspace=cropedWs, WorkspaceIndexList=ic.maskedDetectorIdx)
    return cropedWs


def createSlabGeometry(ic):
    half_height, half_width, half_thick = 0.5*ic.vertical_width, 0.5*ic.horizontal_width, 0.5*ic.thickness
    xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"

    if ic.runningSampleWS and ic.runningJackknife:
        CreateSampleShape(ic.parentWS, xml_str)
    else:
        CreateSampleShape(ic.name, xml_str)


def fitNcpToWorkspace(ic, ws):
    """
    Performs the fit of ncp to the workspace.
    Firtly the arrays required for the fit are prepared and then the fit is performed iteratively
    on a spectrum by spectrum basis.
    """
    dataYws, dataXws, dataEws = arraysFromWS(ws)   
    dataY, dataX, dataE = histToPointData(dataYws, dataXws, dataEws)      

    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(ic, dataX)
    
    print("\nFitting NCP:\n")

    arrFitPars = fitNcpToArray(ic, dataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)
    createTableWSForFitPars(ws.name(), ic.noOfMasses, arrFitPars)
    arrBestFitPars = arrFitPars[:, 1:-2]
    allNcpForEachMass, allNcpTotal = calculateNcpArr(ic, arrBestFitPars, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)
    createNcpWorkspaces(allNcpForEachMass, allNcpTotal, ws, ic)
    return


def arraysFromWS(ws):
    """Output: dataY, dataX and dataE as arrays"""
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    return dataY, dataX, dataE


def histToPointData(dataY, dataX, dataE):
    """Output: middle points of dataX hists"""

    histWidths = dataX[:, 1:] - dataX[:, :-1]
    assert np.min(histWidths) == np.max(histWidths), "Histogram widhts need to be the same length"
    
    dataYp = dataY[:, :-1]
    dataEp = dataE[:, :-1] 
    dataXp = dataX[:, :-1] + histWidths[0, 0]/2 

    # dataYp, dataXp, dataEp = filterNanColumns(dataYp, dataXp, dataEp)

    return dataYp, dataXp, dataEp


# def filterNanColumns(dataY, dataX, dataE):
#     zeroCol = np.all(dataY == 0, axis=0)   # When whole column is zero, take it out of point data

#     dataYf = dataY[:, ~zeroCol]
#     dataXf = dataX[:, ~zeroCol]
#     dataEf = dataE[:, ~zeroCol]
#     return dataYf, dataXf, dataEf

# def jackSampleFromPointData(dataY, dataX, dataE, j):
#     jackDataY = np.delete(dataY, j, axis=1)
#     jackDataX = np.delete(dataX, j, axis=1)
#     jackDataE = np.delete(dataE, j, axis=1)
#     return jackDataY, jackDataX, jackDataE


def prepareFitArgs(ic, dataX):
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
    # For spec no below 135, back scattering detectors, mode is double difference
    # For spec no 135 or above, front scattering detectors, mode is single difference
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
    """
    Exchanges the first two axes of an array A.
    Rearranges array to match iteration per spectrum
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertDataXToYSpacesForEachMass(dataX, masses, delta_Q, delta_E):
    "Calculates y spaces from TOF data, each row corresponds to one mass" 
    
    #prepare arrays to broadcast
    dataX = dataX[np.newaxis, :, :]
    delta_Q = delta_Q[np.newaxis, :, :]
    delta_E = delta_E[np.newaxis, :, :]  

    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    masses = masses.reshape(masses.size, 1, 1)

    energyRecoil = np.square( hbar * delta_Q ) / 2. / masses              
    ySpacesForEachMass = masses / hbar**2 /delta_Q * (delta_E - energyRecoil)    #y-scaling  
    return ySpacesForEachMass


def fitNcpToArray(ic, dataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass):
    """Takes dataY as a 2D array and returns the 2D array best fit parameters."""

    arrFitPars = np.zeros((len(dataY), len(ic.initPars)+3))
    for i in range(len(dataY)):

        specFitPars = fitNcpToSingleSpec(
            dataY[i],
            dataE[i],
            ySpacesForEachMass[i],
            resolutionPars[i],
            instrPars[i],
            kinematicArrays[i],
            ic
            ) 

        arrFitPars[i] = specFitPars

        if np.all(specFitPars==0):
            print("Skipped spectra.")
        else:
            print(f"Fitted spectra {int(specFitPars[0]):3}")
    
    assert ~np.all(arrFitPars==0), "Either Fits are all zero or assignment of fitting not working"
    return arrFitPars


def createTableWSForFitPars(wsName, noOfMasses, arrFitPars):
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsName+"_Best_Fit_NCP_Parameters")
    tableWS.setTitle("SCIPY Fit")
    tableWS.addColumn(type='float', name="Spec Idx")
    for i in range(int(noOfMasses)):
        tableWS.addColumn(type='float', name=f"Intensity {i}")
        tableWS.addColumn(type='float', name=f"Width {i}")
        tableWS.addColumn(type='float', name=f"Center {i}")
    tableWS.addColumn(type='float', name="Norm Chi2")
    tableWS.addColumn(type='float', name="No Iter")

    for row in arrFitPars:    # Pass array onto table ws
        tableWS.addRow(row)
    return 


def calculateNcpArr(ic, arrBestFitPars, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass):
    """Calculates the matrix of NCP from matrix of best fit parameters"""

    allNcpForEachMass = []
    for i in range(len(arrBestFitPars)):

        ncpForEachMass = calculateNcpRow(
            arrBestFitPars[i],
            ySpacesForEachMass[i], 
            resolutionPars[i], 
            instrPars[i], 
            kinematicArrays[i],
            ic
            )
            
        allNcpForEachMass.append(ncpForEachMass)

    allNcpForEachMass = np.array(allNcpForEachMass)
    allNcpTotal = np.sum(allNcpForEachMass, axis=1)        
    return allNcpForEachMass, allNcpTotal


def calculateNcpRow(initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """input: all row shape
       output: row shape with the ncpTotal for each mass"""

    if np.all(initPars==0):  
        return np.zeros(ySpacesForEachMass.shape) 
    
    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)        
    return ncpForEachMass


def createNcpWorkspaces(ncpForEachMass, ncpTotal, ws, ic):
    """Creates workspaces from ncp array data"""

    # Need to rearrage array of yspaces into seperate arrays for each mass
    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)

    # Use ws dataX to match with histogram data
    dataX = ws.extractX()[:, :-1]

    # if ic.runningJackknife:
    #     dataX = np.delete(dataX, ic.jackIter, axis=1)

    assert ncpTotal.shape == dataX.shape, "DataX and DataY in ws need to be the same shape."

    # Total ncp workspace

    # # Add zeros column
    # ncpTotalf = addZeroCol(ncpTotal, ws)
    # assert ncpTotalf.shape == dataX.shape, "DataX and DataY in ws need to be the same shape."
    ncpTotalf = ncpTotal

    ncpTotWs = CreateWorkspace(
        DataX=dataX.flatten(), 
        DataY=ncpTotalf.flatten(),
        Nspec=len(dataX), 
        OutputWorkspace=ws.name()+"_TOF_Fitted_Profiles")
    SumSpectra(InputWorkspace=ncpTotWs, OutputWorkspace=ncpTotWs.name()+"_Sum" )

    # Individual ncp workspaces
    for i, ncp_m in enumerate(ncpForEachMass):

        # ncp_mf = addZeroCol(ncp_m, ws)
        ncp_mf = ncp_m

        ncpMWs = CreateWorkspace(
            DataX=dataX.flatten(), 
            DataY=ncp_mf.flatten(), 
            Nspec=len(dataX),
            OutputWorkspace=ws.name()+"_TOF_Fitted_Profile_"+str(i))
        SumSpectra(InputWorkspace=ncpMWs, OutputWorkspace=ncpMWs.name()+"_Sum" )


# def addZeroCol(ncp, ws):
#     ncpf = ws.extractY()[:, :-1]
#     zeroCol = np.all(ncpf==0, axis=0)
#     ncpf[:, ~zeroCol] = ncp
#     return ncpf
  

def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def createMeansAndStdTableWS(wsName, ic):
    # Extract widths and intensities from tableWorkspace
    fitParsTable = mtd[wsName+"_Best_Fit_NCP_Parameters"]
    widths = np.zeros((ic.noOfMasses, fitParsTable.rowCount()))
    intensities = np.zeros(widths.shape)
    for i in range(ic.noOfMasses):
        widths[i] = fitParsTable.column(f"Width {i}")
        intensities[i] = fitParsTable.column(f"Intensity {i}")

    assert len(widths) == ic.noOfMasses, "Widths and intensities must be in shape (noOfMasses, noOfSpec)"

    meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios = calculateMeansAndStds(widths, intensities)

    meansTableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsName+"_Mean_Widths_And_Intensities")
    meansTableWS.addColumn(type='float', name="Mass")
    meansTableWS.addColumn(type='float', name="Mean Widths")
    meansTableWS.addColumn(type='float', name="Std Widths")
    meansTableWS.addColumn(type='float', name="Mean Intensities")
    meansTableWS.addColumn(type='float', name="Std Intensities")

    print("\nCreated Table with means and std:")
    print("\nMass    Mean \u00B1 Std Widths    Mean \u00B1 Std Intensities\n")
    for m, mw, stdw, mi, stdi in zip(ic.masses.astype(float), meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios):
        meansTableWS.addRow([m, mw, stdw, mi, stdi])
        print(f"{m:5.2f}  {mw:10.5f} \u00B1 {stdw:7.5f}  {mi:10.5f} \u00B1 {stdi:7.5f}")
    print("\n")
    return meanWidths, meanIntensityRatios


def calculateMeansAndStds(widthsIn, intensitiesIn):

    betterWidths, betterIntensities = filterWidthsAndIntensities(widthsIn, intensitiesIn)
    
    meanWidths = np.nanmean(betterWidths, axis=1)  
    stdWidths = np.nanstd(betterWidths, axis=1)

    meanIntensityRatios = np.nanmean(betterIntensities, axis=1)
    stdIntensityRatios = np.nanstd(betterIntensities, axis=1)

    return meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios


def filterWidthsAndIntensities(widthsIn, intensitiesIn):
    """Puts nans in places to be ignored"""

    widths = widthsIn.copy()      # Copy to avoid accidental changes in arrays
    intensities = intensitiesIn.copy()

    zeroSpecs = np.all(widths==0, axis=0)   # Catches all failed fits, not just masked spectra
    widths[:, zeroSpecs] = np.nan
    intensities[:, zeroSpecs] = np.nan

    meanWidths = np.nanmean(widths, axis=1)[:, np.newaxis]  

    widthDeviation = np.abs(widths - meanWidths)
    stdWidths = np.nanstd(widths, axis=1)[:, np.newaxis]  

    # Put nan in places where width deviation is bigger than std
    filterMask = widthDeviation > stdWidths
    betterWidths = np.where(filterMask, np.nan, widths)
    
    betterIntensities = np.where(filterMask, np.nan, intensities)
    betterIntensities = betterIntensities / np.sum(betterIntensities, axis=0)   # Not nansum()

    assert np.all(meanWidths!=np.nan), "At least one mean of widths is nan!"
    assert np.sum(filterMask) >= 1, "No widths satisfy filtering condition"
    assert not(np.all(np.isnan(betterWidths))), "All filtered widths are nan"
    assert not(np.all(np.isnan(betterIntensities))), "All filtered intensities are nan"
    assert np.nanmax(betterWidths) != np.nanmin(betterWidths), f"All fitered widths have the same value: {np.nanmin(betterWidths)}"
    assert np.nanmax(betterIntensities) != np.nanmin(betterIntensities), f"All fitered widths have the same value: {np.nanmin(betterIntensities)}"
   
    return betterWidths, betterIntensities


def fitNcpToSingleSpec(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Fits the NCP and returns the best fit parameters for one spectrum"""

    if np.all(dataY == 0) : 
        return np.zeros(len(ic.initPars)+3)  

    result = optimize.minimize(
        errorFunction, 
        ic.initPars, 
        args=(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic),
        method='SLSQP', 
        bounds = ic.bounds, 
        constraints=ic.constraints
        )

    fitPars = result["x"]

    noDegreesOfFreedom = len(dataY) - len(fitPars)
    specFitPars = np.append(instrPars[0], fitPars)
    return np.append(specFitPars, [result["fun"] / noDegreesOfFreedom, result["nit"]])


def errorFunction(pars, dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Error function to be minimized, operates in TOF space"""

    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, pars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)

    # Additional treatement for jackknife
    zerosMask = dataY==0
    ncpTotal = ncpTotal[~zerosMask]
    dataYf = dataY[~zerosMask]   
    dataEf = dataE[~zerosMask]   

    # dataYf = dataY
    # dataEf = dataE


    if np.all(dataE == 0) | np.all(np.isnan(dataE)):
        # This condition is currently never satisfied, 
        # but I am keeping it for the unlikely case of fitting NCP data without errors.
        # In this case, we can use a statistical weight to make sure 
        # chi2 is not too small for minimize.optimize().
        chi2 = (ncpTotal - dataYf)**2 / dataYf**2
    else:
        chi2 =  (ncpTotal - dataYf)**2 / dataEf**2    
    return np.sum(chi2)


def calculateNcpSpec(ic, pars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):    
    """Creates a synthetic C(t) to be fitted to TOF values of a single spectrum, from J(y) and resolution functions
       Shapes: datax (1, n), ySpacesForEachMass (4, n), res (4, 2), deltaQ (1, n), E0 (1,n),
       where n is no of bins"""
    
    masses, intensities, widths, centers = prepareArraysFromPars(ic, pars) 
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

    masses = ic.masses[:, np.newaxis]    
    intensities = initPars[::3].reshape(masses.shape)
    widths = initPars[1::3].reshape(masses.shape)
    centers = initPars[2::3].reshape(masses.shape)  
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

    shapeOfArrays = centers.shape
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
    assert masses.shape == (masses.size, 1), f"masses.shape: {masses.shape}. The shape of the masses array needs to be a collumn!"

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
    assert masses.shape == (masses.size, 1), "The shape of the masses array needs to be a collumn!"
        
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
    pseudo_voigt = eta * lorentizian(x, gamma_v) + (1.-eta) * gaussian(x, sigma_v)
    # TODO: Ask about this comment
    # norm = np.sum(pseudo_voigt)*(x[1]-x[0])
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


def createWorkspacesForMSCorrection(ic, meanWidths, meanIntensityRatios):
    """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""

    sampleProperties = calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios)
    print("\nThe sample properties for Multiple Scattering correction are:\n\n", 
            sampleProperties, "\n")
    
    if ic.runningSampleWS and ic.runningJackknife:
        return createMulScatWorkspaces(ic, ic.parentWS.name(), sampleProperties)
    else:
        return createMulScatWorkspaces(ic, ic.name, sampleProperties)


def calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios):
    masses = ic.masses.flatten()

    # If Backsscattering mode and H is present in the sample, add H to MS properties
    if (ic.modeRunning == "BACKWARD"):
        if (ic.HToMass0Ratio != None):  # If H is present, ratio is a number
            masses = np.append(masses, 1.0079)
            meanWidths = np.append(meanWidths, 5.0)
            # HIntensity = ic.HToMass0Ratio * meanIntensityRatios[0]
            HIntensity = ic.HToMass0Ratio * meanIntensityRatios[ic.HRatioMassIdx]
            meanIntensityRatios = np.append(meanIntensityRatios, HIntensity)
            meanIntensityRatios /= np.sum(meanIntensityRatios)

    MSProperties = np.zeros(3*len(masses))
    MSProperties[::3] = masses
    MSProperties[1::3] = meanIntensityRatios
    MSProperties[2::3] = meanWidths
    sampleProperties = list(MSProperties)   

    return sampleProperties


def createMulScatWorkspaces(ic, wsName, sampleProperties):
    """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""

    print("\nEvaluating the Multiple Scattering Correction...\n")
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
        SumSpectra(wsName+workspace, OutputWorkspace=wsName+workspace+"_Sum")
        
    DeleteWorkspaces(
        [data_normalisation, simulation_normalisation, trans, dens]
        )
    # The only remaining workspaces are the _MulScattering and _TotScattering
    return mtd[wsName+"_MulScattering"]


def createWorkspacesForGammaCorrection(ic, meanWidths, meanIntensityRatios):
    """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

    if ic.runningSampleWS and ic.runningJackknife:
        inputWS = ic.parentWS.name()
    else:
        inputWS = ic.name

    # I do not know why, but setting these instrument parameters is required
    SetInstrumentParameter(inputWS, ParameterName='hwhm_lorentz', 
                            ParameterType='Number', Value='24.0')
    SetInstrumentParameter(inputWS, ParameterName='sigma_gauss', 
                            ParameterType='Number', Value='73.0')

    profiles = calcGammaCorrectionProfiles(ic.masses, meanWidths, meanIntensityRatios)

    background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=inputWS, ComptonFunction=profiles)
    
    RenameWorkspace(InputWorkspace= background, OutputWorkspace = inputWS+"_Gamma_Background")
    Scale(InputWorkspace = inputWS+"_Gamma_Background", OutputWorkspace = inputWS+"_Gamma_Background", 
        Factor=0.9, Operation="Multiply")
    DeleteWorkspace(corrected)
    return mtd[inputWS+"_Gamma_Background"]


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


class resultsObject:
    """Used to collect results from workspaces and store them in .npz files for testing."""
    def __init__(self, ic):

        allIterNcp = []
        allFitWs = []
        allTotNcp = []
        allBestPar = []
        allMeanWidhts = []
        allMeanIntensities = []
        allStdWidths = []
        allStdIntensities = []
        j=0
        while True:
            try:
                wsIterName = ic.name+str(j)

                # Extract ws that were fitted
                ws = mtd[wsIterName]
                allFitWs.append(ws.extractY())

                # Extract total ncp
                totNcpWs = mtd[wsIterName+"_TOF_Fitted_Profiles"]
                allTotNcp.append(totNcpWs.extractY())

                # Extract best fit parameters
                fitParTable = mtd[wsIterName+"_Best_Fit_NCP_Parameters"]
                bestFitPars = []
                for key in fitParTable.keys():
                    bestFitPars.append(fitParTable.column(key))
                allBestPar.append(np.array(bestFitPars).T)
                
                # Extract individual ncp 
                allNCP = []
                i = 0
                while True:   # By default, looks for all ncp ws until it breaks
                    try:
                        ncpWsToAppend = mtd[wsIterName+"_TOF_Fitted_Profile_"+str(i)]
                        allNCP.append(ncpWsToAppend.extractY())
                        i += 1
                    except KeyError:
                        break
                allNCP = switchFirstTwoAxis(np.array(allNCP))
                allIterNcp.append(allNCP)
                
                # Extract Mean and Std Widths, Intensities
                meansTable = mtd[wsIterName + "_Mean_Widths_And_Intensities"]
                allMeanWidhts.append(meansTable.column("Mean Widths"))
                allStdWidths.append(meansTable.column("Std Widths"))
                allMeanIntensities.append(meansTable.column("Mean Intensities"))
                allStdIntensities.append(meansTable.column("Std Intensities"))  
                
                j+=1
            except KeyError:
                break

        self.all_fit_workspaces = np.array(allFitWs)
        self.all_spec_best_par_chi_nit = np.array(allBestPar)
        self.all_tot_ncp = np.array(allTotNcp)
        self.all_ncp_for_each_mass = np.array(allIterNcp)

        self.all_mean_widths = np.array(allMeanWidhts)
        self.all_mean_intensities = np.array(allMeanIntensities)
        self.all_std_widths = np.array(allStdWidths)
        self.all_std_intensities = np.array(allStdIntensities)

        # Pass all attributes of ic into attributes to be used whithin this object
        self.maskedDetectorIdx = ic.maskedDetectorIdx
        self.masses = ic.masses
        self.noOfMasses = ic.noOfMasses
        self.resultsSavePath = ic.resultsSavePath


    def save(self):
        """Saves all of the arrays stored in this object"""

        # TODO: Take out nans next time when running original results
        # Because original results were recently saved with nans, mask spectra with nans
        self.all_spec_best_par_chi_nit[:, self.maskedDetectorIdx, :] = np.nan
        self.all_ncp_for_each_mass[:, self.maskedDetectorIdx, :, :] = np.nan
        self.all_tot_ncp[:, self.maskedDetectorIdx, :] = np.nan

        savePath = self.resultsSavePath
        np.savez(savePath,
                 all_fit_workspaces=self.all_fit_workspaces,
                 all_spec_best_par_chi_nit=self.all_spec_best_par_chi_nit,
                 all_mean_widths=self.all_mean_widths,
                 all_mean_intensities=self.all_mean_intensities,
                 all_std_widths=self.all_std_widths,
                 all_std_intensities=self.all_std_intensities,
                 all_tot_ncp=self.all_tot_ncp,
                 all_ncp_for_each_mass=self.all_ncp_for_each_mass)

           