
from vesuvio_analysis.core_functions.ICHelpers import buildFinalWSName, completeICFromInputs, completeBootIC, completeYFitIC
from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, runJointBackAndForwardProcedure, runPreProcToEstHRatio, createTableWSHRatios, isHPresent
from mantid.api import mtd


def runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC):

    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    completeICFromInputs(bckwdIC, scriptName, wsBackIC)
    completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC) 
    completeYFitIC(yFitIC, scriptName)
    
    checkInputs(userCtr, bootIC)

    def runProcedure():
        proc = userCtr.procedure  # Shorthad to make it easier to read

        if proc==None:
            return
        
        ranPreliminary = False
        if (proc=="BACKWARD") | (proc=="JOINT"):
            if isHPresent(fwdIC.masses) & (bckwdIC.HToMassIdxRatio==None):
                HRatios, massIdxs = runPreProcToEstHRatio(bckwdIC, fwdIC)   # Sets H ratio to bckwdIC automatically
                ranPreliminary = True
            assert (isHPresent(fwdIC.masses) != (bckwdIC.HToMassIdxRatio==None)), "When H is not present, HToMassIdxRatio has to be set to None"

        if (proc=="BACKWARD"): res = runIndependentIterativeProcedure(bckwdIC)
        if (proc=="FORWARD"): res = runIndependentIterativeProcedure(fwdIC)
        if (proc=="JOINT"): res = runJointBackAndForwardProcedure(bckwdIC, fwdIC)

        # If preliminary procedure ran, make TableWS with H ratios values
        if ranPreliminary: createTableWSHRatios(HRatios, massIdxs)
        return res

    # Names of workspaces to be fitted in y space
    wsNames = []
    ICs = []
    for mode, IC in zip(["BACKWARD", "FORWARD"], [bckwdIC, fwdIC]):
        if (userCtr.fitInYSpace==mode) | (userCtr.fitInYSpace=="JOINT"):
            wsNames.append(buildFinalWSName(scriptName, mode, IC))
            ICs.append(IC)


    # If bootstrap is not None, run bootstrap procedure and finish
    if bootIC.runBootstrap:
        assert (bootIC.procedure=="FORWARD") | (bootIC.procedure=="BACKWARD") | (bootIC.procedure=="JOINT"), "Invalid Bootstrap procedure."
        return runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC), None
    
    # Default workflow for procedure + fit in y space
    if userCtr.runRoutine:
        # Check if final ws are loaded:
        wsInMtd = [ws in mtd for ws in wsNames]     # Bool list
        if (len(wsInMtd)>0) and all(wsInMtd):       # When wsName is empty list, loop doesn't run
            for wsName, IC in zip(wsNames, ICs):  
                resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
            return None, resYFit       # To match return below. 
        
        checkUserClearWS()      # Check if user is OK with cleaning all workspaces
        res = runProcedure()

        resYFit = None
        for wsName, IC in zip(wsNames, ICs):
            resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
        
        return res, resYFit   # Return results used only in tests


def checkUserClearWS():
    """If any workspace is loaded, check if user is sure to start new procedure."""

    if len(mtd) != 0:
        userInput = input("This action will clean all current workspaces to start anew. Proceed? (y/n): ")
        if (userInput == "y") | (userInput == "Y"):
            pass
        else:
            raise KeyboardInterrupt("Run of procedure canceled.")
    return



def checkInputs(userCtr, bootIC):
    
    for flag in [userCtr.procedure, userCtr.fitInYSpace, bootIC.procedure, bootIC.fitInYSpace]:
        assert (flag=="BACKWARD") | (flag=="FORWARD") | (flag=="JOINT"), "Option not recognized."

    for inIC in [userCtr, bootIC]:
        if inIC.procedure != "JOINT":
            assert inIC.procedure == inIC.fitInYSpace

    if userCtr.runRoutine & bootIC.runBootstrap: 
        raise ValueError ("""
            Script is set to run both the main routine and bootstrap.
            Please select only one of those at a time.
            """)
