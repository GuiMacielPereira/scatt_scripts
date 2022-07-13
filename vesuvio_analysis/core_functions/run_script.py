
from vesuvio_analysis.core_functions.ICHelpers import buildFinalWSNames, completeICFromInputs, completeBootIC, completeYFitIC
from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, runJointBackAndForwardProcedure, isHPresent
from mantid.api import mtd


def runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC):

    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    completeICFromInputs(bckwdIC, scriptName, wsBackIC)
    completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC) 
    completeYFitIC(yFitIC, scriptName)
    
    checkInputs(userCtr)


    if userCtr.procedure == None:
         def runProcedure():
            return None 

    elif userCtr.procedure == "BACKWARD":
        if isHPresent(fwdIC.masses) and (bckwdIC.HToMassIdxRatio==None) and (bckwdIC.noOfMSIterations>0):
            raise ValueError ("""
                Tried to run independent backscattering with MS corrections:
                Need a known value for HToMassIdxRatio.
                Run procedure='JOINT' first to activate an estimation for this value.
                """)

        def runProcedure():
            return runIndependentIterativeProcedure(bckwdIC)

    elif userCtr.procedure == "FORWARD":
        def runProcedure():
            return runIndependentIterativeProcedure(fwdIC)
    
    elif userCtr.procedure == "JOINT":
        def runProcedure():
            return runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    else:
        raise ValueError("Procedure option not recognized.")


    if userCtr.fitInYSpace == None:
        wsNames = []
        ICs = []

    elif userCtr.fitInYSpace == "BACKWARD":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD"], [bckwdIC])
        ICs = [bckwdIC]

    elif userCtr.fitInYSpace == "FORWARD":
        wsNames = buildFinalWSNames(scriptName, ["FORWARD"], [fwdIC])
        ICs = [fwdIC]

    elif userCtr.fitInYSpace == "JOINT":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD", "FORWARD"], [bckwdIC, fwdIC])
        ICs = [bckwdIC, fwdIC]
    else:
        raise ValueError("fitInYSpace option not recognized.")


    # If bootstrap is not None, run bootstrap procedure and finish
    if bootIC.runBootstrap == True:
        assert (bootIC.procedure=="FORWARD") | (bootIC.procedure=="BACKWARD") | (bootIC.procedure=="JOINT"), "Invalid Bootstrap procedure."
        return runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC), None
    
    # Default workflow for procedure + fit in y space

    # Check if final ws are loaded:
    wsInMtd = [ws in mtd for ws in wsNames]   # List of bool

    # If final ws are already loaded
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



def checkInputs(userCtr):
    if (userCtr.procedure=="BACKWARD") | (userCtr.procedure=="FORWARD"):
        if userCtr.fitInYSpace != None:
            assert userCtr.procedure == userCtr.fitInYSpace, "For isolated forward and backward, procedure needs to match fitInYSpace."
    # TODO: Add some more checks 