import platform
from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.core_functions.run_script import runScript
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from .tests_IC import  scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC
testPath = Path(__file__).absolute().parent 

np.random.seed(1)   # Set seed so that tests match everytime

oriBootBack = testPath / "stored_boot_back.npz"
oriBootFront = testPath / "stored_boot_front.npz"
oriBootYFit = testPath / "stored_boot_yfit.npz"
if platform.system() == "Linux":
    oriBootFront = testPath / "Linux" / "boot_spec_164-175_iter_1_MS_GC_nsampl_3.npz"
    oriBootYFit = testPath / "Linux" / "boot_spec_164-175_iter_1_MS_GC_ySpaceFit_nsampl_3.npz"
    oriBootBack = testPath / "Linux" / "boot_spec_3-13_iter_0_nsampl_3.npz"


class BootstrapInitialConditions:
    runBootstrap = True

    procedure = "JOINT"
    fitInYSpace = "FORWARD"

    bootstrapType = "BOOT_RESIDUALS" 
    nSamples = 3
    skipMSIterations = False
    runningTest = True
    userConfirmation = False

class UserScriptControls:
    runRoutine = False 
    procedure = "FORWARD"   
    fitInYSpace = None    
    # bootstrap = "JOINT"   

bootIC = BootstrapInitialConditions
userCtr = UserScriptControls

# Change yFItIC to default settings, running tests for yfit before hand changes this
yFitIC.fitModel = "SINGLE_GAUSSIAN"
yFitIC.symmetrisationFlag = True

bootRes, noneRes = runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

#TODO: Figure out why doing the two tests simultaneously fails the testing
# Probably because running bootstrap alters the initial conditions of forward scattering
# Test Joint procedure
bootBackSamples = bootRes["bckwdScat"].bootSamples
bootFrontSamples = bootRes["fwdScat"].bootSamples
bootYFitSamples = bootRes["fwdYFit"].bootSamples

class TestJointBootstrap(unittest.TestCase):

    def setUp(self):
        self.oriJointBack = np.load(oriBootBack)["boot_samples"]
        self.oriJointFront = np.load(oriBootFront)["boot_samples"]
        self.oriJointYFit = np.load(oriBootYFit)["boot_samples"]

    def testBack(self):
        nptest.assert_array_almost_equal(bootBackSamples, self.oriJointBack)

    def testFront(self):
        nptest.assert_array_almost_equal(bootFrontSamples, self.oriJointFront)

    def testYFit(self):
        nptest.assert_array_almost_equal(bootYFitSamples, self.oriJointYFit)


# # Test Single procedure
# bootSingleResults = runIndependentBootstrap(bckwdIC, bootIC, yfitIC)

# bootSingleBackSamples = bootSingleResults[0].bootSamples
# bootSingleYFitSamples = bootSingleResults[1].bootSamples

# oriSingleBootBack = testPath / "stored_single_boot_back.npz"
# oriSingleBootYFit = testPath / "stored_single_boot_back_yfit.npz"

# class TestIndependentBootstrap(unittest.TestCase):
#     def setUp(self):
#         self.oriBack = np.load(oriSingleBootBack)["boot_samples"]
#         self.oriYFit = np.load(oriSingleBootYFit)["boot_vals"]

#     def testBack(self):
#         nptest.assert_array_almost_equal(bootSingleBackSamples, self.oriBack)

#     def testYFit(self):
#         nptest.assert_array_almost_equal(bootSingleYFitSamples, self.oriYFit)


