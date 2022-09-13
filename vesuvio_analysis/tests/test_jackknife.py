import platform
from vesuvio_analysis.core_functions.run_script import runScript
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from .tests_IC import scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC
testPath = Path(__file__).absolute().parent 

np.random.seed(3)   # Set seed so that tests match everytime

class BootstrapInitialConditions:
    runBootstrap = True

    procedure = "JOINT"
    fitInYSpace = None

    bootstrapType = "JACKKNIFE"
    nSamples = 3   # Overwritten by running Jackknife
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

bootRes, noneRes = runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

jackBackSamples = bootRes["bckwdScat"].bootSamples
jackFrontSamples = bootRes["fwdScat"].bootSamples

oriJackBack = testPath / "stored_joint_jack_back.npz"
oriJackFront = testPath / "stored_joint_jack_front.npz"
if platform.system() == "Linux":
    oriJackFront = testPath / "Linux" / "jack_spec_164-175_iter_1_MS_GC_nsampl_3.npz"
    oriJackBack = testPath / "Linux" / "jack_spec_3-13_iter_0_nsampl_3.npz"

class TestJointBootstrap(unittest.TestCase):

    def setUp(self):
        self.oriJointBack = np.load(oriJackBack)["boot_samples"]
        self.oriJointFront = np.load(oriJackFront)["boot_samples"]

    def testBack(self):
        nptest.assert_array_almost_equal(jackBackSamples, self.oriJointBack)

    def testFront(self):
        nptest.assert_array_almost_equal(jackFrontSamples, self.oriJointFront)

 
