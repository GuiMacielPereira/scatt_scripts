from ..core_functions.fit_in_yspace import fitInYSpaceProcedure
from mantid.simpleapi import Load
from mantid.api import AnalysisDataService
from pathlib import Path
import numpy as np
import unittest
import numpy.testing as nptest
from .tests_IC import icWSFront, fwdIC, bootIC, yfitIC
testPath = Path(__file__).absolute().parent 

AnalysisDataService.clear()

wsFinal = Load(str(testPath / "wsFinal.nxs"))
for i in range(fwdIC.noOfMasses):
    fileName = "wsFinal_ncp_"+str(i)+".nxs"
    Load(str(testPath / fileName), 
        OutputWorkspace=wsFinal.name()+"_TOF_Fitted_Profile_"+str(i))


ySpaceFitResults = fitInYSpaceProcedure(yfitIC, fwdIC, wsFinal)

# Test yspace
np.set_printoptions(suppress=True, precision=8, linewidth=150)

oriPath = testPath / "stored_yspace_fit.npz"
storedResults = np.load(oriPath)
currentResults = ySpaceFitResults


class TestSymSumYSpace(unittest.TestCase):
    def setUp(self):
        self.oridataY = storedResults["YSpaceSymSumDataY"]
        self.oridataE = storedResults["YSpaceSymSumDataE"]

        self.optdataY = currentResults.YSpaceSymSumDataY
        self.optdataE = currentResults.YSpaceSymSumDataE
        self.rtol = 0.000001
        self.equal_nan = True
        self.decimal = 6

    def test_YSpaceDataY(self):
        nptest.assert_allclose(self.oridataY, self.optdataY)

 
    def test_YSpaceDataE(self):
        nptest.assert_allclose(self.oridataE, self.optdataE)


class TestResolution(unittest.TestCase):
    def setUp(self): 
        self.orires = storedResults["resolution"]

        self.optres = currentResults.resolution

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 8

    def test_resolution(self):
        nptest.assert_array_equal(self.orires, self.optres)


class TestHdataY(unittest.TestCase):
    def setUp(self):
        self.oriHdataY = storedResults["HdataY"]

        self.optHdataY = currentResults.HdataY

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 4

    def test_HdataY(self):
        # mask = np.isclose(self.oriHdataY, self.optHdataY, rtol=1e-9)
        # plt.imshow(mask, aspect="auto", cmap=plt.cm.RdYlGn, 
        #                 interpolation="nearest", norm=None)
        # plt.show()
        nptest.assert_array_equal(self.oriHdataY, self.optHdataY)


class TestFinalRawDataY(unittest.TestCase):
    def setUp(self):
        self.oriFinalDataY = storedResults["finalRawDataY"]

        self.optFinalDataY = currentResults.finalRawDataY

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_FinalDataY(self):
        nptest.assert_array_equal(self.oriFinalDataY, self.optFinalDataY)


class TestFinalRawDataE(unittest.TestCase):
    def setUp(self):
        self.oriFinalDataE = storedResults["finalRawDataE"]

        self.optFinalDataE = currentResults.finalRawDataE

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_HdataE(self):
        nptest.assert_array_equal(self.oriFinalDataE, self.optFinalDataE)


class Testpopt(unittest.TestCase):
    def setUp(self):
        self.oripopt = storedResults["popt"]

        # Select only Fit results due to Mantid Fit
        self.optpopt = currentResults.popt
    
    def test_opt(self):
        print("\nori:\n", self.oripopt, "\nopt:\n", self.optpopt)
        nptest.assert_array_equal(self.oripopt, self.optpopt)


class Testperr(unittest.TestCase):
    def setUp(self):
        self.oriperr = storedResults["perr"]

        self.optperr = currentResults.perr
    
    def test_perr(self):
        # print("\norierr:\n", self.oriperr, "\nopterr:\n", self.optperr)
        nptest.assert_array_equal( self.oriperr, self.optperr)



if __name__ == "__main__":
    unittest.main()