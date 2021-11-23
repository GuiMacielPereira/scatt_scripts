import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=8, linewidth=150)
# plt.style.use('dark_background')

currentPath = Path(__file__).absolute().parent  # Path to the repository

forward = False
if forward:
    pathToOriginal = currentPath / "cleaning" / "starter_forward.npz" 
    pathToOptimized = currentPath / "cleaning" / "current_forward.npz" 
else:
    pathToOriginal = currentPath / "cleaning" / "starter_backward.npz" 
    pathToOptimized = currentPath / "cleaning" / "current_backward.npz" 

class TestFitParameters(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        oriPars = originalResults["all_spec_best_par_chi_nit"][0]
        self.orispec = oriPars[:, 0]
        self.orichi2 = oriPars[:, -2]
        self.orinit = oriPars[:, -1]
        self.orimainPars = oriPars[:, 1:-2]
        self.oriintensities = self.orimainPars[:, 0::3]
        self.oriwidths = self.orimainPars[:, 1::3]
        self.oricenters = self.orimainPars[:, 2::3]

        optimizedResults = np.load(pathToOptimized)
        optPars = optimizedResults["all_spec_best_par_chi_nit"][0]
        self.optspec = optPars[:, 0]
        self.optchi2 = optPars[:, -2]
        self.optnit = optPars[:, -1]
        self.optmainPars = optPars[:, 1:-2]
        self.optintensities = self.optmainPars[:, 0::3]
        self.optwidths = self.optmainPars[:, 1::3]
        self.optcenters = self.optmainPars[:, 2::3]

        self.rtol = 0.0001
        self.equal_nan = True

    def test_mainPars(self):
        nptest.assert_array_equal(self.orimainPars, self.optmainPars)

    def test_chi2(self):
        nptest.assert_array_equal(self.orichi2, self.optchi2)

    def test_nit(self):
        nptest.assert_array_equal(self.orinit, self.optnit)

    def test_intensities(self):
        nptest.assert_array_equal(self.oriintensities, self.optintensities)


def displayMask(mask, rtol, string):
    noDiff = np.sum(mask)
    maskSize = mask.size
    print("\nNo of different "+string+f", rtol={rtol}:\n",
        noDiff, " out of ", maskSize,
        f"ie {100*noDiff/maskSize:.1f} %")    


class TestNcp(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orincp = originalResults["all_tot_ncp"][0]
        
        optimizedResults = np.load(pathToOptimized)
        self.optncp = optimizedResults["all_tot_ncp"][0]

        self.rtol = 0.001
        self.equal_nan = True

    def test_ncp(self):
        nptest.assert_array_equal(self.orincp, self.optncp)


class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanwidths = originalResults["all_mean_widths"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanwidths = optimizedResults["all_mean_widths"][0]
    
    def test_widths(self):
        nptest.assert_array_equal(self.orimeanwidths, self.optmeanwidths)


class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanintensities = originalResults["all_mean_intensities"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanintensities = optimizedResults["all_mean_intensities"][0]
    
    def test_intensities(self):
        nptest.assert_array_equal(self.orimeanintensities, self.optmeanintensities)


class TestFitWorkspaces(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriws = originalResults["all_fit_workspaces"][0]
        
        optimizedResults = np.load(pathToOptimized)
        self.optws = optimizedResults["all_fit_workspaces"][0]

        self.decimal = 8

    def test_ws(self):
        nptest.assert_array_equal(self.optws, self.oriws)


class TestSymSumYSpace(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oridataY = originalResults["YSpaceSymSumDataY"]
        self.oridataE = originalResults["YSpaceSymSumDataE"]

        optimizedResults = np.load(pathToOptimized)
        self.optdataY = optimizedResults["YSpaceSymSumDataY"]
        self.optdataE = optimizedResults["YSpaceSymSumDataE"]
        self.rtol = 0.000001
        self.equal_nan = True
        self.decimal = 6

    def test_YSpaceDataY(self):
        nptest.assert_array_equal(self.oridataY, self.optdataY)

 
    def test_YSpaceDataE(self):
        nptest.assert_array_equal(self.oridataE, self.optdataE)


class TestResolution(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orires = originalResults["resolution"]

        optimizedResults = np.load(pathToOptimized)
        self.optres = optimizedResults["resolution"]

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 8

    def test_resolution(self):
        nptest.assert_array_equal(self.orires, self.optres)


class TestHdataY(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriHdataY = originalResults["HdataY"]

        optimizedResults = np.load(pathToOptimized)
        self.optHdataY = optimizedResults["HdataY"]

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 4

    def test_HdataY(self):
        nptest.assert_array_equal(self.oriHdataY, self.optHdataY)


class TestFinalRawDataY(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriFinalDataY = originalResults["finalRawDataY"]

        optimizedResults = np.load(pathToOptimized)
        self.optFinalDataY = optimizedResults["finalRawDataY"]

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_HdataY(self):
        nptest.assert_array_equal(self.oriFinalDataY, self.optFinalDataY)


class TestFinalRawDataE(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriFinalDataE = originalResults["finalRawDataE"]

        optimizedResults = np.load(pathToOptimized)
        self.optFinalDataE = optimizedResults["finalRawDataE"]

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_HdataE(self):
        nptest.assert_array_equal(self.oriFinalDataE, self.optFinalDataE)


class Testpopt(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oripopt = originalResults["popt"]

        optimizedResults = np.load(pathToOptimized)
        self.optpopt = optimizedResults["popt"]
    
    def test_intensities(self):
        nptest.assert_array_equal(self.oripopt, self.optpopt)


class Testperr(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriperr = originalResults["perr"]

        optimizedResults = np.load(pathToOptimized)
        self.optperr = optimizedResults["perr"]
    
    def test_intensities(self):
        nptest.assert_array_equal( self.oriperr, self.optperr)



if __name__ == "__main__":
    unittest.main()