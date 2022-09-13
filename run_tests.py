"""
The tests below only pass in Mantid version 6.2, 
other versions give fail.
"""

import unittest

# Initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# Import modules to be tested
import vesuvio_analysis.tests.test_analysis as analysis
# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(analysis))

import vesuvio_analysis.tests.test_yspace_fit as yspacefit
suite.addTests(loader.loadTestsFromModule(yspacefit))
import vesuvio_analysis.tests.test_yspace_fit_GC as yspacefit_GC
suite.addTests(loader.loadTestsFromModule(yspacefit_GC))

import vesuvio_analysis.tests.test_bootstrap as bootstrap
suite.addTests(loader.loadTestsFromModule(bootstrap))

import vesuvio_analysis.tests.test_jackknife as jackknife
suite.addTests(loader.loadTestsFromModule(jackknife))


# Initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)
