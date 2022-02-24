#%%
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from scipy import optimize


def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


np.random.seed(1)
x = np.linspace(-20, 20, 100)
yerr = np.random.rand(x.size) * 0.01
dataY = HermitePolynomial(x, 1, 0, 5, 1, 1) + yerr * np.random.randn(x.size)

p0 = [1, 0, 4, 0, 0]

costFun = cost.LeastSquares(x, dataY, yerr, HermitePolynomial)

m = Minuit(costFun, A=1, x0=0, sigma1=4, c4=0, c6=0)
m.limits["A"] = (0, None)

m.simplex()
constraints = optimize.NonlinearConstraint(lambda *pars: HermitePolynomial(x, *pars), 0, np.inf)
# constraints = ()
m.scipy(constraints=constraints)
# m.migrad()

m.hesse()
# yfit, ycov =  util.propagate(lambda pars: HermitePolynomial(x, *pars), m.values, m.covariance)
# ci = np.sqrt(np.diag(ycov)) * m.fval / (len(x)-m.nfit)


# m.minos()
m.draw_mnprofile("sigma1", bound=3)
# m.draw_contour("c4", "c6")
plt.show()
#%%
# cov = m.covariance.correlation()

#%%
# plt.errorbar(x, dataY, yerr, fmt="o")
# plt.plot(x, yfit, color="red", label="Minuit with Scipy")
# plt.fill_between(x, yfit-ci, yfit+ci, color="red", alpha=0.2, label="CI")

# # display legend with some fit info
# fit_info = [
#     f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {x.size - m.nfit}",
# ]
# for p, v, e in zip(m.parameters, m.values, m.errors):
#     fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

# plt.legend(title="\n".join(fit_info))

# plt.show()