
#testing a jupyter notebook

#%%
import numpy as np

arrayA = np.arange(12)
shapeOfA = (3,4)
arrayA = arrayA.reshape(shapeOfA)
print(arrayA)

selected = arrayA.flatten() * np.ones((3,1)) [False, False, True]
print(selected)

#%%
import sys
for p in sys.path:
    print(p)

#%%
%load_ext snakeviz
%snakeviz import Final_optimized_bckwd_script

# %%
import numpy as np
newResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild_cleanest.npz")
oldResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild.npz")

#np.testing.assert_allclose(newResults["all_mean_intensities"], oldResults["all_mean_intensities"])

for key in oldResults:
    try:
        print("\nevaluating: ",key)
        np.testing.assert_allclose(newResults[key][0], oldResults[key][0], rtol=1e-4)            
        print("shape: ", newResults[key].shape)
    except KeyError:
        print("KeyError: one of the results doesnt have this key")
    except AssertionError:
        print("Assertion Error")

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

syn = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild_synthetic.npz")

ws = syn["all_fit_workspaces"][0, :, :-1]
ncp = syn["all_tot_ncp"][0]

x = np.linspace(0, 1, len(ncp[0]))
plt.figure(3)
spec_idx = 0
plt.plot(x, ws[spec_idx], label="synthetic ncp", linewidth = 2)
plt.plot(x, ncp[spec_idx], "--", label="fitted ncp", linewidth = 2)
plt.legend()
plt.show()

ncp_mask = np.isclose(ws, ncp, rtol=0.01, equal_nan = True)
plt.figure(0)
plt.imshow(ncp_mask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
plt.title("Comparison between ws and ncp")
plt.xlabel("TOF")
plt.ylabel("spectrums")
plt.show()

# %%
