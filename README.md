# scatt_scripts

## Repository for the optimized NCP analysis procedures on VESUVIO

Three example scripts are provided, BaH2_500C, D_HMT and starch_80_RD, each detailing the initial conditions for each sample.

Start with script starch_80_RD, includes most complete comments.

How to use for a new sample:

    1. Copy one of the main scripts (for example starch_80_RD.py) and create a new .py file with the name of the desired sample and in the sample directory as D_HMT.py or starch_80_RD.py. 

    2. Fill in the new script with the desired initial conditions.

    3. Run the script. The script will create a new directory for the sample under experiments/, and will try to use LoadVesuvio to store the workspaces locally for future runs. If LoadVesuvio fails, the user needs to copy the worksapces as .nxs files onto experiments/sample/input_ws/, using the same format as the example samples provided.

    4. After the workspaces are stored locally, any further data reduction will run using the local workspaces. 

    5. Bootstrap option is still under development, but any data from running bootstrap is stored under experiments/sample/bootstrap_data/ or experiments/sample/jackknife_data

    6. Analysis of bootstrap data works only with stored data in directories mentioned in point 5, it does not run any bootstrap
