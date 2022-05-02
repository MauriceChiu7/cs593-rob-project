# CS593 Robotics Spring 2022

## Learning Dynamical Systems for Model Predictive Control

### Team Members:
Purdue University   <br />
Gloria Ma – ma571@purdue.edu        <br />
Brandon Lee – lee3008@purdue.edu    <br />
Maurice Chiu – chiu93@purdue.edu    <br />
Kevin Chen – chen4066@purdue.edu    <br />

### Code Repositiories:
Main project repository: https://github.com/MauriceChiu7/cs593-rob-project

Libraries that we are using:    <br />
UR5: https://github.com/sholtodouglas/ur5pybullet       <br />
A1: https://github.com/unitreerobotics/unitree_pybullet <br />

### Implementation based on:
Sanchez-Gonzalez, A., Heess, N., Springenberg, J. T., Merel, J., Riedmiller, M., Hadsell, R., & Battaglia, P. (2018, July). Graph networks as learnable physics engines for inference and control. In International Conference on Machine Learning (pp. 4470-4479). PMLR.

Link to paper: https://arxiv.org/abs/1806.01242

### How to run codes for the UR5:
**First, be in this directory: `/cs593-rob-project/ur5`**

**Dependencies:**
Will need the files in `/cs593-rob-project/ur5pybullet` to run properly. These files were cloned from: https://github.com/sholtodouglas/ur5pybullet

---

**To generate a path using the PyBullet simulator, run: `python3 ur5.py`**

**Description:**
Uses the PyBullet simulator to load UR5 onto a plane and use MPC to plan for a path. 

**Produces:**
Upon completion the planning process, file `ur5sample.pkl` will be saved to `/cs593-rob-project/ur5/trainindDataWithEE` for playing back.

---

**To generate multiple paths, run: `python3 generatePathsUr5.py`**

**Description:**
Calls the same code that's in `ur5.py` in a loop. 
Takes in two commandline arguments: starting_index and ending_index. Including the starting_index and excluding the ending_index.
These indices are only for naming the generated paths so they don't overwrite each other.
For example: `python3 generatePathsUr5.py 120 123` will generate 3 paths, `ur5sample_120.pkl`, `ur5sample_121.pkl`, `ur5sample_122.pkl`.

**Produces:**
Upon completion, multiple files named in the format `ur5sample_{index}.pkl` will be saved to `/cs593-rob-project/ur5/trainindDataWithEE`.

---

**To train the neural net with UR5 paths, run: `python3 train.py`**

**Description:**
This program will take all of the paths generated from the `ur5.py` or `generatePathsUr5.py` from `/cs593-rob-project/ur5/trainindDataWithEE` and train the neural network.

**Produces:**
1. The model of the trained neural net: `UR5_V1_Model_2Layers_model1.pt` will be saved to `/cs593-rob-project/ur5/mult_models/UR5_V1_Model_2Layers`
2. Temporary saves of the model at each epoch: named in the format `UR5_V1_Model_2Layers_epoch{index}.pt` will be saved to `/cs593-rob-project/ur5/mult_models/UR5_V1_Model_2Layers`
3. A figure named `UR5_V1_Model_2Layers_MSE_Results.png` that shows the average MSE per epoch will be saved under `/cs593-rob-project/ur5/graphs`

---

**To do MPC with our trained neural net, run: `python3 nnur5mpc.py`**

**Description:**
This program is similar to that of `ur5.py`. The difference is that we've taken out the simulator PyBullet. And in place of PyBullet, we have our trained NN to predict the next states for our robot.

**Produces:**
Upon completion the planning process, file `test.pkl` will be saved to `/cs593-rob-project/ur5/testRunResults` for playing back.

---

**To play back the planned motion, run: `python3 playback_ur5.py`**

**Description:**
Please pass in the optional arguments to play back your desired simulations:
1. `'-m', '--mode', type=str, default='mpc', choices=['mpc', 'nnmpc']`
   - Use `-m mpc` to playback paths generated with the PyBullet MPC.
   - Use `-m nnmpc` to play back paths generated with the neural net MPC.
2. `'-pn', '--path-number', type=int, default=0`
   - You can specify which path to playback by giving it a path index. Use `-pn 33` to play back path 33 if using `--mode mpc`.

All of the paths files required are stored in `/cs593-rob-project/ur5/trainingDataWithEE`

If the file this program is trying to read doesn't exist, please be sure to run one of the following programs first:
1. `python3 ur5.py`
2. `python3 generatePathsUr5.py`

**Produces:**
N/A

---

#### Auxiliary Files for the UR5:
1. `plot.py` plots the Error vs Epoch graph using data from `/cs593-rob-project/ur5/graphs/error_epoch.pkl`.
2. `plotTrajComp.py` plots the trajectory and the end-effector positions at each environment step for a single path. To specify which path to plot, change the path number in line 14 of the code `path = 37` to the desired path number.
3. `globalMinMax.py` finds you the global minimum and global maximum values for the x, y, and z coordinates of the end-effector from all of the generated paths.
4. `findGoodPaths.py` compares and calculates the MSE between the trajectory and the end-effector positions at each environment step for every paths generated using data from `/cs593-rob-project/ur5/error`, prints out all the MSE values, all the good paths, and a percentage that indicates how many of them were good paths. Also produces figures and saves them as `traj_comp_{index}.png` at `/cs593-rob-project/ur5/figures`.

---

### How to run the codes for Unitree A1 Robot:

**Be in this directory: `/cs593-rob-project/a1`**

**Dependencies:**
Will need the files in `/cs593-rob-project/unitree_pybullet` to run properly. These files were cloned from: https://github.com/unitreerobotics/unitree_pybullet

---

**To generate a path using the PyBullet simulator, be in this directory: `/cs593-rob-project/a1/mpc` and run: `python3 run.py`**

**Description:**
Uses the PyBullet simulator to load A1 onto a plane and use MPC to plan for a path.

**Produces:**
Upon completion, file named in the format `run_I{iterations}_E{epochs}_Eps{episodes}_H{horizon}.pkl` is written to `/cs593-rob-project/a1/mpc/results` for playing back.

---

**To generate multiple paths, be in this directory: `/cs593-rob-project/a1/mpc` and run: `python3 generateData.py` or `python3 generateDataCuda.py`**

**Description:**
Calls the same code that's in `run.py` in a loop.
Takes in two commandline arguments: starting_index and ending_index. Including the starting_index and excluding the ending_index.
These indices are only for naming the generated paths so they don't overwrite each other.
For example: `python3 generateData.py 120 123` will generate 3 paths, `sample_120.pkl`, `sample_121.pkl`, `sample_122.pkl`.

**Produces:**
Upon completion, multiple files named in the format `sample_{index}.pkl` will be saved to `/cs593-rob-project/a1/mpc/trainingData/iter_{iterations}_epochs_{epochs}_episodes_{episodes}_horizon_{Horizon}`.

---

**To play back the planned motion generated by `/cs593-rob-project/a1/mpc/run.py`, be in this directory: `/cs593-rob-project/a1/mpc` and run: `python3 playback.py`**

**Description:**
Please pass in the optional arguments to playback your desired simulations:
1. `'--mode', type=str, default='mpc', choices=['mpc', 'gendata']`
   - Use `--mode mpc` to playback paths generated with `run.py`.
   - Use `--mode nnmpc` to playback paths generated with `generateData.py` or `generateData.py`.
2. `'--file', type=str, default='results/BESTRun.pkl`.
   - You can specify which path to playback by giving passing in the direct file path.

**Produces:**
N/A

---

**To generate data used in training the neural network, be in this directory: `/cs593-rob-project/a1/nnmpc` and run: `python3 generateData.py`**

**Description:**
This program runs the PyBullet MPC 6 times, each time, saving two top actions for environment steps.

**Produces:**
Upon completion, many files will be produced: 
1. multiple files each containing a list of actions for the best and second best paths of each environment step named in the format `MultRun_{multRun}_Iter_{iter}_Top{index}.pkl` will be saved to `/cs593-rob-project/a1/nnmpc/multActions_I{Iterations}_E{Epochs}_Eps{Episodes}/`. These lists of actions are only used for playing back. To playback these paths, see documentation on how to run `python3 playback_Train_Data.py`.
2. a file `MULT{multRun}_run_I{iterations}_E{epochs}_Eps{Episodes}.pkl`, containing all of the state-action-state pairs, will be saved to `/cs593-rob-project/a1/nnmpc/saData_Mult`. This file is used for training the neural net that will be replacing out simulator, PyBullet. 

---

**To train the neural network for A1, be in this directory: `/cs593-rob-project/a1/nnmpc` and run: `python3 train.py` or `python3 trainCuda.py`**

Note: `/cs593-rob-project/a1/nnmpc/NEWtrain.py` trains on updated state vector, but data needs to match (it does not match right now).

**Description:**
This program reads in the state-action-state pairs generated from `/cs593-rob-project/a1/nnmpc/generateData.py` stored in `/cs593-rob-project/a1/nnmpc/saData_Mult` and trains the neural network. It does so by first shuffling all the data, spliting them into a training data set and a testing data set, and then use the data from the training data set to train the neural network.

**Produces:**
1. The model of the trained neural net: `V2_Model_model1.pt` will be saved to `/cs593-rob-project/a1/nnmpc/mult_models/V2_Model`
2. Temporary saves of the model at each epoch: named in the format `V2_Model_epoch{index}.pt` will be saved to `/cs593-rob-project/a1/nnmpc/mult_models/V2_Model`
3. A figure named `V2_Model_MSE_Results.png` that shows the average MSE per epoch will be saved under `/cs593-rob-project/a1/mmnpc/graphs`

---

**To do MPC with our trained neural net, be in this directory: `/cs593-rob-project/a1/nnmpc` and run: `/cs593-rob-project/a1/nnmpc/run.py`**

Note: `/cs593-rob-project/a1/nnmpc/NEWrun.py` truns on model that was trained on updated state vector.

**Description:**
This program is similar to that of `/cs593-rob-project/a1/mpc/run.py`. The difference is that we've taken out the simulator PyBullet. And in place of PyBullet, we have our trained NN to predict the next states for our robot.

**Produces:**
Upon completion the planning process, two files are produced:
1. a file named in the format `V1_run_I{iterations}_E{epochs}_Eps{Episodes}.pkl` containing a list of best actions will be written to `NN_MPC_Action_Results`
2. a file named in the format `V1_run_I{iterations}_E{epochs}_Eps{episodes}_NNPREDICTED.pkl` will be saved at `/cs593-rob-project/a1/nnmpc/trajectories` contains a list of coordinates representing the predicted center of the A1 robot at each environment step. This is the trajectory of the A1 predicted by the neural net.

---

**To playback the paths generated from `/cs593-rob-project/a1/nnmpc/generateData.py` stored in `/cs593-rob-project/a1/nnmpc/multActions_I{Iterations}_E{Epochs}_Eps{Episodes}/`, be in this directory: `/cs593-rob-project/a1/nnmpc` and run: `python3 playback_Train_Data.py`**

**Description:**
This program lets you play back the list of actions generated by `/cs593-rob-project/a1/nnmpc/generateData.py`.

**Produces:**
N/A

---

**To playback the paths generated from `/cs593-rob-project/a1/nnmpc/run.py` stored in `/cs593-rob-project/a1/nnmpc/NN_MPC_Action_Results`, be in this directory: `/cs593-rob-project/a1/nnmpc` and run: `python3 playback_MPC_Results.py`**

**Description:**
This program lets you play back the paths generated by `/cs593-rob-project/a1/nnmpc/run.py`.
Takes in an optional argument `'--file', type=str, default='V1_run_I150_E3_Eps70.pkl`. You can also specify which file stored in the `NN_MPC_Action_Results` directory to run.

**Produces:**
This program produces a file named in the format `V1_run_I{iterations}_E{epochs}_Eps{episodes}.pkl_ACTUAL` (yes, we messed up the naming but it reads fine ;)) that will be saved at `/cs593-rob-project/a1/nnmpc/trajectories` contains a list of coordinates that is the actual center of A1 robot at each environment step, recorded during playback. This actual trajectory is to be used for comparing to the trajectory predicted by our neural net from `/cs593-rob-project/a1/nnmpc/run.py`

---

#### Auxiliary files for the A1:
1. `/cs593-rob-project/a1/mpc/epochErrorGrapher.py` plots the error data generated from `/cs593-rob-project/a1/mpc/run.py` and stores it in `/cs593-rob-project/a1/mpc/graphs`
2. `/cs593-rob-project/a1/nnmpc/graph3d.py` plots the trajectory predicted by our neural net recorded from `/cs593-rob-project/a1/nnmpc/run.py` and the actual trejectory produced through simulation recorded from `/cs593-rob-project/a1/nnmpc/playback_MPC_Results.py` and shows it
3. `/cs593-rob-project/a1/nnmpc/maxStateAction.py` is the program we used to find out the global minimum and global maximum values from all of our sampled data. We gather this data so that we can normalize our data for training and unnormalize them to be used in a simulation.
