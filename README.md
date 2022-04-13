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

**To plan for a motion using the PyBullet simulator, run: `python3 ur5.py`**
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
   - You can specify which path to playback by giving it a path index. Use `-pn 33` to play back path 33.

All of the paths files required are stored in `/cs593-rob-project/ur5/trainingDataWithEE`

If the file this program is trying to read doesn't exist, please be sure to run one of the following programs first:
1. `python3 ur5.py`
2. `python3 generatePathsUr5.py`

**Produces:**
N/A

#### Auxiliary Files for the UR5:
1. `plot.py` plots the Error vs Epoch graph using data from `/cs593-rob-project/ur5/graphs/error_epoch.pkl`.
2. `plotTrajComp.py` plots the trajectory and the end-effector positions at each environment step for a single path. To specify which path to plot, change the path number in line 14 of the code `path = 37` to the desired path number.
3. `globalMinMax.py` finds you the global minimum and global maximum values for the x, y, and z coordinates of the end-effector from all of the generated paths.
4. `findGoodPaths.py` compares and calculates the MSE between the trajectory and the end-effector positions at each environment step for every paths generated using data from `/cs593-rob-project/ur5/error`, prints out all the MSE values, all the good paths, and a percentage that indicates how many of them were good paths. Also produces figures and saves them as `traj_comp_{index}.png` at `/cs593-rob-project/ur5/figures`.

---

For the Unitree A1 Robot:   <br />
Be in this directory: `/cs593-rob-project/unitree_pybullet`

To plan for a motion, run: `python3 a1.py`  <br />
Upon completion the planning process, file `final_actions.csv` will be saved to your current directory for playing back.

To play back the planned motion, run: `python3 a1.py --play`.   <br />
If the `final_actions.csv` doesn't exist, please be sure to run the plan command.