# CS593 Robotics Spring 2022

## Dynamical systems for Model Predictive Control

### Team members:
Purdue University
Gloria Ma – ma571@purdue.edu
Brandon Lee – lee3008@purdue.edu
Maurice Chiu – chiu93@purdue.edu
Kevin Chen – chen4066@purdue.edu

### Code repositiories:
Main project repository: https://github.com/MauriceChiu7/cs593-rob-project

Libraries that we are using:
UR5: https://github.com/sholtodouglas/ur5pybullet
A1: https://github.com/unitreerobotics/unitree_pybullet

### The paper we are trying to implement:
Sanchez-Gonzalez, A., Heess, N., Springenberg, J. T., Merel, J., Riedmiller, M., Hadsell, R., & Battaglia, P. (2018, July). Graph networks as learnable physics engines for inference and control. In International Conference on Machine Learning (pp. 4470-4479). PMLR.

Link to paper: https://arxiv.org/abs/1806.01242

### Links to DEMO videos:
A1 Robot Planning Phase:  https://youtu.be/sz_-2mRpAds
A1 Robot Plan Execution:  https://youtu.be/USulme13TeM
UR5 Robot Planning Phase: https://youtu.be/yiAxxk-3bPQ
UR5 Robot Plan Execution: https://youtu.be/xhQ95UoJWm8

### How to run:
For the UR5 Manipulator:
Be in this directory: `/cs593-rob-project`

To plan for a motion, run: `python3 ur5mpc.py`
Upon completion the planning process, file `ur5_final_actions.csv` will be saved to your current directory for playing back.

To play back the planned motion, run: `python3 ur5mpc.py --play`.
If the `ur5_final_actions.csv` doesn't exist, please be sure to run the plan command.

For the Unitree A1 Robot:
Be in this directory: `/cs593-rob-project/unitree_pybullet`

To plan for a motion, run: `python3 a1.py`
Upon completion the planning process, file `final_actions.csv` will be saved to your current directory for playing back.

To play back the planned motion, run: `python3 a1.py --play`.
If the `final_actions.csv` doesn't exist, please be sure to run the plan command.