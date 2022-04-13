How to run:

# This runs regular MPC algorithm
# 1. Parameters can be changed, see TODO in file
# 2. If want to graph error after each iteration, need to uncomment some lines, see TODO in file
# 3. Saved in results/ folder
run.py:
python3 run.py

# Playback either mpc made run, or generateData.py runs that store s0,a0,s1
playback.py:
python3 playback --file <path> --mode mpc
python3 playback --file <path> --mode gendata   # this is to playback stuff from generateData.py

# This runs MPC algorithm multiple times to generate data
# 1. For file naming purposes:
# start: index to start
# end: index to end
# 2. Saved in trainingData folder
generateData.py:
python3 generateData.py <start> <end>

# To generate faster on GPUs
generateDataCuda.py:
python3 generateDataCuda.py <start> <end>

# If you graphed error from run.py: can display errors here:
grapher.py
python3 grapher.py <path> 
