How to run:

# This runs regular MPC algorithm
# 1. Change parameters (if needed) on lines 144-147
# 2. If want to graph error after each iteration, uncomment lines 201,216-217
# 3. Saved in results/ folder
run.py:
python3 run.py

playback.py:
python3 playback --file <path>

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
