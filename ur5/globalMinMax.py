import os
import pickle
import numpy as np

min_x = float('inf')
min_y = float('inf')
min_z = float('inf')
max_x = float('-inf')
max_y = float('-inf')
max_z = float('-inf')

mins = [float('inf')]*78
maxes = [float('-inf')]*78

# print(mins)
# exit()

for filename in os.listdir("./trainingData"):
    # path = filename.split("_")[1].split(".")[0]
    
    print(filename)
    # print(path)
    
    sample = f"./trainingData/{filename}"
    # simFile = f"./error/finalEePos_{path}.pkl"
    with open(sample, 'rb') as f:
        data = pickle.load(f)
        # print(data)
        # exit()
        # print(len(data))
        # print(len(data[0]))
        # print("\n")
    

    for d in data:
        for i in range(78):
            mins[i] = min(d[i], mins[i])
            maxes[i] = max(d[i], maxes[i])

print("mins:\n", np.array(mins))
print("maxes:\n", np.array(maxes))

        # min_x = min(min(d[8], d[27]), min_x)
        # min_y = min(min(d[9], d[28]), min_y)
        # min_z = min(min(d[10], d[29]), min_z)
        
        # max_x = max(max(d[8], d[27]), max_x)
        # max_y = max(max(d[9], d[28]), max_y)
        # max_z = max(max(d[10], d[29]), max_z)


print(len(os.listdir("./trainingData")))

# print("global max x: ", max_x)
# print("global max y: ", max_y)
# print("global max z: ", max_z)

# print("global min x: ", min_x)
# print("global min y: ", min_y)
# print("global min z: ", min_z)
