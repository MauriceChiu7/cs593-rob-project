import os
import pickle

min_x = float('inf')
min_y = float('inf')
min_z = float('inf')
max_x = float('-inf')
max_y = float('-inf')
max_z = float('-inf')

for filename in os.listdir("./trainingDataWithEE"):
    # path = filename.split("_")[1].split(".")[0]
    
    print(filename)
    # print(path)
    
    sample = f"./trainingDataWithEE/{filename}"
    # simFile = f"./error/finalEePos_{path}.pkl"
    with open(sample, 'rb') as f:
        data = pickle.load(f)
    

    for d in data:
        min_x = min(min(d[8], d[27]), min_x)
        min_y = min(min(d[9], d[28]), min_y)
        min_z = min(min(d[10], d[29]), min_z)
        
        max_x = max(max(d[8], d[27]), max_x)
        max_y = max(max(d[9], d[28]), max_y)
        max_z = max(max(d[10], d[29]), max_z)

print("global max x: ", max_x)
print("global max y: ", max_y)
print("global max z: ", max_z)

print("global min x: ", min_x)
print("global min y: ", min_y)
print("global min z: ", min_z)

# exit()
    # print(traj)
    # print(sim)
    
#     error = mse(traj, sim)
#     # print(error)

#     print(f"{filename} has MSE: {error}")

#     if not os.path.exists(f"./figures/traj_comp_{path}.png"):
#         print(f"creating figure for path {path}")
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.set_aspect('auto')
#         ax.set_xlim([-0.8,0.8])
#         ax.set_ylim([-0.8,0.8])
#         ax.set_zlim([-0,0.8])
#         ax.scatter3D(traj[:,0], traj[:,1], traj[:,2], c='green')
#         ax.scatter3D(sim[:,0], sim[:,1], sim[:,2], c='red')
#         plt.figtext(0.1, 0.95, f"path {path}")
#         plt.savefig(f"./figures/traj_comp_{path}.png")
#         plt.close()

#     if error <= 0.02:
#         goodPaths.append(int(path))

# goodPaths.sort()
# print(f"Good Paths: {goodPaths}")

# percentage = len(goodPaths)/len(os.listdir("./trainingData"))
# print(f"{len(goodPaths)} out of {len(os.listdir('./trainingData'))} are good paths. {percentage*100}%")

