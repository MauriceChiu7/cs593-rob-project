# mse_x = np.square(np.subtract(x, actual_x)).mean()
# mse_y = np.square(np.subtract(y, actual_y)).mean()
# plt.plot(np.arange(0, len(errors_x)), errors_x, 'b', np.arange(0, len(errors_y)), errors_y, 'r')
# plt.savefig('err_v_iter_x-y.png')
# plt.show()
# plt.clf()

# import argparse
# import csv
import matplotlib.pyplot as plt
import pickle
# import re

colors = ["Orange", "Blue", "Green", "Red", "Black"]

def readPickle(file):
    '''
    Description: Reads a pickle file.

    Input:
    :file - {str} The path to the pickle file.

    Returns:
    :content - {list} The content of the pickle file.
    '''
    print("\nreading data as pickle...")
    with open(file, 'rb') as f:
        content = pickle.load(f)
    for c in content:
        print(c)

    return content

def main():
    '''
    Description: plots and shows the data from the pickle file.

    Input: None

    Returns: None
    '''
    legends = []

    content = readPickle("./graphs/error_epoch.pkl")
    x_axis = []
    y_axis = []
    for (x, y) in content:
        x_axis.append(x)
        y_axis.append(y)
    
    print(len(x_axis))
    print(len(y_axis))
    print(x_axis)
    print(y_axis)
        
        
    subtitle = f"Error v Epochs for the UR5"
        
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    
    plt.plot(x_axis, y_axis, color=colors[1], marker='o')
    legends.append(subtitle)

    for l in legends:
        plt.figtext(0.1, 0.95-(0.05*legends.index(l)), l)

    plt.show()



    # filename = f"./{args.robot}_final_actions.csv"
    # file = open(filename)
    # csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    # finalActions = []
    # for row in csvreader:
    #     finalActions.append(row)
    # file.close()
    # if args.verbose: print(f"\n...final actions read\n")


if __name__ == "__main__":
    
    main()