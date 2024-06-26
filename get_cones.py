import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy


def getCones():
    #data = pd.read_csv("world/data.csv")
    right = pd.read_csv("world/innerCones.csv")
    left = pd.read_csv("world/outerCones.csv")
    
    rightSide = right.to_numpy()[:,:3]
    leftSide = left.to_numpy()[:,:3]
    
    #print(rightSide.shape)
    #print(leftSide.shape)

    #xy = data.to_numpy()
    #leftSide = np.array(xy[:151,:])
    #rightSide = np.array(xy[151:,:])
    cLine = np.zeros(leftSide.shape, dtype=np.float32)

    for idx, lpt in enumerate(leftSide[:,:]):
        lpt=lpt.reshape((1,3))

        rdist = scipy.spatial.distance.cdist(lpt, rightSide)
        closerIdx = np.argmin(rdist)
        cLine[idx,:] = (lpt + rightSide[closerIdx,:]) /2
        
    return leftSide, rightSide, cLine
    

if __name__== "__main__":
    leftSide, rightSide, cLine = getCones() 
    print("l ",len(leftSide))
    print("r ",len(rightSide))
    print("c ",len(cLine))
    print(cLine[44,0], cLine[44,1])
    
    plt.scatter(leftSide[:,0],leftSide[:,1], 5, "blue")
    plt.scatter(rightSide[:,0],rightSide[:,1], 5, 'orange')
    plt.scatter(cLine[:,0], cLine[:,1], 5, 'green')
    plt.scatter(cLine[44,0], cLine[44,1], 5, 'black')
    plt.show()
