import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy


def getCones():
    data = pd.read_csv("world/data.csv")

    xy = data.to_numpy()
    leftSide = np.array(xy[:151,:])
    rightSide = np.array(xy[151:,:])
    cLine = np.zeros(leftSide.shape, dtype=np.float32)

    for idx, lpt in enumerate(leftSide[:,:]):
        lpt=lpt.reshape((1,3))

        rdist = scipy.spatial.distance.cdist(lpt, rightSide)
        closerIdx = np.argmin(rdist)
        cLine[idx,:] = (lpt + rightSide[closerIdx,:]) /2
        
    return leftSide, rightSide, cLine
    

if __name__== "__main__":
    leftSide, rightSide, cLine = getCones() 
    # print("l ",leftSide)
    # print("r ",rightSide)
    # print("c ",cLine)
    
    plt.scatter(leftSide[:,0],leftSide[:,1], 5, "orange")
    plt.scatter(rightSide[:,0],rightSide[:,1], 5, 'blue')
    plt.scatter(cLine[:,0], cLine[:,1], 5, 'green')
    plt.show()
