import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy

#track_number = 4



def getCones(track_number):
    
    # new track
    right = pd.read_csv(f"world/cones_position/coneIN_track{track_number}.csv")
    left = pd.read_csv(f"world/cones_position/coneOUT_track{track_number}.csv")
    
    
    if track_number==1 or track_number==4:
        rightSide = right.to_numpy()[:,:2][::-1]
        leftSide = left.to_numpy()[:,:2][::-1]
    else:
        rightSide = right.to_numpy()[:,:2]
        leftSide = left.to_numpy()[:,:2]
   

    cLine = np.zeros(leftSide.shape, dtype=np.float32)

    for idx, lpt in enumerate(leftSide[:,:]):
        lpt=lpt.reshape((1,2))

        rdist = scipy.spatial.distance.cdist(lpt, rightSide)
        closerIdx = np.argmin(rdist)
        cLine[idx,:] = (lpt + rightSide[closerIdx,:]) /2
        
    return leftSide, rightSide, cLine
    

if __name__== "__main__":
    leftSide, rightSide, cLine = getCones(1) 
    print("l ",len(leftSide))
    print("r ",len(rightSide))
    print("c ",len(cLine))
   
    
    plt.scatter(leftSide[:,0],leftSide[:,1], 5, "blue")
    plt.scatter(rightSide[:,0],rightSide[:,1], 5, 'orange')
    plt.scatter(cLine[:,0], cLine[:,1], 5, 'green')
    plt.show()
