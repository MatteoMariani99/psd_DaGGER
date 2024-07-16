import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy


def getCones(track_number):
    """
        Funzione usata per leggere le posizioni dei coni per i diversi tracciati
        
        Parameters:
        - track_number: numero del track che si vuole visualizzare
        
        Returns:
        - leftSide: posizione coni di sinistra (colore blu)
        - rightSide: posizione coni di destra (colore giallo)
        - cLine: punti della linea intermedia tra i coni gialli e blu
    """
    
    # lettura delle posizioni dei coni
    right = pd.read_csv(f"world/cones_position/coneIN_track{track_number}.csv")
    left = pd.read_csv(f"world/cones_position/coneOUT_track{track_number}.csv")
    
    # aggiustamenti in base al tipo di tracciato 
    if track_number==1 or track_number==4:
        rightSide = right.to_numpy()[:,:2][::-1]
        leftSide = left.to_numpy()[:,:2][::-1]
    else:
        rightSide = right.to_numpy()[:,:2]
        leftSide = left.to_numpy()[:,:2]
   

    cLine = np.zeros(leftSide.shape, dtype=np.float32)

    # calcolo deli punti della linea intermedia
    for idx, lpt in enumerate(leftSide[:,:]):
        lpt=lpt.reshape((1,2))
        rdist = scipy.spatial.distance.cdist(lpt, rightSide)
        closerIdx = np.argmin(rdist)
        cLine[idx,:] = (lpt + rightSide[closerIdx,:]) /2
        
    return leftSide, rightSide, cLine
    

if __name__== "__main__":
    
    track_number = 0
    leftSide, rightSide, cLine = getCones(track_number) 
   
    # plot dei coni
    plt.scatter(leftSide[:,0],leftSide[:,1], 5, "blue")
    plt.scatter(rightSide[:,0],rightSide[:,1], 5, 'orange')
    plt.scatter(cLine[:,0], cLine[:,1], 5, 'green')
    plt.show()
