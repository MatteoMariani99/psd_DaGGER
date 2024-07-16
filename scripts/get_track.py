import numpy as np
import scipy
from matplotlib import pyplot as plt
import scipy.spatial


def loadVertex(circuit='world/models/road/meshes/barca_track.obj'):
    """
    Funzione usata caricare i vertici della strada dal file .obj
    
    Parameters:
    - circuit: path della strada
    
    Returns:
    - vertex: rappresentano i punti della strada
    """
    vertex = None
    with open(f'{circuit}', 'r') as objf:
        for l in objf.readlines():
            if vertex ==None and not ('Line001Mesh' in l):
                continue
            if vertex == None:
                vertex = []
                continue
            el = l.split(' ')
            if el[0] != 'v':
                break
            vertex.append([float(x) for x in el[1:]])
    return  np.array(vertex)


def splitTrack(vn, skew=6):
    """
        Funzione usata per dividere i vertici tra sinistra e destra
        
        Parameters:
        - track_number: numero del track che si vuole visualizzare
        
        Returns:
        - leftSide: posizione punti di sinistra 
        - rightSide: posizione punti di destra 
        - cLine: punti della linea intermedia 
    """
    nVertex = int((vn.shape[0])/2)+skew
    leftSide=vn[:nVertex]
    rightSide=vn[nVertex:]
    cline = np.zeros(leftSide.shape, dtype=np.float32)

    for idx, lpt in enumerate(leftSide[:,:]):
        lpt=lpt.reshape((1,3))

        rdist = scipy.spatial.distance.cdist(lpt, rightSide)
        closerIdx = np.argmin(rdist)
        cline[idx,:] = (lpt + rightSide[closerIdx,:]) /2
    
    return leftSide, rightSide, cline


def getPointToStart(cLine,target,threshold):
    """
        Funzione usata per capire quale è il punto della linea intermedia (più vicino
        alla posizione della macchina) da cui partire con il controllore 
        
        Parameters:
        - cline: linea intermedia
        - target: posizione della macchina
        - threshold: soglia per allargare il raggio di ricerca del punto di partenza
        
        Returns:
        - index: indice del punto selezionato
        - position: punto della linea intermedia più vicino alla posizione della macchina
    """
    # verifico che la posizone non sia vuota e aumento la threshold
    index = [index for index,pos in enumerate(cLine) if abs(pos[0]-(target[0]))<threshold and abs(pos[1]-(target[1]))<threshold]
    position = cLine[index]
    
    return index, position

def computeTrack(debug=True):
    vn = loadVertex()
    lSide, rSide, cLine = splitTrack(vn,6)
    
    if debug:
        plt.scatter(lSide[:,0], lSide[:,1], 1,'red')
        plt.scatter(rSide[:,0], rSide[:,1], 1, 'blue')
        plt.scatter(cLine[:,0], cLine[:,1], 1, 'green')
        plt.show()
    return lSide, rSide, cLine


if __name__== "__main__":
    computeTrack(debug=True)
           