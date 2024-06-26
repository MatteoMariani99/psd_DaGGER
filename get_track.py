import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import scipy.spatial

def loadVertex(circuit='world&car/meshes/barca_track.obj'):
    vertex = None
    mydir = os.path.split(__file__)[0]
    with open(f'{mydir}/{circuit}', 'r') as objf:
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

def computeTrack(debug=True):
    vn = loadVertex()
    lSide, rSide, cLine = splitTrack(vn,6)
    print(rSide[len(rSide)-75:len(rSide)].tolist())
    #getPointToStart(cLine,[0,0])
    if debug:
        plt.scatter(lSide[:,0], lSide[:,1], 1,'red')
        plt.scatter(rSide[:,0], rSide[:,1], 1, 'blue')
        plt.scatter(cLine[:,0], cLine[:,1], 1, 'green')
        plt.show()
    return lSide, rSide, cLine

def getPointToStart(cLine,target,threshold):

    # verifico che la posizone non sia vuota e aumento la threshold
    index = [index for index,pos in enumerate(cLine) if abs(pos[0]-(target[0]))<threshold and abs(pos[1]-(target[1]))<threshold]
    position = cLine[index]
    
    return index, position




if __name__== "__main__":
    computeTrack() 
           