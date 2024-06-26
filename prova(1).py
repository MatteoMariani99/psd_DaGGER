
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import random
from get_track import *
from ultralytics import YOLO
import time
from get_cones import *

from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from collections import Counter


turtle= None


wrap2pi = lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi

class cameraMgr:
    #                                                     UL        UR          LR        LL
    #roadPts4Homography = np.array([[214,355], [487,355], [632,472], [89,472] ]).astype(np.float32)
    conePts4Homography = np.array([[241,327], [479,327], [602,476], [23,476] ]).astype(np.float32)
    roadDesiredPts     = np.array([[ 150,355], [500,355], [500,472], [150,472]  ]).astype(np.float32)
    #HomoMat =cv2.getPerspectiveTransform(roadPts4Homography[:,::1], roadDesiredPts[:,::1])
    HomoMat = None

    @staticmethod
    def getCamera_image():
        global turtle
        camInfo = p.getDebugVisualizerCamera()
        ls = p.getLinkState(turtle,7, computeForwardKinematics=True)
        camPos = ls[0]
        camOrn = ls[1]
        camMat = p.getMatrixFromQuaternion(camOrn)

        #upVector = [0,0,1]
        forwardVec = [camMat[0],camMat[3],camMat[6]]
        #sideVec =  [camMat[1],camMat[4],camMat[7]]
        camUpVec =  [camMat[2],camMat[5],camMat[8]]
        camTarget = [camPos[0]+forwardVec[0]*10,camPos[1]+forwardVec[1]*10,camPos[2]+forwardVec[2]*10]
        #camUpTarget = [camPos[0]+camUpVec[0],camPos[1]+camUpVec[1],camPos[2]+camUpVec[2]]
        viewMat = p.computeViewMatrix(camPos, camTarget, camUpVec)
        projMat = camInfo[3]
       
        # ottengo le 3 immagini: rgb, depth, segmentation
        width, height, rgbImg, depthImg, segImg= p.getCameraImage(640,480,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # faccio un reshape in quanto da sopra ottengo un array di elementi
       

        # Tolgo il canale alpha e converto da BGRA a RGB per la rete
        imgHSV = cv2.cvtColor(rgbImg[:,:,:3], cv2.COLOR_RGB2HSV)
        skyMsk = cv2.inRange(imgHSV[:,:,0], 115,125)
        yellowMsk = cv2.inRange(imgHSV[:,:,0], 28,32)
        yM = cv2.inRange(rgbImg[:,:,2], 0,1)
        bM = cv2.inRange(rgbImg[:,:,2], 140,215)
        blueMsk = cv2.inRange(imgHSV[:,:,0], 118,122)
        #greenMsk = cv2.inRange(imgHSV[:,:,0], 55,65)
 

 
        #greenMsk = cv2.morphologyEx(greenMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=2)
        skyMsk = cv2.morphologyEx(skyMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
        yellowMsk = cv2.morphologyEx(yellowMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
        blueMsk = cv2.morphologyEx(blueMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
       
       
        # findEdge = lambda x, line: np.argwhere(np.abs(np.diff(x[line,:].astype(np.int16)))==255)
        # topEdge = findEdge(greenMsk,320).flatten()
        # botEdge = findEdge(greenMsk, 460).flatten()
   
        #       print(f'BOT{botEdge}')
        #       print(f'TOP{topEdge}')
        # global HomoMat
        # if HomoMat is None and len(topEdge)==2 and len(botEdge)==2:
        #        origPts = np.array([[topEdge[0], 320],[topEdge[1], 320],[botEdge[1], 460],[botEdge[0], 460]]).astype(np.float32)
        #        destPts = np.array([[250,360],[390,360],[390,460],[250,460]]).astype(np.float32)
        #HomoMat =cv2.getPerspectiveTransform(conePts4Homography, roadDesiredPts)
        #print(HomoMat)
        # HomoMat = np.array([[-2.98962782e-01, -1.24344112e+00,  3.93075046e+02],
        # [-3.38241831e-15, -2.16351434e+00,  5.70860281e+02],
        # [-8.01126080e-18, -4.17937767e-03,  1.00000000e+00]])
       
        #print(colorOnly.shape)
        #cv2.imshow('testIMAGE', bM)

        #cv2.imshow('rectified', cv2.warpPerspective(255-(greenMsk+skyMsk), HomoMat, (640,480),flags=cv2.INTER_LINEAR)[::3,::3])
        #cv2.imshow('rectified', cv2.warpPerspective(255-(skyMsk), HomoMat, (640,480),flags=cv2.INTER_LINEAR)[::3,::3])
        #cv2.waitKey(1)
        #print("Reshaped RGB image size:", rgb_image.shape)

        return rgbImg[:,:,:3]


    @staticmethod
    def getHomographyFromPts():
        if cameraMgr.HomoMat is not None:
            return cameraMgr.HomoMat
        # scelta dei 4 punti nell'immagine di partenza da mappare poi nei punti di destinazione dst dell'immagine finale
        # questi 4 punti sono la regione d'interesse
        # basso sinistra, alto sinistra, alto destra, basso destra
        #src = np.float32([[img.shape[1]*(0.5-self.mid_width/2), img.shape[0]*self.height_pct],[img.shape[1]*(0.5+self.mid_width/2),img.shape[0]*self.height_pct],[img.shape[1]*(0.5+self.bot_width/2), img.shape[0]*self.bottom_trim],[img.shape[1]*(0.5-self.bot_width/2), img.shape[0]*self.bottom_trim]])
        #print(img_size)
        #src = np.float32([[25,44],[188,44],[188,65],[25,65]])
        #conePts4Homography = np.array([[196,307], [444,307], [580,385], [61,385]]).astype(np.float32)
        cameraMgr.conePts4Homography = np.array([[186,324], [458,324], [601,412], [43,412]]).astype(np.float32)
        #                                                         UL        UR          LR        LL
        #conePts4Homography = np.array([[171,334], [449,334], [586,479], [12,479] ]).astype(np.float32)
        cameraMgr.roadDesiredPts     = np.array([[ 140,307], [500,307], [500,400], [140,400]  ]).astype(np.float32)
       
        # for i in src:
        #         print(i)
        #         cv2.circle(image,[int(i[0]),int(i[1])],2,(0,0,255),-1)
               
       
        #dst = np.float32([[0+150,0],[img_size[0]-150,0],[img_size[0]-150,img_size[1]],[0+150,img_size[1]]])


        #print(src)
        # con la funzione cv2.getPerspectiveTransform otteniamo la matrice di rotazione che ci porta dai punti
        # sorgenti a quelli di destinazione
        #HomoMat = cv2.getPerspectiveTransform(conePts4Homography,roadDesiredPts)
        #print(np.array(HomoMat))
        # HomoMat = np.array([[-0.43976, -1.3553,  459.86],
        #         [-3.9202e-16, -2.5465,  688.78],
        #         [-4.7317e-19, -0.0042441 ,  1.00000000e+00]])
        # HomoMat = np.array([[   -0.43976  ,   -1.3553 ,     459.86],
        #             [          0   ,  -2.0589,      539.06],
        #             [ 4.7317e-19,  -0.0042441  ,         1]])
        cameraMgr.HomoMat = np.array([[   -0.46095,     -1.3316,      468.43],
                            [ 9.4581e-16,     -2.0326,      551.64],
                            [ 2.3918e-18,      -0.0041613,       1]])
        return cameraMgr.HomoMat

    @staticmethod    
    def birdEyeView(image):

        # Setting parameter values
        #t_lower = 120  # Lower Threshold
        #t_upper = 210  # Upper threshold
       
        # immagine canny
        #edge = cv2.Canny(image, t_lower, t_upper)
        #y,x = np.where(edge>0)

        #image_edges = image.copy()
       
        #imgHSV = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2HSV)
        #skyMsk = cv2.inRange(imgHSV[:,:,0], 115,125)
        #yellowMsk = cv2.inRange(imgHSV[:,:,0], 28,32)
        #yM = cv2.inRange(image[:,:,2], 0,1)
        rM = cv2.inRange(image[:,:,2], 254,255)
        #blueMsk = cv2.inRange(imgHSV[:,:,0], 118,122)
        #greenMsk = cv2.inRange(imgHSV[:,:,0], 55,65)
   

   
        #greenMsk = cv2.morphologyEx(greenMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=2)
        # skyMsk = cv2.morphologyEx(skyMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
        # yellowMsk = cv2.morphologyEx(yellowMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
        # blueMsk = cv2.morphologyEx(blueMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
       
       
       
        #edges = cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB)
        #edges[y,x] = [255,255,255]

        #cv2.imshow("Camera", edges)
        #cv2.waitKey(0)

        #abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
        #abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
        #scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        #print(scaled_sobel)
        #edge = cv2.Canny(img, 300, 355,L2gradient=True)
       
        #mask = cv2.inRange(image, t_lower, t_upper)
        #lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Usate per rimuovere il rumore
        # lower = np.array([0, 140, 140])
        # upper = np.array([255, 190, 190])
        #mask = cv2.inRange(lab_img, lower, upper)
        #filter1 = cv2.morphologyEx(abs_sobel, cv2.MORPH_OPEN, np.ones((1, 5), np.uint8), iterations=1)
        #filter = cv2.morphologyEx(filter1, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8), iterations=1)
       
        #ret, filtered = cv2.threshold(edges, 5, 255, cv2.THRESH_BINARY)
        # blur = cv2.GaussianBlur(edge, (0,0), sigmaX=33, sigmaY=33)

        # # # divide
        # divide = cv2.divide(edge, blur, scale=255)

        # # # otsu threshold
        # thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # # # apply morphology
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        #img = cv2.undistort(image,mtx,dist,None,mtx)
        #img_size = (image.shape[1],image.shape[0])  
       

        # # Combina i gradienti in uscita dalla funzione di Sobel con la binary image
        #preprocessImage = np.zeros_like(img[:,:,0]) # canale blue
        # gradx = self.abs_sobel_thresh(img, orient='x', thresh_min=12, thresh_max=255)
        # grady = self.abs_sobel_thresh(img, orient='y', thresh_min=25, thresh_max=255)
        # c_binary = self.color_threshold(img, sthresh=(100,255), vthresh=(50,255))
        # Questa parte seleziona i pixel in preprocessImage dove sia gradx che grady sono uguali a 1 oppure c_binary
        # è uguale a 1.
        # = 255: infine imposta il valore di intensità di questi pixel selezionati su 255 ovvero rendendoli bianchi.
        #preprocessImage[((gradx == 1) & (grady ==1) | (c_binary == 1))] = 255

        HomoMat = cameraMgr.getHomographyFromPts()
       
        # la funzione cv2.warpPerspective viene utilizzata per applicare la Perspective Transform ad un'immagine,
        # data la matrice di trasformazione
        # INPUT:
        # input image
        # transformation matrix
        # dimensioni immagine finale
        # metodo di interpolazione
       
        #print("r: ",rM.shape)
        bird_eye = cv2.warpPerspective(image, HomoMat, (640,480),flags=cv2.INTER_LINEAR)
       
        #print(bird_eye.shape)

        # viene ritornata anche questa in quanto permette di avere la birdEye view sull'immagine originale
        #warped_original = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

        return bird_eye

    @staticmethod
    def birdViewImg(blue_center, yellow_center, img):
       
        center_of_camera = [306,477] # in coordinate camera
       
        distance_to_cones_b, distance_to_cones_y = [],[]
       
        if blue_center.shape[0] == 0:
            blue_center = np.array([[479,0]], dtype=int)

        if yellow_center.shape[0] == 0:
            yellow_center = np.array([[479,639]], dtype=int)
       
        # riordino l'array
        pts_blue, pts_yellow = sortedPoints(blue_center, yellow_center)
       
        if len(pts_blue)>10:
            pts_blue = pts_blue[len(pts_blue)-10:len(pts_blue)]
       
        if len(pts_yellow)>10:
            pts_yellow = pts_yellow[len(pts_yellow)-10:len(pts_yellow)]
       
        #HomoMat = getHomographyFromPts()
        # distanza
        # applyHomo = lambda pts: cv2.convertPointsFromHomogeneous((HomoMat@cv2.convertPointsToHomogeneous(pts)[:,0,:].T).T).astype(int)
       
        # pts_bird_b = applyHomo(pts_blue)
        # pts_bird_y = applyHomo(pts_yellow)
       
        for pts in pts_blue:
           
            distance = center_of_camera-pts
            distance_to_cones_b.append(distance)
       
        #print(np.array(distance_to_cones_b))
       
        for pts in pts_yellow:
           
            distance = pts-center_of_camera
            distance_to_cones_y.append(distance)

        distance_to_cones_y = np.array(distance_to_cones_y)
        distance_to_cones_b = np.array(distance_to_cones_b)

        print("---------------------------")
        print(pts_blue)
        print(distance_to_cones_b)
        print("---------------------------")
        print(pts_yellow)
        print(distance_to_cones_y)
        print("---------------------------")


        #idx_b = np.where((distance_to_cones_b[:,0] > 294) | (abs(distance_to_cones_y[:,1])>230))
        idx_y = np.where((distance_to_cones_y[:,0] > 294) | (abs(distance_to_cones_y[:,1])>220))


        pts_yellow = np.delete(pts_yellow, idx_y, axis=0)
        #pts_blue = np.delete(pts_blue, idx_b, axis=0)
       
       
        print("After Bird B: ",pts_blue)
        print("Aftr Bird Y: ",pts_yellow)
       
        #pts_blue = np.append(pts_blue,[np.array([0,479])],axis=0)
        #pts_yellow = np.append(pts_yellow,[np.array([639,479])],axis=0)
        #pts_yellow = pts_yellow[::-1]
        #print(pts_blue)
        #print(pts_yellow)
       
        z_b = np.polyfit(pts_blue[:,0], pts_blue[:,1], 2)
        z_y = np.polyfit(pts_yellow[:,0], pts_yellow[:,1], 2)

        p_b = np.poly1d(z_b)
        p_y = np.poly1d(z_y)


        #print(pts_bird_b[:,0,0].min())
        #print(pts_bird_b[:,0,0].max())
       
        # Generate a range of x values for a smooth curve
        #x_smooth_b = np.linspace(pts_bird_b[:,0,0].min(), pts_bird_b[:,0,0].max(), 10)
        #x_smooth_y = np.linspace(pts_bird_y[:,0,0].min(), pts_bird_y[:,0,0].max(), 10)
        #print(x_smooth_b)
        #print(x_smooth_y)
       
        y_fit_b = p_b(pts_blue[:,0])
        y_fit_y = p_y(pts_yellow[:,0])
        #y_fit = np.polyval(z, lspace)   # evaluate the polynomial
        #y_fit = p(lspace)

        #draw_points = (np.asarray([pts_bird_b[:,0,0], y_fit]).T).astype(np.int32)   # needs to be int32 and transposed

        verts_b = np.array(list(zip(pts_blue[:,0].astype(int),y_fit_b.astype(int))))
        verts_y = np.array(list(zip(pts_yellow[:,0].astype(int),y_fit_y.astype(int))))
       
        # impilo i punti
        #points = np.vstack((pts_blue, pts_yellow))

        # reshape utile a polylines
        #pts_b = pts_blue.reshape((-1, 1, 2))
        #pts_y = pts_yellow.reshape((-1, 1, 2))

        imgB = np.zeros((480,640), dtype=np.uint8)
        cv2.polylines(img, [verts_b], isClosed=False,color=(255, 255, 255), thickness=3)
        cv2.polylines(img, [verts_y], isClosed=False,color=(255, 255, 255), thickness=3)
        # cv2.polylines(img, [pts_b], isClosed=False,color=(255, 0, 0), thickness=2)
        # cv2.polylines(img, [pts_y], isClosed=False,color=(51, 255, 255), thickness=2)

        # cv2.fillPoly(img, pts=[points], color=(0, 0, 255))
        # cv2.circle(img, (center_of_camera[0],center_of_camera[1]), 2, (30,255,255), 2)
   
        bird = cameraMgr.birdEyeView(img)
        #bird = np.zeros((480,640), dtype=np.uint8)
        # for i in pts_bird_b:
        #     print(i)
        #     cv2.circle(bird, (i[0][0],i[0][1]), 2, (255,255,255), 2)
       
        # for i in pts_bird_y:
        #     print(i)
        #     cv2.circle(bird, (i[0][0],i[0][1]), 2, (255,255,255), 2)
       
        return img

    @staticmethod
    def birdViewPts(blue_center: np.array, yellow_center: np.array, img):
        HomoMat = cameraMgr.getHomographyFromPts()
        # # riordino l'array
       
        if blue_center.shape[0] == 0:
            blue_center = np.array([[479,0]], dtype=int)

        if yellow_center.shape[0] == 0:
            yellow_center = np.array([[479,639]], dtype=int)
       
        pts_blue, pts_yellow = cameraMgr.sortedPoints(blue_center, yellow_center)
       
        # prendo solo 5 coni a destra e sinistra
        # if len(pts_blue)>12:
        #     pts_blue = pts_blue[len(pts_blue)-12:len(pts_blue)]
       
        # if len(pts_yellow)>12:
        #     pts_yellow = pts_yellow[len(pts_yellow)-12:len(pts_yellow)]
       
        #pts_blue = np.append(pts_blue,[np.array([0,479])],axis=0)
        #pts_yellow = np.append(pts_yellow,[np.array([639,479])],axis=0)
        #pts_yellow = pts_yellow[::-1]

       
        #pts_b = pts_blue.reshape((-1, 1, 2))
        #pts_y = pts_yellow.reshape((-1, 1, 2))
       
        applyHomo = lambda pts: cv2.convertPointsFromHomogeneous((HomoMat@cv2.convertPointsToHomogeneous(pts)[:,0,:].T).T).astype(int)

        pts_bird_b = applyHomo(pts_blue)
        pts_bird_y = applyHomo(pts_yellow)


       
        # def filter_cones(cone_pos, width=638):
        #     # suppondo siano orinati in maniera decrescente per coord x
        #     # cerco il primo indice
        #     xpos = cone_pos[:,0,0]
        #     xpos[::-1].sort()
           
        #     start = np.count_nonzero(xpos > width)
        #     start = start-1 if start >0 and (xpos[start-1]<1.2*width) else start

        #     end = np.count_nonzero(xpos < 0)
        #     end = end-1 if end >0 and (xpos[len(xpos)-end]<-.2*width) else end
        #     ppos = cone_pos[start:len(xpos)-end,:,:]
        #     return ppos[ppos[:,0,1]>-.2*width]

        print("Bird B: ",pts_bird_b)
        print("Bird Y: ",pts_bird_y)

        dist = lambda p : abs(p[0,0]-320)+abs(p[0,1]-480)
        select = lambda v: v[np.array([dist(ve) for ve in v])<980]
        pts_bird_b = select(pts_bird_b)
        pts_bird_y = select(pts_bird_y)
       
        if pts_bird_b.shape[0]>0 and pts_bird_y.shape[0]>0:
            roadCenterX = int((pts_bird_b[-1,0,0]+pts_bird_y[-1,0,0])/2)
            roadCenterL = np.array([[[roadCenterX-180,480]]])
            roadCenterR = np.array([[[roadCenterX+180,480]]])
            pts_bird_b = np.concatenate((pts_bird_b, roadCenterL),axis=0)
            pts_bird_y = np.concatenate((pts_bird_y, roadCenterR),axis=0)
       
        if pts_bird_b.shape[0]==0:
            pts_bird_b = np.array([[[0, 480]]])
        if pts_bird_y.shape[0]==0:
            pts_bird_y = np.array([[[640,480]]])
       
       
    #    idx_y = np.where((pts_bird_y[:,0,0] < 638) & (pts_bird_y[:,0,0]>0)) 
    #    print(idx_b)
        # plt.scatter(pts_bird_b[:,0,0],pts_bird_b[:,0,1], 5, "blue")
        # plt.scatter(pts_bird_y[:,0,0],pts_bird_y[:,0,1], 5, "orange")
       
        # plt.show(block = False)
        #plt.show()
        #pts_bird_b = filter_cones(pts_bird_b)
        #pts_bird_y = filter_cones(pts_bird_y)
        #print(idx_b)
        #print(idx_y)
       
       
        #pts_bird_y = np.delete(pts_bird_y, idx_y, axis=0)
        #pts_bird_b = np.delete(pts_bird_b, idx_b, axis=0)
       
        # pts_bird_b = pts_bird_b[idx_b]
        # pts_bird_y = pts_bird_y[idx_y]

        print("After Bird B: ",pts_bird_b)
        print("Aftr Bird Y: ",pts_bird_y)
       
        #lspace = np.linspace(0,  img.shape[1]-1, img.shape[1])

        #z_b = np.polyfit(pts_bird_b[:,0,0], pts_bird_b[:,0,1], 2)
        #z_y = np.polyfit(pts_bird_y[:,0,0], pts_bird_y[:,0,1], 2)

        #p_b = np.poly1d(z_b)
        #p_y = np.poly1d(z_y)

       
        # Generate a range of x values for a smooth curve
        #x_smooth_b = np.linspace(pts_bird_b[:,0,0].min(), pts_bird_b[:,0,0].max(), 5)
        #x_smooth_y = np.linspace(pts_bird_y[:,0,0].min(), pts_bird_y[:,0,0].max(), 5)

       
        #y_fit_b = p_b(pts_bird_b[:,0,0])
        #y_fit_y = p_y(pts_bird_y[:,0,0])
        #y_fit = np.polyval(z, lspace)   # evaluate the polynomial
        #y_fit = p(lspace)

        # #print(pts_bird_b[:,0,0])
        # y = pts_bird_b[::-1][:,0,1].tolist()
       
        # Counter(x)
       
        # print(duplicate)
        # l
        # y_new = interp1d(x,y,kind="quadratic")
        # Define the cubic spline interpolation
        #cs = CubicSpline(x, y, bc_type="natural")
       
       


       
        bird = np.zeros((480,640), dtype=np.uint8)
        #cv2.polylines(bird,[verts_b],False,(255,255,255),thickness=3)
        #cv2.polylines(bird,[verts_y],False,(255,255,255),thickness=3)
        # for i in pts_bird_b:
        #     #print(i)
        #     cv2.circle(bird, (i[0][0],i[0][1]), 2, (255,255,255), 2)
           
        # for i in pts_bird_y:
        #     #print(i)
        #     cv2.circle(bird, (i[0][0],i[0][1]), 2, (255,255,255), 2)
           
       
        #plt.scatter(pts_bird_b[:,0,0],pts_bird_b[:,0,1], 5, "blue")
        #plt.scatter(pts_bird_y[:,0,0],pts_bird_y[:,0,1], 5, 'red')
        #plt.scatter(cLine[:,0], cLine[:,1], 5, 'green')
        #plt.scatter(cLine[44,0], cLine[44,1], 5, 'black')
        #plt.show()

        #pts_bird_y = pts_bird_y[::-1]
       
        points = np.vstack((pts_bird_b, pts_bird_y[::-1]))
       
        cv2.polylines(bird, [pts_bird_b], isClosed=False,color=(255), thickness=3, lineType=cv2.LINE_AA)
        cv2.polylines(bird, [pts_bird_y], isClosed=False,color=(255), thickness=3, lineType=cv2.LINE_AA)
       
        cv2.floodFill(bird,None,(320,479),255)
        #cv2.fillPoly(bird, pts=[points], color=255)
        return bird


        #draw_points = (np.asarray([pts_bird_b[:,0,0], y_fit]).T).astype(np.int32)   # needs to be int32 and transposed

        #verts_b = np.array(list(zip(pts_bird_b[:,0,0].astype(int),y_fit_b.astype(int))))
        #verts_y = np.array(list(zip(pts_bird_y[:,0,0].astype(int),y_fit_y.astype(int))))
       
        #print(verts_b)
        #print(verts_y)
       
        # x = pts_bird_b[::-1][:,
    # Function to draw the center of the bounding box on the image

    @staticmethod
    def birdViewSpline(blue_center: np.array, yellow_center: np.array, img):
        def getLine(cones):
            """Torna un insieme di punti campione a partire da punti sparsi nell'immagine"""
            len=cones.shape[0]
            ii=scipy.interpolate.interp1d(np.arange(len), cones.squeeze(), axis=0, kind='linear',bounds_error=False, fill_value='extrapolate')
            pts=ii(np.arange(-len,2*len,.25))
            return pts.reshape(-1,1,2).astype(np.int32)
        HomoMat = cameraMgr.getHomographyFromPts()

        # Elimino coni lontani       
        if blue_center.shape[0] == 0:
            blue_center = np.array([[479,0]], dtype=int)

        if yellow_center.shape[0] == 0:
            yellow_center = np.array([[479,639]], dtype=int)
       
        pts_blue, pts_yellow = cameraMgr.sortedPoints(blue_center, yellow_center)

        # Trasforma la posizione dei coni in vista verticale       
        applyHomo = lambda pts: cv2.convertPointsFromHomogeneous((HomoMat@cv2.convertPointsToHomogeneous(pts)[:,0,:].T).T).astype(int)
        pts_bird_b = applyHomo(pts_blue)
        pts_bird_y = applyHomo(pts_yellow)
       

        # Dopo la vista verticale posso eliminare altri coni...
        dist = lambda p : abs(p[0,0]-320)+abs(p[0,1]-480)
        select = lambda v: v[np.array([dist(ve) for ve in v])<980]
        pts_bird_b = select(pts_bird_b)
        pts_bird_y = select(pts_bird_y)

        # Aggiungo coni in caso mancano nel numero sufficiente ???

        # if pts_bird_b.shape[0]>0 and pts_bird_y.shape[0]>0:
        #     roadCenterX = int((pts_bird_b[-1,0,0]+pts_bird_y[-1,0,0])/2)
        #     roadCenterL = np.array([[[roadCenterX-180,480]]])
        #     roadCenterR = np.array([[[roadCenterX+180,480]]])
        #     pts_bird_b = np.concatenate((pts_bird_b, roadCenterL),axis=0)
        #     pts_bird_y = np.concatenate((pts_bird_y, roadCenterR),axis=0)
       
        # if pts_bird_b.shape[0]==0:
        #     pts_bird_b = np.array([[[0, 480]]])
        # if pts_bird_y.shape[0]==0:
        #     pts_bird_y = np.array([[[640,480]]])
       
       
       
        bird = np.zeros((480,640), dtype=np.uint8)
       
        if pts_bird_b.shape[0]>1:
            cv2.polylines(bird, [getLine(pts_bird_b)], isClosed=False,color=(255), thickness=1, lineType=cv2.LINE_AA)
            id_b = np.where((getLine(pts_bird_b)[:,0,0]>=0) & (getLine(pts_bird_b)[:,0,0]<=640) & (getLine(pts_bird_b)[:,0,1]>=0) & (getLine(pts_bird_b)[:,0,1]<=480))[0]
            if id_b.shape[0]==0:
                point = (0, 479)
            else:
                print(getLine(pts_bird_b))
                point = (int(getLine(pts_bird_b)[id_b[-1]].squeeze()[0]+30),479)
       
        
            
        if pts_bird_y.shape[0]>1:
            cv2.polylines(bird, [getLine(pts_bird_y)], isClosed=False,color=(255), thickness=1, lineType=cv2.LINE_AA)
            id_y = np.where((getLine(pts_bird_y)[:,0,0]>=0) & (getLine(pts_bird_y)[:,0,0]<=640) & (getLine(pts_bird_y)[:,0,1]>=0) & (getLine(pts_bird_y)[:,0,1]<=480))[0]
            
            if id_y.shape[0]==0:
                point = (639, 479)
            else:
                print(getLine(pts_bird_y))
                print(int(getLine(pts_bird_y)[id_y[-1]].squeeze()[1])-1)
                point = (int(getLine(pts_bird_y)[id_y[-1]].squeeze()[0]-30),479)
       
        print(point)
        cv2.floodFill(bird,None,point,255)
        return bird





    @staticmethod
    def divide_centers(boxes_xywh, boxes_cls):
        yellow_center,blue_center = [],[]

        for box,cls in zip(boxes_xywh,boxes_cls):
            if cls==0:
                x_blue, y_blue, w, h = box
                blue_center.append([x_blue.cpu().item(),y_blue.cpu().item()])
            elif cls==4:
                x_yellow, y_yellow, w, h = box
                yellow_center.append([x_yellow.cpu().item(),y_yellow.cpu().item()])

        return np.array(blue_center), np.array(yellow_center)


    @staticmethod
    def sortedPoints(blue_center:np.array, yellow_center:np.array):
       
        # riordino l'array
        # sorted_indices_blue = np.argsort(blue_center[:, 1])
        # sorted_arr_blue = np.array(blue_center)[sorted_indices_blue].astype(np.int32)
       
        # sorted_indices_yellow = np.argsort(yellow_center[:, 1])
        # sorted_arr_yellow = np.array(yellow_center)[sorted_indices_yellow].astype(np.int32)

        # la colonna prima è quella con priorità più bassa
        sorted_indices_b = np.lexsort((blue_center[:, 0], blue_center[:, 1]))
        sorted_indices_y = np.lexsort((yellow_center[:, 0], yellow_center[:, 1]))
       
        pts_blue = blue_center[sorted_indices_b].astype(int)
        pts_yellow = yellow_center[sorted_indices_y].astype(int)
        # per le coordinate identiche riordino l'altra coordinata
        # unique_y = np.unique(sorted_arr_yellow[:, 1])
        # sorted_final_y = np.vstack([sorted_arr_yellow[sorted_arr_yellow[:, 1] == y][np.argsort(sorted_arr_yellow[sorted_arr_yellow[:, 1] == y][:, 0])] for y in unique_y])

        # pts_yellow = sorted_final_y[::-1] # necessario il reverse altrimenti non funge
       
        # unique_b = np.unique(sorted_arr_blue[:, 1])
        # sorted_final_b = np.vstack([sorted_arr_blue[sorted_arr_blue[:, 1] == y][np.argsort(sorted_arr_blue[sorted_arr_blue[:, 1] == y][:, 0])] for y in unique_b])

        # #pts_blue = sorted_final_b
        # pts_blue = sorted_final_b[::-1] # necessario il reverse altrimenti non funge

       
        #pts_blue = pts_blue[::-1]
       
        return pts_blue, pts_yellow



def rect_to_polar_relative(goal):

    """
    Funzione usata per la trasformazione in coordinate polari del goal
   
    Parameters:
    - robot_id: id del robot
   
    Returns:
    - r: raggio, ovvero la distanza tra robot e goal
    - theta: angolo di orientazione del robot rispetto al goal
    """
   
    # posizioe del goal
    goal = goal
   
    # calcolo la posizione correte del robot specificato
    robPos, car_orientation = p.getBasePositionAndOrientation(turtle)
    #print("pos: ",robPos)
    # calcolo l'angolo di yaw
    _,_,yaw = p.getEulerFromQuaternion(car_orientation)
   
    # Calculate the polar coordinates (distance, angle) of the vector
    vector_to_goal = np.array([goal[0] - robPos[0], goal[1] - robPos[1]])
    r = np.linalg.norm(vector_to_goal)
    theta = wrap2pi(np.arctan2(vector_to_goal[1], vector_to_goal[0])-wrap2pi(yaw))
    return r, theta


def p_control(yaw_error):
    kp = 0.9
    kv = 0.5
    vel_ang = kp*yaw_error
    vel_lin = 10       
 
    if abs(yaw_error)>0.1:
        vel_lin = (2.3-abs(vel_ang))*4.3
 
    return vel_ang, vel_lin




def choosePositionAndIndex(position,index):
    if len(position!=0):
        for i,j in zip(range(len(position)),position):
            r, yaw_error = rect_to_polar_relative(j[:2])
            vel_ang,_ = p_control(yaw_error)
            print(f"steer {vel_ang} - distance {r}")
            if vel_ang < 1.5:
                positionToStart = j[:2]
                indexToStart = index[i]
                done = True
            else:
                positionToStart = []
                indexToStart = None
                done = False
    else:
        positionToStart = []
        indexToStart = None
        done = False
    return positionToStart, indexToStart, done





def pixel_to_cartesian(pixel_coords, image_height):
        # Convert pixel coordinates to Cartesian coordinates
        cartesian_coords = pixel_coords.copy()
        cartesian_coords = image_height - pixel_coords
        return cartesian_coords





class Environment:
    model = None
    @staticmethod
    def load():
        global turtle
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[3,8,0])
        #p.resetDebugVisualizerCamera(cameraDistance=22, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])

        #p.loadURDF("plane.urdf", useFixedBase = True)
        p.loadURDF("world&car/plane/plane.urdf")

        turtle = p.loadURDF("world&car/simplecar.urdf", [-7.6,6,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(90)]))
        #turtle = p.loadURDF("world&car/simplecar.urdf", [0.1,-0.5,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(90)]))
        #turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
        #turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-9,-6.5,.3])
        #turtle = p.loadURDF("f10_racecar/simplecar.urdf", [21,0,.3],  p.getQuaternionFromEuler([0,0,np.deg2rad(-55)]))
        #turtle = p.loadURDF("f10_racecar/simplecar.urdf", [35.5,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)]))
        #track = p.loadSDF("f10_racecar/meshes/prova.sdf")

        # carica i coni fino L 190 COMPRESO
        #p.loadSDF("world&car/meshes/barca_track_modified.sdf", globalScaling=1)
        p.loadSDF("world/track.sdf",globalScaling = 1)
        #p.loadSDF("world/track_copy.sdf",globalScaling = 1)

        #p.loadSDF("world/cones_blue.sdf",globalScaling = 1)
        #p.loadSDF("world/cones_yellow.sdf",globalScaling = 1)

        #coni = p.loadURDF("world/test.urdf")
        #[p.changeDynamics(idx,-1, mass=0) for idx in track]

        env1 = [[-10,1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
        env2 = [[-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
        env3 = [[-9,-6.5,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(0)])]
        env4 = [[0,0,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(55)])]
        env5 = [[35,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)])]

        env_list = [env1,env2,env3,env4,env5]
        index_env = random.randint(0,len(env_list)-1)

        #turtle = p.loadURDF("world&car/simplecar.urdf", env_list[index_env][0],env_list[index_env][1])
        #turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-12,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)]))



        robPos, car_orientation = p.getBasePositionAndOrientation(turtle)

        # calcolo l'angolo di yaw
        _,_,yaw = p.getEulerFromQuaternion(car_orientation)
        #print("yaw: ",yaw)
           
        #turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
        #turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
        # sphere_color = [1, 0, 0,1]  # Red color
        # sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
        # sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[-10,5, 0])
        # p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)
        #track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
        #p.setRealTimeSimulation(1)


        # image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # image = np.transpose(image, (2, 0, 1))
        #print(image.shape)
        #print(image[np.newaxis,...].shape)
        #prediction = model(torch.from_numpy(image[np.newaxis,...]).type(torch.FloatTensor))

        #print(prediction)
        p.setGravity(0,0,-9.81)

        # sphere_color = [1, 0, 0,1]  # Red color
        # sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
        # sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[0,1,0 ])
        # p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)



        # for j in range (p.getNumJoints(turtle)):
        #        print(p.getJointInfo(turtle,j))
        Environment.model = YOLO('best.pt','gpu')
        forward=0
        turn=0


    def run():
        global turtle
        threshold = 0.25

        while (1):
            p.setGravity(0,0,-9.81)

            # testo un'immagine
            # per avere il real time del video (sorgente 2 era la zed2)
           

            # time.sleep(1./240.)
           
            # leftWheelVelocity=0
            # rightWheelVelocity=0
            # speed=10
               
            car_position, car_orientation = p.getBasePositionAndOrientation(turtle)
           
            #_,_,centerLine = computeTrack(debug=False)
            _,_,centerLine = getCones()
           
            positionToStart = []
            while len(positionToStart)==0:
                index,position = getPointToStart(centerLine,car_position[:2], threshold)
                positionToStart, indexToStart, done = choosePositionAndIndex(position,index)
                #print(positionToStart)
                threshold+=0.05

                   
            # creo le liste di indici
            indexList_1 = [idx for idx in range(indexToStart,len(centerLine),1)]
            indexList_2 = [idx for idx in range(0,indexToStart,1)]
           
            total_index = indexList_1+indexList_2

           
            for i in total_index:
                #print("Index: ",i)
                goal = centerLine[i][:2] # non prendo la z
                #print("Goal: ",goal)
                r, yaw_error = rect_to_polar_relative(goal)
                car_position, car_orientation = p.getBasePositionAndOrientation(turtle)

               
                while r>0.5:
                    # keys = p.getKeyboardEvents()
                    # for k,v in keys.items():

                    #     if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    #             turn = -0.5
                    #     if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                    #             turn = 0
                    #     if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    #             turn = 0.5
                    #     if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                    #             turn = 0

                    #     if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    #             forward=10
                    #     if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                    #             forward=0
                    #     if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                    #             forward=-10
                    #     if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                    #             forward=0
                    start1 = time.time()
                    img = cameraMgr.getCamera_image()
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                   
           
                    results = Environment.model.predict(img,conf=0.8)
                    boxes_xywh = results[0].boxes.xywh  # boxes mi da xywh di tutti i coni visti
                    boxes_cls = results[0].boxes.cls  # boxes mi da le classi di tutti i coni visti
                   

                    annotated_frame = results[0].plot()
                    blue_center, yellow_center = cameraMgr.divide_centers(boxes_xywh, boxes_cls)

                    #bird = birdViewImg(blue_center,yellow_center,img)
#                    bird = cameraMgr.birdViewPts(blue_center, yellow_center, img)
                    bird = cameraMgr.birdViewSpline(blue_center, yellow_center, img)
                   
                    # Display the annotated frame
                   
                    cv2.imshow("YOLOv8 Tracking", bird)
                    # Display the annotated frame
                    #cv2.imshow("YOLOv8 detect", annotated_frame)
                    #cv2.imshow('IMAGE', img)
                    k = cv2.waitKey(1)
                    if k==ord('p'):
                        #plt.show()
                        cv2.waitKey(0)
                    #bird_eye = birdEyeView(img)


                    #reshaped_image = cv2.resize(img, (96, 84))

                    r, yaw_error = rect_to_polar_relative(goal)
                    #print(f"goal {goal},distance {r} ,yaw_error{yaw_error}")
                    vel_ang,vel_lin = p_control(yaw_error)
                    forward = vel_lin
                    turn = vel_ang
                    #print(f"steer {vel_ang} - vel {vel_lin}")

                   
                   
                    # SIMPLE CAR
                    # turn positivo giro a sinistra
                    # turn negativo giro a destra
                    #forward = 20
                    p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=turn)
                    p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=turn)
                    p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
                    p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
                    p.setJointMotorControl2(turtle,4,p.VELOCITY_CONTROL,targetVelocity=forward)
                    p.setJointMotorControl2(turtle,5,p.VELOCITY_CONTROL,targetVelocity=forward)
                    #print("-----secondi-----", time.time()-start1)

                    for _ in range(24):
                        p.stepSimulation()


Environment.load()
Environment.run()