import sys
import math
import cv2 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def sortLineKmean(kmean1,kmean2):
    tempL = []
    tempR = []
    for i in range(len(kmean1)-1):
        if kmean1[0][0] < 0:
            tempL.append(kmean1[0])
        if kmean1[1][0] < 0:
            tempL.append(kmean1[1])
        if kmean2[0][0] < 0:
            tempL.append(kmean2[0])
        if kmean2[1][0] < 0:
            tempL.append(kmean2[1])

    for i in range(len(kmean1)-1):
        if kmean1[0][0] > 0:
            tempR.append(kmean1[0])
        if kmean1[1][0] > 0:
            tempR.append(kmean1[1])
        if kmean2[0][0] > 0:
            tempR.append(kmean2[0])
        if kmean2[1][0] > 0:
            tempR.append(kmean2[1])

    for i in range(len(kmean1)-1):
        if tempL[0][1] < tempL[1][1]:
            tempL = [  tempL[1],tempL[0]  ]
        else:
            tempL = [  tempL[0],tempL[1]  ]
            
        if tempR[0][1] > tempR[1][1]:
            tempR = [  tempR[1],tempR[0]  ]
        else:
            tempR = [  tempR[0],tempR[1]  ]
            
    return np.array(tempL),np.array(tempR)

def kmean(kmean):
    try:
        km = KMeans(
            n_clusters=2, init='random',
            n_init=2, max_iter=300, 
            tol=1e-04, random_state=0
        )
        km.fit_predict(kmean)
        return km.cluster_centers_
    except ValueError:
        return 0

def road_detection(img):
    ##ทำ Canny
    
    ##***** ใหม่
    img_clone = img.copy()
    ##img = cv2.GaussianBlur(img, (5,5),1.4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr7 = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    ##***** ใหม่
    
    dst = cv2.Canny(img, ret, 200, None, 3)
    cv2.imshow("F",dst) 
    ##ทำเส้น Canny ให้สมูท
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dilation = cv2.dilate(dst,kernel,iterations = 1)

    kernel = np.ones((11,11),np.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    
       
    ##Hough 
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    kmeanP1 = []
    kmeanP2 = []
    
    ##Hough คัดเลือกเส้น
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            try:
                ##หาความชัน
                sun = abs((pt2[1] - pt1[1])/(pt2[0] - pt1[0]))
            except ZeroDivisionError:
                ##กรณีที่ไม่มีเส้นสักเส้น
                sun = 0
            
            if (sun > 0.25 and sun < 0.8):
                ##ทดสอบเก็บค่า XY ทุกเส้นในแต่ละรอบ
                kmeanP1.append(pt1)
                kmeanP2.append(pt2)
                
    plt.scatter(kmeanP1, kmeanP2)
    plt.show()
    if (len(kmeanP1)+len(kmeanP2)) < 4:
        return img_clone

    ##หาค่าเฉลี่ยของจุดทั้งหมด
    kmeanP1 = kmean(np.array(kmeanP1))
    kmeanP2 = kmean(np.array(kmeanP2))
    plt.scatter(kmeanP1, kmeanP2)
    plt.show()
    ##เรียงเส้นใหม่
    kmeanP1,kmeanP2 = sortLineKmean(kmeanP1,kmeanP2)
    
    pt1 = (int(kmeanP1[0][0]), int(kmeanP1[0][1]))
    pt3 = (int(kmeanP2[0][0]), int(kmeanP2[0][1]))
    pt2 = (int(kmeanP1[1][0]), int(kmeanP1[1][1]))
    pt4 = (int(kmeanP2[1][0]), int(kmeanP2[1][1]))

    ##เปลี่ยนเป็นรูปไม่เบลอ
    img = img_clone
    cv2.line(img, pt1, pt3, (255,0,0), 3, cv2.LINE_AA)
    cv2.line(img, pt2, pt4, (255,0,0), 3, cv2.LINE_AA)

    ##ทำให้เส้นสูงขึ้นเพื่อ mask เส้นที่ 1
    vertices = np.array([pt1,pt3,[pt3[0]-1000,pt1[1]-2000]],np.int32)
    pts = vertices.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=20)
    cv2.fillPoly(img, [pts], color=(0, 0, 0))

    ##ทำให้เส้นสูงขึ้นเพื่อ mask เส้นที่ 2
    vertices = np.array([pt2,pt4,[pt4[0]-500,pt2[1]-3000]],np.int32)
    pts = vertices.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=20)
    cv2.fillPoly(img, [pts], color=(0, 0, 0))
    plt.imshow(img)
    return img


def removeBlackBackground(image):
    ##ตัดส่วนที่เป็นสีดำออกให้เหลือแต่ถนน
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]



if __name__ == "__main__":
    img = cv2.imread("Untitled2.png")
    
    ##ส่งไป road_detection และ ลบพื้นสีดำออก removeBlackBackground
    cdst = removeBlackBackground(road_detection(img))

    ##สำหรับแสดงผล ไม่มีผลต่อภาพ
    resized = cv2.resize(cdst, (800,500), interpolation = cv2.INTER_AREA)

    
    plt.imshow(resized[0])
    plt.show()
    
    cv2.imshow("Road Detection",resized)
