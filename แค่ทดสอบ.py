##เลือกภาพจาก index และตัดภาพ
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = [( -832 , 695 ),
( 1079 , 111 ),
( -577 , 947 ),
( 1100 , -142 ),
( -919 , -411 ),
( 794 , 618 ),
( -580 , 942 ),
( 1096 , -147 ),
( -933 , -383 ),
( 799 , 616 ),
( -554 , 966 ),
( 1103 , -152 ),
( -557 , 961 ),
( 1100 , -157 ),
( -602 , 922 ),
( 1093 , -137 ),
( -905 , -437 ),
( 790 , 622 ),
( -645 , 882 ),
( 1087 , -117 ),
( -600 , 926 ),
( 1096 , -133 ),
( -530 , 985 ),
( 1107 , -161 ),
( -610 , 924 ),
( 1103 , -105 ),
( -485 , 1017 )]
  

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

print(y_km)


##cap = cv2.VideoCapture("NO20180205-010358-000025.mp4") 
##cap.set(1,5000); 
##eat, frame = cap.read()
##height, width = frame.shape[:2]
##cropped_image = frame[int(height/2) : height, int(width/5) : width-int(width/5)]
##cv2.imshow("cropped", cropped_image)
##cv2.imwrite("frame%d.jpg" % 1, cropped_image)     


##หาวินาทีวิดีโอ
##import cv2
##cap = cv2.VideoCapture("videoplayback.mp4")
##fps = cap.get(cv2.CAP_PROP_FPS)     
##frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
##duration = frame_count/fps
##print('fps = ' + str(fps))
##print('number of frames = ' + str(frame_count))
##print('duration (S) = ' + str(duration))
##minutes = int(duration/60)
##seconds = duration%60
##print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
##cap.release()

##อ่านวิดีโอและแปลงเป็นภาพเก็บไว้ในคอม
##import cv2
##vidcap = cv2.VideoCapture('videoplayback.mp4')
##success,image = vidcap.read()
##count = 0
##while success:
##  cv2.imwrite("frame%d.jpg" % count, image)     
##  success,image = vidcap.read()
##  print('Read a new frame: ', success)
##  count += 1
##  if count == 10:
##      break
