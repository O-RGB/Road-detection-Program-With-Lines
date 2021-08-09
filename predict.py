import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.image import imread
import torch
from torchvision import transforms

from torch.autograd import Variable
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import cv2

    
class ImageClassificationModelBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.cross_entropy(out, targets)  
        print(loss)    
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        loss = F.cross_entropy(out, targets)  # Calculate loss
        acc = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_acc': acc }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,result['train_loss'], result['val_loss'], result['val_acc']))


class ImageClassificationModel(ImageClassificationModelBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #output 32 X 100 X 100 | (Receptive Field (RF) -  3 X 3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   #output 64 X 100 X 100 | RF 5 X 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 50 x 50 | RF 10 X 10

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 64 x 50 x 50 | RF 12 X 12
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 25 x 25  | RF 24 X 24
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 x 25 x 25  | RF 26 X 26
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 12 x 12 | RF 52 X 52
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1), #512* 10* 10 | RF 54 X 54
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 512 x 5 x 5 | RF - 108X 108
            

            nn.Flatten(),
            nn.Linear(512 * 5 * 5,10))
         
    def forward(self, xb):
        return self.network(xb)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
  
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)




def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images():
    from torch.utils.data.sampler import SubsetRandomSampler
    loader = DataLoader(cropped_image)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels



transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20, resample=Image.BILINEAR),
    transforms.RandomCrop(200),
    transforms.Resize((100,100)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('Road_model.pth',map_location ='cpu')
model.eval()


def srotD1(kmean1,kmean2):
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



from sklearn.cluster import KMeans
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


##หาจุดตัดของเส้น
def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    try:
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return round(px) ,round(py)
    except ZeroDivisionError:
        return 0,0


##ทำ Canny ทำ Houg 
import sys
import math
import cv2 as cv
import numpy as np
def ton(img):
    
    ##img = cv2.GaussianBlur(img,(3,3),0)
    ##median = cv.medianBlur(img, 5)
    
    ##ทำ Canny
    dst = cv.Canny(img, 50, 200, None, 3)

    ##ทำเส้น Canny ให้สมูท
##    kernel = np.ones((3,3),np.uint8)
##    dilation = cv2.dilate(dst,kernel,iterations = 1)
##    kernel = np.ones((11,11),np.uint8)
##    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
##    kernel = np.ones((3,3),np.uint8)
##    dst = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)


    ##Hough 
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    arraytest=[]
    kmeanP1 = []
    kmeanP2 = []
    ##Hough ขีดเส้น
    if lines is not None:
        for i in range(0, len(lines)):
            
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1500*(-b)), int(y0 + 1500*(a)))
            pt2 = (int(x0 - 1500*(-b)), int(y0 - 1500*(a)))
            try:
                sun = abs((pt2[1] - pt1[1])/(pt2[0] - pt1[0]))
            except ZeroDivisionError:
                sun = 0
            
            if (sun > 0.25 and sun < 0.8):
                ##ทดสอบเก็บค่า XY แต่ละรอบ
                kmeanP1.append(pt1)
                kmeanP2.append(pt2)
                
                arraytest.append(pt1)
                arraytest.append(pt2)
                ##วาดเส้นลงรูป
                ##cv.line(img2, pt1, pt2, (255,0,255), 2, cv.LINE_AA)
    if (len(kmeanP1)+len(kmeanP2)) < 4:
        return img
    kmeanP1 = kmean(np.array(kmeanP1))
    kmeanP2 = kmean(np.array(kmeanP2))
    
    kmeanP1,kmeanP2 = srotD1(kmeanP1,kmeanP2)

    
    pt1 = (int(kmeanP1[0][0]), int(kmeanP1[0][1]))
    pt3 = (int(kmeanP2[0][0]), int(kmeanP2[0][1]))

    pt2 = (int(kmeanP1[1][0]), int(kmeanP1[1][1]))
    pt4 = (int(kmeanP2[1][0]), int(kmeanP2[1][1]))

    cv.line(img, pt1, pt3, (255,0,0), 3, cv.LINE_AA)
    cv.line(img, pt2, pt4, (255,0,0), 3, cv.LINE_AA)

    
    ##ทำให้เส้นสูงขึ้นเพื่อ mask เส้นที่ 1
    vertices1 = np.array([pt1,pt3,[pt3[1]-1000,pt1[0]-1000]],np.int32)
    ##vertices = np.array([pt1,pt3,[pt3[0]-1000,pt1[1]-2000]],np.int32)
    pts1 = vertices1.reshape((-1, 1, 2))
    cv2.polylines(img, [pts1], isClosed=True, color=(0, 0, 0), thickness=20)
    cv2.fillPoly(img, [pts1], color=(0, 0, 0))

    

    ##ทำให้เส้นสูงขึ้นเพื่อ mask เส้นที่ 2
    vertices2 = np.array([pt2,pt4,[pt4[1]+1000,pt2[0]+1000]],np.int32)
    ##vertices = np.array([pt2,pt4,[pt4[0]-500,pt2[1]-3000]],np.int32)
    pts2 = vertices2.reshape((-1, 1, 2))
    cv2.polylines(img, [pts2], isClosed=True, color=(0, 0, 0), thickness=20)
    cv2.fillPoly(img, [pts2], color=(0, 0, 0))
    plt.imshow(img)
    return img



img = cv2.imread("testimg/5/3.jpg")
height, width = img.shape[:2]
##img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##เริ่มต้นการทำนาย AI
im_pil = Image.fromarray(img)
image_tensor = transforms(im_pil).float()
image_tensor = image_tensor.unsqueeze_(0)
input = Variable(image_tensor)
input = input.to(device)
output = model(input)
index = output.data.cpu().numpy().argmax()
classesFile = ['CRACK_ROAD', 'POTHOLE_ROAD', 'ROAD']

cv2.imwrite("testimg/5/%s_1.jpg" % (classesFile[index]), img)
 



