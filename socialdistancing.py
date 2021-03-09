import numpy as np
import cv2
import imutils
import time
import math

l="./coco.names"
labels=open(l).read().strip().split("\n")
np.random.seed(42)
colors=np.random.randint(0,255,size=(len(labels),3),dtype="uint8")
weights="./yolov3.weights"
config="./yolov3.cfg"
print("************loading model************")
net=cv2.dnn.readNetFromDarknet(config,weights)

print("********starting video*********")
cam=cv2.VideoCapture(0)
while (cam.isOpened()):
    _,img=cam.read()
    img=imutils.resize(img,width=800)
    (h,w)=img.shape[:2]
    ln=net.getLayerNames()
    ln=[ln[i[0]-1]for i in net.getUnconnectedOutLayers()]
    blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    start=time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Prediction time : {:.6f} seconds".format(end - start))
    boxes=[]
    confi=[]
    classids=[]
    for output in layerOutputs:
     for detection in output:
        scores=detection[5:]
        classid=np.argmax(scores)
        confid=scores[classid]
        if confid>0.1 and classid==0:
            box=detection[0:4]*np.array([w,h,w,h])
            (centerx,centery,width,height)=box.astype("int")
            x=int(centerx-(width/2))
            y=int(centery-(height/2))
            boxes.append([x,y,int(width),int(height)])
            confi.append(float(confid))
            classids.append(classid)

    idxs=cv2.dnn.NMSBoxes(boxes,confi,0.5,0.3)
    ind=[]
    for i in range(0,len(classids)):
        if(classids[i]==0):
            ind.append(i)

    a=[]
    b=[]

    if len(idxs)>0:
        for i in idxs.flatten():
            (x,y)=(boxes[i][0],boxes[i][1])
            (w,h)=(boxes[i][2],boxes[i][3])
            a.append(x)
            b.append(y)

    dist=[]
    nsd=[]
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                xx=(a[k]-a[i])
                yy=(b[k]-b[i])
                d=math.sqrt(xx*xx+yy*yy)
                dist.append(d)
                if(d<=220):
                    nsd.append(i)
                    nsd=append(k)
                    nsd=list(list.fromkeys(nsd))
                    print(nsd)
    color=(0,0,255)
    for i in nsd:
        (x,y)=(boxes[i][0],boxes[i][1])
        (w,h)=(boxes[i][2],boxes[1][3])
        cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
        text="RED ALERT!!!"
        cv2.putText(img,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,3)
    color=(0,255,0)
    if len(idxs)>0:
        for i in idxs.flatten():
            if (i in nsd):
                break;
            else:
                (x,y)=(boxes[i][0],boxes[i][1])
                (w,h)=(boxes[i][2],boxes[1][3])
                cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
                text="NORMAL"
                cv2.putText(img,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,3)
    cv2.imshow("SOCIAL DISTANCING DETECTOR",img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cam.release()
cv2.destroyAllWindows()
