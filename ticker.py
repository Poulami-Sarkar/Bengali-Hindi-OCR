import pytesseract
import cv2
import sys
import math
import pickle
from os import listdir
import numpy as np
from bengali import ocr,fetch_output

im = cv2.imread('img/input.png', cv2.IMREAD_COLOR)

confThreshold = 0.5
nmsThreshold = 0.5
inpWidth = 480
inpHeight = 320
model = "frozen_east_text_detection.pb"

net = cv2.dnn.readNet("frozen_east_text_detection.pb")

kWinName = "Text Detector running"
#cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)
outNames = []
outNames.append("feature_fusion/Conv_7/Sigmoid")
outNames.append("feature_fusion/concat_3")

def hash(vertex):
  return int(vertex/10)

def ticker_detect(vertices,ticker):
  if int(vertices[1][1]) >= 460:
    # is a ticker
    if ticker[2] > vertices[1][1]:
      ticker[2] = vertices[1][1]
    if ticker[3] < vertices[0][1]:
      ticker[3] = vertices[0][1]
    if ticker[0] > vertices[0][0] and vertices[0][0] > 0:
      ticker[0]  = vertices[0][0]   
    if ticker[1] < vertices[2][0] and vertices[2][0] < 640:
      ticker[1] = vertices[2][0]
    return [1,ticker]
  else :
    return [0,ticker]
       

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []
    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]
            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue
            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])
            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))
    # Return detections and confidences
    return [detections, confidences]

cap = ap = cv2.VideoCapture(str(sys.argv[1]))
op = open('outputs/output.txt',"w+")

#frame no
print("press 1 to quit")
def detect_text(file):
    no = 109
    cap = ap = cv2.VideoCapture(file)
    while cv2.waitKey(1) < 0:
        no+=1
        backup=0
        hasFrame, frame = cap.read()
        
        if no%110 != 0 :
          backup =1
          if no%114 != 0:
            continue
        
        
        arg =len(sys.argv)
        # Read frame 
        if not hasFrame:
            cv2.waitKey()
            break
        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)
        # Create a 4D blob from frame.
        blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        # Run the model
        net.setInput(blob)
        outs = net.forward(outNames)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)
        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        
        #ticker coordinates x1:x2,y1:y2
        ticker =[999,0,999,0]
        
        for i in indices:
            snip = 0
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            #rint(p1," ",p2)
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            print(no)
            print(ticker)
            resp,ticker = ticker_detect(vertices,ticker)
            # if ticker is detected skip
            if resp == 0:
                continue
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                #if arg <=2:
                    #cv2.line(frame, p1, p2, (0, 255, 0), 2);
            #print(vertices)
            cropped = frame[math.floor(vertices[1][1])-4:math.ceil(vertices[3][1]+4),math.floor(vertices[1][0])-4:math.ceil(vertices[3][0])+4]
            if arg >2:
                cv2.imwrite('img/f-'+'-'+str(hash(vertices[1][1]))+'.'+str(hash(vertices[1][0]))+'.jpg',cropped)
        ## OCR
        if  arg >2:
            if str(sys.argv[2]) == 'O':
                fetch_output(op)
                op.write('\n')
        # Put efficiency information
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        # Display the frame
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cropped = frame[int(453):int(485),int(1):int(500)]
        if backup == 0:
          cv2.imwrite('img/tick'+'-'+str(no)+'.jpg',cropped)
          prev = no
        else:
          cv2.imwrite('backup/f-'+'-'+str(prev)+'tick.jpg',cropped)
        #cv2.destroyAllWindows()
        cv2.imshow(kWinName,frame)
    op.close()   
    print("done")
    print("Writing")
    print(no)

detect_text('video/2019-01-13_1230_IN_DD-News_Desh_Pradesh.mp4')
'''
for file in listdir("video"):

    detect_text("video/"+file)
#fetch_output()'''