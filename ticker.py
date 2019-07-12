import pytesseract
import cv2
import sys
import math
import pickle
from os import listdir
import numpy as np
#from bengali import ocr,fetch_output

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
  return int(vertex/10)*10

def find_boxes(vertices,array):
  y = hash(vertices[1][1])
  if y in array:
    array[y] = ticker_detect(vertices,array[y])
  else:
    array[y] = [vertices[1][0],vertices[3][0],vertices[1][1],vertices[3][1]]
  return array

def ticker_detect(vertices,ticker):
  # is a ticker
  if ticker[2] > vertices[1][1]:
    ticker[2] = vertices[1][1]
  if ticker[3] < vertices[0][1]:
    ticker[3] = vertices[0][1]
  if ticker[0] > vertices[0][0]:
    if ticker[1]+150<vertices[0][0]:
      return ticker
    if vertices[0][0] < 0: 
      ticker[0] = 0
    else:
      ticker[0]  = vertices[0][0] 
  if ticker[1] < vertices[2][0] and vertices[2][0] < 640:
    ticker[1] = vertices[2][0]
  return ticker
       
def color_detect_ticker(frame):
    if frame.size == 0:
      return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    blue_lower=np.array([99,115,150],np.uint8)
    blue_upper=np.array([110,255,255],np.uint8) 
    red_lower=np.array([136,87,111],np.uint8)
    red_upper=np.array([180,255,255],np.uint8)

    blue = cv2.inRange(hsv, blue_lower, blue_upper) 
    red = cv2.inRange(hsv, red_lower, red_upper) 

    kernal = np.ones((5 ,5), "uint8")
    blue=cv2.dilate(blue,kernal)
    red=cv2.dilate(red,kernal)
    resb=cv2.bitwise_and(frame, frame, mask = blue)
    resr=cv2.bitwise_and(frame, frame, mask = red)

    if (np.count_nonzero(resb)>1500 and np.count_nonzero(resr)>9000):
      return 1
    else :
      return 0

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

print("press 1 to quit")
def detect_text(file):
    no = 109
    cap = ap = cv2.VideoCapture(file)
    while cv2.waitKey(1) < 0:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
          break
        no+=1
        backup=0
        hasFrame, frame = cap.read()
        
        if no%(110) != 0 :
          backup =1
          if (no-10)%(110) != 0:
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
        array ={}
        
        for i in indices:
            snip = 0
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            #rint(p1," ",p2)
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            #print(no)
            
            if int(vertices[1][1]) >= 450:
              ticker = ticker_detect(vertices,ticker)
              resp = 1
              continue
            else :  resp =0
            array = find_boxes(vertices,array)
            '''
            try:
              array = find_boxes(vertices,array)
            except:
              array ={}
              array = find_boxes(vertices,array)'''
            
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                if arg <2:
                  cv2.line(frame, p1, p2, (0, 255, 0), 2);
            # if ticker is detected skip
            if resp == 1:
                continue            
            #print(vertices)
            cropped = frame[math.floor(vertices[1][1])-4:math.ceil(vertices[3][1]+4),math.floor(vertices[1][0])-4:math.ceil(vertices[3][0])+4]
            #if arg >2:
              #  cv2.imwrite('img/f-'+str(hash(vertices[1][1]))+'.'+str(hash(vertices[1][0]))+'.jpg',cropped)
        ## OCR
        '''if  arg >2:
            if str(sys.argv[2]) == 'O':
                fetch_output(op)
                op.write('\n')'''
        # Put efficiency information
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
        #cropped = frame[int(453):int(485),int(1):int(500)]
        cropped = frame[int(ticker[2]):int(ticker[3]),int(1):int(500)]
        if color_detect_ticker(cropped):
          array[int(ticker[2])] = [110,600,ticker[2],ticker[3]]
          '''cropped = frame[int(ticker[2]):int(ticker[3]),int(110):int(600)]
          cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
          cv2.imwrite('scene/'+str(int(cap.get(cv2.CAP_PROP_POS_MSEC)))+'.'+str(hash(ticker[2]))+'.'+str(110)+'.jpg',cropped)'''
          cropped = np.empty(0)
        
        # Convert to grayscale
        if cropped.size: cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        print(no)
        print(ticker)
        if backup == 0:
          cv2.imwrite('img/tick'+'-'+str(cap.get(cv2.CAP_PROP_POS_MSEC))+'.jpg',cropped)
          prev = cap.get(cv2.CAP_PROP_POS_MSEC)
        else:
          cv2.imwrite('backup/tick-'+str(prev)+'.jpg',cropped)
          print(array)
          if len(array)>1:
            for i in array.values():
              boxes = i
              print(boxes)
              cropped = frame[int(boxes[2]):int(boxes[3]),int(boxes[0]-4):int(boxes[1])+4]
              cv2.imwrite('scene/'+str(prev)+'.'+str(hash(boxes[2]))+'.'+str(hash(boxes[0]))+'.jpg',cropped)
              array ={}
        #cv2.destroyAllWindows()
        cv2.imshow(kWinName,frame)
    print("done")
    print("Writing")
    print(no)

detect_text('video/2019-01-05_1300_IN_DD-News_Nationwide.mp4')
'''
for file in listdir("video"):

    detect_text("video/"+file)
#fetch_output()'''
