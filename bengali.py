import pytesseract
import cv2
import sys
import math
import numpy as np
import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import re

def ocr(file,option): 
  # Define config parameters.
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l eng+hin --oem 1 --psm 3')
  if option == 1:
    # Read image from disk
    im = cv2.imread(file, cv2.IMREAD_COLOR)
  else :
    im = file
  im = cv2.bitwise_not(im)
  im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
  thresh = 127
  im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
  # Run tesseract OCR on image
  
  text = pytesseract.image_to_string(im, config=config)
 
  # Print recognized text
  return(text)

filename = 'img/'
print("text")
op = open('outputs/output.txt',"w+")
#op.write(ocr(filename+'f--6160tick.jpg',1))

def frameno(f):
  return re.search('\d+',f).group(0)

def fetch_output(op):
  filename = 'img/'
  #op = open('outputs/output.txt',"w+")
  print("Writing")
  l =listdir('img/')
  for i in range(0,len(l)):
    if re.match('.*\.??g',l[i]):
      l[i] = int(frameno(l[i]))
  l = list(map(lambda x:'tick-'+str(x)+'.jpg',sorted(l)))

  for f in l:
    try:
      op.write(ocr(filename+f,1))
      op.write('\n')
    except:
      #op.write(ocr('backup/'+f,1))
      op.write('\n')
      print(f)
          #os.remove(filename+f)

  op.close()

fetch_output(op)
