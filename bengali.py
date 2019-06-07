import pytesseract
import cv2
import sys
import math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re

def ocr(file,option): 
  # Define config parameters.
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l eng --oem 1 --psm 3')
  if option == 1:
    # Read image from disk
    im = cv2.imread(file, cv2.IMREAD_COLOR)
  else :
    im = file
  # Run tesseract OCR on image
  
  text = pytesseract.image_to_string(im, config=config)
 
  # Print recognized text
  return(text)

filename = 'img/'
print("text")
#print(ocr(filename+'f-1-14.0202.487.0994.jpg',1))

def fetch_output(op):
  filename = 'img/'
  #op = open('outputs/output.txt',"w+")
  print("Writing")
  l =sorted(listdir('img/'))
  for f in l:
      if re.match('.*\.??g',f):
          try:
            op.write(ocr(filename+f,1))
            op.write(' ')
          except:
            print(f)
          os.remove(filename+f)

  #op.close()

#fetch_output()
