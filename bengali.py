import pytesseract
import cv2
import sys
import math
import numpy as np
from os import listdir
from os.path import isfile, join
import re

def ocr(file,option): 
  # Define config parameters.
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l ben+eng --oem 1 --psm 3')
  if option == 1:
    # Read image from disk
    im = cv2.imread(file, cv2.IMREAD_COLOR)
  else :
        im = file
  # Run tesseract OCR on image
  text = pytesseract.image_to_string(im, config=config)
 
  # Print recognized text
  return(text)


#print(ocr(filename+'cr2.png',0))

def fetch_output():
  filename = 'img/'
  op = open('outputs/output.txt',"w")
  print("Writing")
  for f in listdir('img/'):
      if re.match('.*\.??g',f):
          op.write(ocr(filename+f,1))
          op.write('\n')

  op.close()
