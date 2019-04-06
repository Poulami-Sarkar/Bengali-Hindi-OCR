import pytesseract
import cv2
import sys
import math
import numpy as np
from os import listdir
from os.path import isfile, join
import re

def ocr(file): 
  # Define config parameters.
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l ben --oem 1 --psm 3')
 
  # Read image from disk
  im = cv2.imread(file, cv2.IMREAD_COLOR)
 
  # Run tesseract OCR on image
  text = pytesseract.image_to_string(im, config=config)
 
  # Print recognized text
  return(text)

filename = 'img/'
op = open('outputs/output.txt',"w")
for f in listdir('img/'):
    if re.match('\d.png',f):
        op.write(ocr(filename+f))
        op.write('\n')
        print(filename+f)

op.close()