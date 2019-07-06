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
import pandas as pd

def ocr(file,option,d): 
  # Define config parameters.
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l ben --oem 1 --psm 3')
  if option == 1:
    # Read image from disk
    im = cv2.imread(file, cv2.IMREAD_COLOR)
  else :
    im = file
   
  '''temp = im
  temp = cv2.bitwise_not(temp)
  temp = cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
  thresh = 127
  temp = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)[1]
  temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY_INV)[1]
  con = pytesseract.image_to_data(temp, output_type='data.frame')
  con = con[con.conf != -1]
  con = con.groupby(['block_num'])['conf'].mean()
  text = pytesseract.image_to_string(temp, config=config)'''
  temp = im
  temp = cv2.fastNlMeansDenoisingColored(temp,None,20,10,7,21)
  temp = cv2.fastNlMeansDenoising(temp,None,10,7,21)
  temp = cv2.bitwise_not(temp)
  temp = cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
  thresh = 127
  temp = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)[1]
  #temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY_INV)[1]
  con = pytesseract.image_to_data(temp, output_type='data.frame')
  con = con[con.conf != -1]
  con = con.groupby(['block_num'])['conf'].mean()
  text = pytesseract.image_to_string(temp, config=config)

  temp1 =im
  temp1 = cv2.fastNlMeansDenoisingColored(temp1,None,20,10,7,21)
  temp1 = cv2.fastNlMeansDenoising(temp1,None,10,7,21)
  temp1 = cv2.bitwise_not(temp1)
  temp1 = cv2.resize(temp1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
  thresh = 127
  temp1 = cv2.threshold(temp1, thresh, 255, cv2.THRESH_BINARY)[1]
  temp1 = cv2.threshold(temp1, 0, 255, cv2.THRESH_BINARY_INV)[1]  
  con1 = pytesseract.image_to_data(temp1, output_type='data.frame')
  con1 = con1[con1.conf != -1]
  con1 = con1.groupby(['block_num'])['conf'].mean()    
  text1 = pytesseract.image_to_string(temp1, config=config) 
  # Run tesseract OCR on image
  f=0
  if con.empty and text != '' and con1.empty and text1 != '':
    print("no conf ",file,text,text1)
    return text
  if con.empty and con1.empty:
    if text1 != '':
      return text1  
    else: return text
  elif con1.empty:
    con1 =con
    temp1 =temp
    f =1
  elif con.empty:
    con =con1
    temp =temp1
 
  if (con[1] <40) and (con1[1]< 40):
    print(file)
    print('low',con1[1], con[1])
    #return (text)
  if con[1] > con1[1]:
    text = text
    print(con[1])
  elif con1[1] >con[1]:
    text = text1    
    print(con1[1])
  #print(text)
  # Print recognized text
  return(text)

filename = ''
print("text")
er = open('outputs/output.txt',"w+")
op = open('outputs/output.srt',"w+")
'''file = 'img/tick-594040.0.jpg'
text =(ocr(filename+file,1,1))
if text == '':
  text = (ocr(filename+file,1,0))
#op.write(text)'''
#print(text)

def writefile(h,m,s,ms,no,f,text):
  op.write(str(no))
  op.write('\n')
  op.write(str(h)+':'+str(m)+':'+str(s)+','+str(ms)+' --> '+str(h)+':'+str(m)+':'+str(s+2)+','+str(ms+200))
  op.write('\n')
  op.write(str(text).replace('\n',' '))
  op.write('\n\n')

def frameno(f):
  return re.search('[1-9]\d*(\.\d+)?',f).group(0)

def fetch_output(op):
  filename = 'img/'
  print("Writing")
  l =listdir('img/')
  for i in range(0,len(l)):
    if re.match('.*\.??g',l[i]):
      l[i] = float(frameno(l[i]))
  l = sorted(l)
  #l = list(map(lambda x:'tick-'+str(x)+'.jpg',sorted(l)))
  prev='p'
  no = 1
  for f in l[:]:
    s,ms=divmod(f,1000)
    m,s=divmod(s,60)
    h,m=divmod(m,60)
    f = 'tick-'+str(f)+'.jpg'
    try:
      text = ocr(filename+f,1,1)
      if "".join(text.split()) == '':
        raise Exception('blank')
      text = text.split(' ')
      '''
      if (prev[len(prev)-1][0] != text[0][0]):
        op.write(prev[len(prev)-1])
        op.write(text[0]+' ')  
        print(prev[len(prev)-1][0],text[0][0])'''
      stripped = " ".join(text[0:len(text)])
      prev = text
      writefile(h,m,s,ms,no,f,stripped)      
      no+=1
    except:
      try:
        text =ocr('backup/'+f,1,1)
        writefile(h,m,s,ms,no,f,text)
        no+=1
        op.write('\n')
        print('try',f,text)
      except Exception as err:
        er.write(f+' '+ str(err))
        print(err)
        #op.write("ERROR "+f)
        #op.write("\n")
        er.write('\n')
          #os.remove(filename+f)'''

#op.close()

fetch_output(op)
