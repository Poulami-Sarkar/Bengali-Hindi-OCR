import pytesseract
import cv2
import sys
import math
import numpy as np
from datetime import datetime,timedelta
import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import re
import pandas as pd


base_dir ="/mnt/"
#base_dir =""                                                   #Uncomment to run locally

lang ='hin+eng'
def ocr(file,lang,option,d): 
  # Define config parameters.
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l '+lang+' --oem 1 --psm 3')
  if option == 1:
    # Read image from disk
    im = cv2.imread(file, cv2.IMREAD_COLOR)
  else :
    im = file
  
  if d == 1:
    temp = im
    temp = cv2.bitwise_not(temp)
    temp = cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thresh = 127
    temp = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)[1]
    temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY_INV)[1]
    con = pytesseract.image_to_data(temp, output_type='data.frame')
    con = con[con.conf != -1]
    con = con.groupby(['block_num'])['conf'].mean()
    text = pytesseract.image_to_string(temp, config=config)
  else:
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
  #Comment for Bengali 
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

  # Test conditions
  f=0
  if con.empty and text != '' and con1.empty and text1 != '':
    #print("no conf ",file,text,text1)
    return (text,con)
  if con.empty and con1.empty:
    if text1 != '':
      #print(1)
      return (text1,con1)  
    else: return (text,con)
  elif con1.empty and text !='':
    con1 =con
    return (text,con)
  elif con.empty and text1 !='':
    con =con1
    return (text1,con1)

  #if (con[1] <40) and (con1[1]< 40):
    #print(file)
    #print('low',con1[1], con[1])
    #return (text)
  if con[1] > con1[1]:
    text = text
    #print(con[1])
  elif con1[1] >con[1]:
    text = text1
    con = con1    
    #print(con1[1])
  #print(text)
  # Print recognized text
  return(text,con)

filename = ''
print("text")
er = open(base_dir+'outputs/output1.txt',"w+")
op = open(base_dir+'outputs/output1.srt',"w+")

def writefile(op,boxes,no,ms,base,text,lang):
  start = base+timedelta(milliseconds=ms)
  end = end = start + timedelta(milliseconds = 2200)
  st = int(''.join(re.findall('\d',str(start))))/1000000
  en = int(''.join(re.findall('\d',str(end))))/1000000
  # Modify text
  if lang =='ben' and len(text.split(' '))>2:
    text =text.split(' ')[1:-1]
    text = str(' '.join(text)).replace('\n',' ')
  # Write output to file
  #print(boxes)
  op.write(str("%.3f"%round(st,3)) +'|'+str("%.3f"%round(en,3))+'|TIC2|'+str("%06d" %no)+'|'+\
    str("%03d" %int(boxes[0]))+' '+str("%03d" %int(boxes[2]))+' '+str("%03d" %abs(boxes[1]-boxes[0]))+' '+str("%03d" %abs(boxes[3]-boxes[2]))+'|')
  print(text)
  op.write(text.replace('\n',' ').replace('\r',' ')+'\n')


## fetch_output(file,boxes,timestamp)

def ocr_ticker(op,boxes,no,ts,base,lang):
  text=''
  try:
    text,con = ocr(base_dir+'tickimg.jpg',lang,1,1)
    if "".join(text.split()) == '':
      raise Exception('blank')
    writefile(op,boxes,no,ts,base,text,lang)     
    os.remove(base_dir+'tickimg.jpg')
    os.remove(base_dir+'backup.jpg') 
  except:
    #Execute backup if tickimg is blank or exception
    try:
      text,con =ocr(base_dir+'backup.jpg',lang,1,1)
      if text != '':
        writefile(op,boxes,no,ts,base,text,lang)
      os.remove(base_dir+'tickimg.jpg')
      os.remove(base_dir+'backup.jpg') 
    except Exception as err:
      er.write(str(no)+str(err))
      print(err)
      er.write('\n')
  
  #return (text,con)
