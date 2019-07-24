from bengali import ocr
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

txtfile = sys.argv[2]

def scene(img):
    try:
        text,con = ocr(img,lang,0,0) 
    except:
        return
    if text != '':
        if len(re.findall('[a-z0-9]',text.lower())) > (len(re.findall('[\u0900-\u097Fa-zA-Z0-9]',text))*0.5) and (con.empty or con[1] <65):
            print('skip',len(re.findall('[a-z0-9]',text.lower())) > (len(re.findall('[\u0900-\u097Fa-zA-Z0-9]',text))*0.5) and con.empty,text)
        else:
            text_keychars = re.findall(['a-z0-9\u0900-\u097F'],text.lower())
            if text.lower()['a-z0-9\u0900-\u097F']



'''def frameno(f):
    return re.search('[1-9]\d*(\.\d+)?',f).group(0)

def get_images():
    l =listdir('scene/')
    d = dict()
    for i in range(0,len(l)):
        if re.match('^\d+.*.??g',l[i]):
            #print(l[i],frameno(l[i]))
            ts = float(frameno(l[i]))
            no = float(re.search('(\d+\.\d+).jpg',l[i]).group(0)[:-4])
            if ts in d:
                d[ts].append(no)
            else:
                d[ts] =[no]
    for time in d:
        d[time]=sorted(d[time])
    t = sorted(d.items())
    return t

def fetch_output(op):
    img = get_images()
    filename ='scene/'
    no=1
    for lst in img[:10]:
        print(lst[0])
        if len(lst) >1:    
            s,ms=divmod(lst[0],1000)
            m,s=divmod(s,60)
            h,m=divmod(m,60)
            op.write(str(2019011312)+str("%02d" %(m))+str("%02d" %(s))+str('.')+str("%03d" %(ms))+'|')
            s,ms = (s+2,ms+200) if ms+200<1000 else (s+3,ms+200-1000)
            m,s = (m+1,s-60) if s>=60 else (m,s)
            op.write(str(2019011312)+str("%02d" %(m))+str("%02d" %(s+2))+str('.')+str("%03d" %(ms))+'|'+'CC1|')
            for i in lst[1]:
                try:        
                    text,con = ocr(filename+str(lst[0])+'.'+str(i)+'0.jpg',lang,1,0) 
                    if text != '':
                        if len(re.findall('[a-z0-9]',text.lower())) > (len(re.findall('[\u0900-\u097Fa-z0-9]',text))*0.5) and (con.empty or con[1] <65):
                            print('skip',len(re.findall('[a-z0-9]',text.lower())) > (len(re.findall('[\u0900-\u097Fa-z0-9]',text))*0.5) and con.empty,text)
                        else:
                            op.write(text.replace('\r',' ').replace('\n',' '))
                            op.write('. ')
                except Exception as err:
                    print(err,text)
                    try:
                        text,con = ocr(filename+str(lst[0])+'.'+str(i)+'00.jpg',lang,1,0) 
                        if text != '':
                            op.write(text.replace('\r',' ').replace('\n',' '))
                            op.write('. ')
                    except:
                        print(str(lst[0])+'.'+str(i)+'0.jpg')
            no+=1
            op.write('\n')
op = open("outputs/output1.txt","w+")
lang = 'hin+eng'
fetch_output(op)

print(ocr('scene/'+'40.0.20.180.jpg',lang,1,0))
devanagari :2309-2416'''