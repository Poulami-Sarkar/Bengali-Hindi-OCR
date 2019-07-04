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


def frameno(f):
    return re.search('[1-9]\d*(\.\d+)?',f).group(0)

def get_images():
    l =listdir('img/')
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
    filename ='img/'
    no=1
    for lst in img[:]:
        print(lst[0])
        if len(lst) >1:    
            s,ms=divmod(lst[0],1000)
            m,s=divmod(s,60)
            h,m=divmod(m,60)
            op.write(str(no))
            op.write('\n')
            op.write(str(h)+':'+str(m)+':'+str(s)+','+str(ms)+' --> '+str(h)+':'+str(m)+':'+str(s+2)+','+str(ms+200))
            op.write('\n')
            for i in lst[1]:
                try:        
                    text = ocr(filename+str(lst[0])+'.'+str(i)+'0.jpg',1,0) 
                    if text != '':
                        op.write(text)
                        op.write('\n')
                except:
                    try:
                        text = ocr(filename+str(lst[0])+'.'+str(i)+'00.jpg',1,0) 
                        if text != '':
                            op.write(text)
                            op.write('\n')
                    except:
                        print(str(lst[0])+'.'+str(i)+'0.jpg')

            op.write('\n')
            no+=1

op = open("outputs/output.txt","w+")
fetch_output(op)
#print(ocr('img/'+'880040.0.410.100.jpg',1,1))