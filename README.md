# Bengali-OCR
## Introduction

This project implements OCR for television news from Bengali and Hindi news channels. I am using OpenCV along with a pre-trained TensorFlow model called EAST(An Efficient and Accurate Scene Test detector) for detecting ROI (Regions of interest) from the news videos.
Then the detected ROIs are extracted and OCR, implemented using tesseract 4.0 is used to extract the text.
 
## Prerequisites

`apt update`
`apt install -y python3-pip build-essential libssl-dev` `libffi-dev python3-dev`
`apt install  -y tesseract-ocr`
`apt install -y libtesseract-dev libsm-dev`
`pip3 install pytesseract opencv-python numpy`

## Working



## Running the code

`python3 textdetection.py <filename>`
Specify additional argument 'O' for running OCR.
The output is saved to the file output/output.txt
