"""
linux:
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev

mac:
brew install tesseract
"""

from PIL import Image
import os
import pytesseract

def test(img_dir):
    for img in os.listdir(img_dir):
        print(img)
        img = img_dir + img
        im = Image.open(img)
        vcode = pytesseract.image_to_string(im)
path = '/workspace/cooperative-vehicle-infrastructure/vehicle-side/training/image_2/'
test(path)