from PIL import Image
import glob
import os

files = glob.glob('../images/train/*.jpg')
print(files)
for f in files:
    try:
        img = Image.open(f)
    except:
        os.remove(f)
