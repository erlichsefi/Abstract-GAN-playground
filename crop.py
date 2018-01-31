from PIL import Image
from os import listdir
import sys
from os.path import isfile, join
mypath='/home/nehorg/Downloads/abstract_copy'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
minwitdh=sys.maxint
minhigth=sys.maxint
for image_path in onlyfiles:
	image_obj = Image.open(image_path)
	width, height = image_obj.size
	minwitdh=min(minwitdh,width)
	minhigth=min(minhigth,height)

for image_path in onlyfiles:
	image_obj = Image.open(image_path)
	width, height = image_obj.size
	wc=width/2
	hc=height/2
	cropped_image = image_obj.crop((wc-(minwitdh/2),hc-(minhigth/2),wc+(minwitdh/2),hc+(minhigth/2)))
	cropped_image.save(image_path)

