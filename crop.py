from PIL import Image
from os import listdir
import sys
from os.path import isfile, join
"""
this python file a fodler path and crop all images
inside to one size, 2 options:
1.by the image with minimal size
2.by a given size
"""
mypath='photo_database_working_folder/'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
findmin=False
minwitdh=10
minhigth=10
if (findmin):
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
