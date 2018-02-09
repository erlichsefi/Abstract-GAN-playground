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
from_path='photo_database/'
onlyfiles = [join(from_path, f) for f in listdir(from_path) if isfile(join(from_path, f))]
findmin=True
minwitdh=50
minhigth=50
if (findmin):
	minwitdh=100000
	minhigth=100000
	for image_path in onlyfiles:
		image_obj = Image.open(image_path)
		width, height = image_obj.size
		minwitdh=min(minwitdh,width)
		minhigth=min(minhigth,height)

for image_path in onlyfiles:
	image_obj = Image.open(image_path).convert('LA')
	width, height = image_obj.size
	wc=width/2
	hc=height/2
	cropped_image = image_obj.crop((wc-(minwitdh/2),hc-(minhigth/2),wc+(minwitdh/2),hc+(minhigth/2)))
	image_path=image_path.replace(image_path.split(".")[-1],"png")
	image_path=image_path.replace(from_path,mypath)
	cropped_image.save(image_path)
