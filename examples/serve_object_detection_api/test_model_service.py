from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import urllib


from googleapiclient import discovery
import os
from io import BytesIO
from PIL import Image
import numpy as np

PROJECT_NAME = os.environ['GOOGLE_PROJECT_NAME']
MODEL = os.environ['MODEL_NAME']
IMAGE_URL = "https://github.com/tensorflow/models/raw/master/research/object_detection/test_images/image2.jpg"


def test_model_service(project, model, payload, version=None):
	service = discovery.build('ml', 'v1')
	name = 'projects/{}/models/{}'.format(project, model)
	if version is not None:
		name += '/versions/{}'.format(version)

	response = service.projects().predict(
		name=name,
		body={'instances': payload}
	).execute()
	return response

def get_image(image_url):
	image_bytes = BytesIO(urllib.request.urlopen(image_url).read())
	image = np.array(Image.open(image_bytes, 'r'))
	return image

def main():
	image = get_image(IMAGE_URL)
	payload = {'inputs': image.tolist()}
	response = test_model_service(PROJECT_NAME, MODEL, payload)
	print(response)

if __name__ == '__main__':
	main()