from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import urllib


from googleapiclient import discovery
import os
from io import BytesIO
from PIL import Image
from scipy.misc import imresize
import numpy as np
import base64

PROJECT_NAME = os.environ['GOOGLE_PROJECT_NAME']
MODEL = os.environ['MODEL_NAME']
IMAGE_URL = "https://media.mnn.com/assets/images/2016/08/Lion-Stalking-Kalahari-Desert.jpg.638x0_q80_crop-smart.jpg"

IMG_HEIGHT = 64
IMG_WIDTH = 64

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
  image = Image.open(image_bytes, 'r')
  image = imresize(image, [IMG_HEIGHT, IMG_WIDTH])
  image = (np.array(image) / 255.).astype(np.float32)
  return image

def main():
  image = get_image(IMAGE_URL)
  payload = {'inputs': image.tolist()}
  # img_64 = base64.b64encode(image).decode('utf-8')
  # payload = {'inputs': {'b64': img_64}}
  response = test_model_service(PROJECT_NAME, MODEL, payload)
  print(response)

if __name__ == '__main__':
  main()