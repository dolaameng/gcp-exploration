from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import urllib


from googleapiclient import discovery
import os
import numpy as np


PROJECT_NAME = os.environ['GOOGLE_PROJECT_NAME']
MODEL = os.environ['MODEL_NAME']

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

def main():
	payload = {'inputs': np.random.rand(300, 3).tolist() }
	response = test_model_service(PROJECT_NAME, MODEL, payload)
        print(response)

if __name__ == '__main__':
	main()