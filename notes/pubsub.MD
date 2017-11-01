# Google Cloud Pub/Sub API

## Basic Concepts
- Two main objects:
	- `topics`: 
		- it can be creatd sepearately without any publisher
		- or created by a publisher
		- it is identified by its topic_path, which is a URL as in `projects/{project-name}/topics/{topic-name}`
		- anyone (via publisher api) can publish to a given topic specified by its unique URL
	- `subscription`:
		- it can be created separately or by a subscriber
		- it MUST always be bound to a topic!
		- it is uniquely identified by subscription_path, as in `projects/{project-name}/subscriptions/{subscription-name}`
		- anyone (via subscriber api) can subscribe and receive the msg
- In a nutshell, the gcp pub/sub api is more like 
	- creating incoming endpoint as `topics`, which are identified by URL
	- creating outgoing endpoints as `subscriptions`, which are identified by URL
	- anyone can subscribe with topics or subscriptions with their URL
	- mutliple subscribers can subscribe with the same `subscription`, but it is recommended. Because as long as the msg on the subscription is acked, it will be removed. So the messages will be distributed to different subscribers randomly.
	- the recommned way of distributing messages to different subscribers, is to use separate subscriptions connected to the same topic.

## Quick Exmample
### 1. Create a topic for input and two subscription for output
```bash
gcloud beta pubsub topics create test_in
```

```bash
gcloud beta pubsub subscriptions create test_out1 --topic test_in
gcloud beta pubsub subscriptions create test_out2 --topic test_in
```


### 2. subscribers code
```python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from google.cloud import pubsub
import time
import os
import sys

PROJECT_NAME = os.environ['GOOGLE_PROJECT_NAME']
SUBSCRIPTION_NAME = os.environ["SUBSCRIPTION_NAME"]


def subscribe(subscriber_name):

	subscriber = pubsub.SubscriberClient()
	subscription_path = subscriber.subscription_path(PROJECT_NAME, SUBSCRIPTION_NAME)

	def callback(msg):
		print('msg received by %s:' % subscriber_name, msg)
		msg.ack()

	# subscribe to subscription_path
	subscriber.subscribe(subscription_path, callback=callback)

	print('%s listening' % subscriber_name)
	while True:
		time.sleep(1000) # doesn't matter as long as it is alive

if __name__ == '__main__':
	subscriber_name = sys.argv[1]
	subscribe(subscriber_name)
```

Create two subscribers
```bash
export GOOGLE_PROJECT_NAME=your-project-name
export SUBSCRIPTION_NAME=test_out1
subscriber.py s1
export SUBSCRIPTION_NAME=test_out2
subscriber.py s2
```

### 3. single publisher
```python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from google.cloud import pubsub
import os

PROJECT_NAME = os.environ['GOOGLE_PROJECT_NAME']
TOPIC_NAME = os.environ["TOPIC_NAME"]

publisher = pubsub.PublisherClient()
topic_path = publisher.topic_path(PROJECT_NAME, TOPIC_NAME)

for n in range(10):
	data = u'message number {}'.format(n)
	data = data.encode('utf-8')
	publisher.publish(topic_path, data)
```

Start publishing
```bash
export GOOGLE_PROJECT_NAME=your-project-name
export TOPIC_NAME=test_in
python publisher.py 
```

## Quick Reference
[GCP PubSub API](https://cloud.google.com/pubsub/docs/quickstarts)