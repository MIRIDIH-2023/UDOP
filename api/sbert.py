import base64
import time
from io import BytesIO

import requests
from PIL import Image

from core.common.utils import img_to_str

# Set the base URL for the Flask server
base_url = 'http://127.0.0.1:8080'
# base_url = 'http://3.37.233.51:5000'


# Define the API endpoint to send a GET request to
api_endpoint = '/models/sbert'

data = {}
data['text'] =  ['in the winter dawn - playlist', '# Hashtag']
data['num_recommend'] = 3
data['use_dalle'] = True
# Determine image to send

# Record the start time
start_time = time.time()

response = requests.post(base_url + api_endpoint, json=data)

# Record the end time
end_time = time.time()

# Calculate the time taken for the request
elapsed_time = end_time - start_time

# Check if the request was successful (status code 200)
if response.status_code == 200:
    data = response.json()
    print(data)
    print(f"Request took {elapsed_time:.2f} seconds to complete.")
else:
    print(f"Request failed with status code: {response.status_code}")
