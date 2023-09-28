import base64
import time
from io import BytesIO

import requests
from PIL import Image

from core.common.utils import img_to_str

# Set the base URL for the Flask server
# base_url = 'http://127.0.0.1:5000'
base_url = 'http://3.37.233.51:5001'


# Define the API endpoint to send a GET request to
api_endpoint = '/models/udop'

data = {}
data['text'] =  ['Sentence 1', 'Sentence 2']

# Determine image to send

image = Image.open("data/images/image_0.png")
#data['image'] = img_to_str(image)
data['url'] = "https://file.miricanvas.com/template_thumb/2021/06/20/21/30/kd0p8muepf6v1q35/thumb.jpg"

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
