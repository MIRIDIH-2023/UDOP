# Example client code
import base64
from io import BytesIO

import requests
from PIL import Image

# Set the base URL for the Flask server
base_url = 'http://localhost:8000'


def img_to_str(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf8')

    return img_str


# Define the API endpoint to send a GET request to
api_endpoint = '/main'

data = {}
data['text'] =  ['Sentence 1', 'Sentence 2']
image = Image.open("data/images/image_0.png")
data['image'] = img_to_str(image)

response = requests.post(base_url + api_endpoint, json=data)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Request failed with status code: {response.status_code}")
