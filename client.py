import base64
import time
from io import BytesIO

import requests
from PIL import Image

from core.common.utils import img_to_str

# Set the base URL for the Flask server
base_url = 'http://localhost:5000'


# Define the API endpoint to send a GET request to
api_endpoint = '/main'

data = {}
data['text'] =  ['Sentence 1', 'Sentence 2']

# Determine image to send

# image = Image.open("data/images/image_0.png")
# data['image'] = img_to_str(image)
data['url'] = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-qOqyC6ffHX8nRqfKFsw77Ffy/user-4lZJyO3Z4Zqxebm86OPrG6cI/img-vGqK9C3bZ05rjGrpdNtmEp0X.png?st=2023-09-07T07%3A42%3A17Z&se=2023-09-07T09%3A42%3A17Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-09-06T23%3A46%3A59Z&ske=2023-09-07T23%3A46%3A59Z&sks=b&skv=2021-08-06&sig=o9uUe2CNFwF6qO3ueLO5EA28qPlXHZ1cVrAV083VjdI%3D"

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
