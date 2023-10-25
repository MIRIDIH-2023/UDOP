import pickle
import json
from PIL import Image
file_ = '/home/work/sy/UDOP/data/json_data/49960.pickle'
file2_ = '/home/work/sy/UDOP/data/json_data/102256.pickle'

image = '/home/work/sy/UDOP/data/images/47008.png'

with open(file_, 'rb') as f:
    with open(file2_, 'rb') as f2:

        obj = pickle.load(f)
        obj2 = pickle.load(f2)
        data = json.loads(json.dumps(obj, default=str))
        data2 = json.loads(json.dumps(obj2, default=str))

        print()

image_data = Image.open(image)
print(image_data)