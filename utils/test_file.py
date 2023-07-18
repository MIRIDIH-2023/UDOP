import pickle
import json

file_ = '../data/json_data/processed_0.pickle'

with open(file_, 'rb') as f:
    obj = pickle.load(f)
    data = json.loads(json.dumps(obj, default=str))
    print()