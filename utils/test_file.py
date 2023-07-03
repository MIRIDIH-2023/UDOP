import pickle
import json

file_ = '../data/json/processed_42534.pickle'

with open(file_, 'rb') as f:
    obj = pickle.load(f)
    data = json.loads(json.dumps(obj, default=str))
    print()