import random
import numpy as np
from tqdm import tqdm

class make_pair:
    
    def __init__(self,keyword_list,text_list,data_list):
        self.keyword_list = keyword_list
        self.text_list = text_list
        self.data_list = data_list

    def weighted_shuffle(self,arr):
        arr = arr.copy()
        shuffled = []
        while arr:
            weights = np.arange(len(arr), 0, -1)
            weights = weights / np.sum(weights)
            choice = np.random.choice(len(arr), p=weights)
            shuffled.append(arr.pop(choice))
        return shuffled

    def get_positive_pairs(self,text, keywords, n=5):
        keywords = keywords.split()
        pairs = []
        for _ in range(n):
            shuffled_keywords = self.weighted_shuffle(keywords)
            pairs.append((text, ' '.join(shuffled_keywords[:])))
        return pairs


    def get_negative_pair(self,text, keyword):
        keyword_set = set(keyword.split())
        negative_keyword = random.choice(self.keyword_list)
        while set(negative_keyword.split()).intersection(keyword_set):
            negative_keyword = random.choice(self.keyword_list)
        negative_keyword = negative_keyword.split()
        shuffled_keywords = self.weighted_shuffle(negative_keyword)
        return (text, ' '.join(shuffled_keywords))


    def get_negative_pairs(self,text, keywords, n=5):
        pairs = []
        for _ in range(n):
            pairs.append(self.get_negative_pair(text, keywords))
        return pairs
    

    def get_both_pairs_labels(self):
        positive_pairs = []
        negative_pairs = []
        
        print("make pairs...")
        for i in tqdm(range(len(self.text_list))):
            text = self.text_list[i]
            keyword = self.keyword_list[i]
            positive_pairs.extend(self.get_positive_pairs(text, keyword, 5))
            negative_pairs.extend(self.get_negative_pairs(text, keyword, 5))
        print("make pairs done!")
        
        pairs = positive_pairs + negative_pairs
        labels = [1]*len(positive_pairs) + [0]*len(negative_pairs)
        
        return pairs, labels
    
    def split_pair(self,data):
        negative_pairs = [dt[0] for dt in data if dt[1] == 0]
        positive_pairs = [dt[0] for dt in data if dt[1] == 1]
        
        return negative_pairs, positive_pairs