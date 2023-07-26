from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import math
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


class MyModel():
    def __init__(self, model, pair_class):
        
        self.model = model
        self.pair_class = pair_class
        self.keyword_list = pair_class.keyword_list
        self.data_list = pair_class.data_list
        self.text_list = pair_class.text_list
        
        pairs, labels = self.pair_class.get_both_pairs_labels()
        self.sample_data = list(zip(pairs, labels))
        self.train_data, self.test_data = train_test_split(self.sample_data, test_size=0.05, random_state=42)
        self.train_examples = [InputExample(texts=[data[0][0], data[0][1]], label=float(data[1])) for data in self.train_data]
        self.train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=256)
        self.train_loss = losses.CosineSimilarityLoss(self.model)
        self.num_epochs = 4
        self.warmup_steps = math.ceil(len(self.train_dataloader) * self.num_epochs * 0.1)
        
        
        self.keyword_embedding_set = []
        
    def train(self, output_path):
        self.model.fit(output_path=output_path,
            train_objectives=[(self.train_dataloader, self.train_loss)],
            epochs=self.num_epochs,
            warmup_steps=self.warmup_steps)

    
    def inference(self, sample_num=5000):
        
        negative_pairs_test, positive_pairs_test = self.pair_class.split_pair(self.test_data)
        
        sum = 0
        print("calculating negative pair score...")    
        for a, b in tqdm(random.sample(negative_pairs_test, sample_num)):
            a = self.model.encode(a)
            b = self.model.encode(b)
            score = cosine_similarity([a], [b])[0][0]
            sum += score
            print("Negative:", sum / sample_num)
        
        sum = 0
        print("calculating positive pair score...")
        for a, b in tqdm(random.sample(positive_pairs_test, sample_num)):
            a = self.model.encode(a)
            b = self.model.encode(b)
            score = cosine_similarity([a], [b])[0][0]
            sum += score
            print("\nPositive:", sum / sample_num)


    def save_emedding_vector(self, save_path, file_name):

        file_path = os.path.join(save_path, file_name)
        
        #keyword_set = list(set( self.keyword_list )) 이거 왜한거지 아 중복제거?
        
        print("making embedding vector...")
        for keyword, json in tqdm( zip(self.keyword_list, self.data_list) ):
            self.keyword_embedding_set.append( (json, self.model.encode(keyword)) ) #json과 임베딩벡터 pair로 저장. json에 모든 정보가 있음
            break
        print("done!")
        
        print("saving vectors...")
        with open(file_path, "wb") as _file:
            pickle.dump(self.keyword_embedding_set, _file)
        print("save done!")

    def load_emedding_vector(self, save_path, file_name):
        
        file_path = os.path.join(save_path, file_name)
        
        print("loading embedding vector...")
        with open(file_path, "rb") as _file:
            self.keyword_embedding_set = pickle.load(_file)
        print("loading done!")


    def get_top_keyword(self, sentence, top_k = 3):
            sentence_embedding = self.model.encode(sentence)
            keyword_score_list = [] # (json, score) pair
            
            for json, keyword_embedding in self.keyword_embedding_set:
                score = cosine_similarity([sentence_embedding], [keyword_embedding])[0][0]
                keyword_score_list.append((json, score)) ################여기 나중에 부하걸릴듯 최적화 필요 (index로 score만 sort? )
            keyword_score_list.sort(key=lambda x: x[1], reverse=True)
            
            return_format = []
            for k in range(top_k):
                k_json = keyword_score_list[k][0]
                
                return_format.append({
                    "json" : k_json,
                    "keyword" : k_json['keyword'],
                    "thumbnail_url" : k_json['thumbnail_url'],
                    "sheet_url" : k_json['sheet_url']
                })
                
            return return_format
            #return keyword_score_list[:top_k]