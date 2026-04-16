import os
import torch
from torch.utils.data import Dataset
from ML.Data.data_processer import preprocess_text,build_vocab,tokenize_text,pad_sequence
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
     
    def __init__(self,x,y):
        self.x = torch.tensor(x, dtype = torch.long)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class DataStruct:
    def __init__(self):

        self.train_texts,self.test_texts,self.train_labels,self.text_labels = self.extract_data()
        self.vocab = None

    def build_vocab(self,min_freq,max_vocab):
        self.vocab = build_vocab(self.train_texts,min_freq,max_vocab)
        


    def extract_data(self):

        data = []
        labels = []

        for root,_,files in os.walk(os.path.join("ML","Data","op_spam_v1.4")):
            for file in files:
                if (file.endswith(".txt")):
                    path = os.path.join(root,file)
                    with open(path,"r") as fh:
                        text = fh.read()

                    label = 1 if "deceptive" in root else 0

                    data.append(text)
                    labels.append(label)

        return train_test_split(data, labels, test_size=0.2, random_state=42)

    
    def get_vocab(self):
        return self.vocab
    

    def preprocess_train_data(self):

        self.train_texts = [preprocess_text(text) for text in self.train_texts]
        self.train_sequences = [tokenize_text(text,self.vocab) for text in self.train_texts]
        self.train_sequences = [pad_sequence(sequence,150) for sequence in self.train_sequences]


    def preprocess_test_data(self):
        
        self.test_texts = [preprocess_text(text) for text in self.test_texts]
        self.test_sequences = [tokenize_text(text,self.vocab) for text in self.test_texts]
        self.test_sequences = [pad_sequence(sequence,150) for sequence in self.test_sequences]


