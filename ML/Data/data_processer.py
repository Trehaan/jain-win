import re
from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab(texts, min_freq=5, max_vocab=5000):
    counter = Counter()
    
    for sentence in texts:
        counter.update(sentence.split())
    
    filtered = [word for word, freq in counter.items() if freq >= min_freq]    
    filtered.sort(key=lambda w: counter[w], reverse=True)   
    filtered = filtered[:max_vocab]
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    for word in filtered:
        vocab[word] = len(vocab)
    
    return vocab

def tokenize_text(text,vocab):
    return [vocab.get(word,vocab.get("<UNK>")) for word in text.split()]

def pad_sequence(sequence,max_padding):
    if len(sequence) >= max_padding:
        return sequence[:max_padding]
    
    return sequence + [0]*(max_padding - len(sequence))



