import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
import joblib
from preprocessing import normalize, tokens, rule_predict

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
torch.set_num_threads(2)

class WordNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(100,64), nn.ReLU(), nn.Dropout(.25), nn.Linear(64,2))
    def forward(self, x):
        return self.net(x)

def vectorize(texts, vectors):
    result = []
    for text in texts:
        found = [vectors[w] for w in tokens(text) if w in vectors]
        result.append(np.mean(found,axis=0) if found else np.zeros(100,dtype=np.float32))
    return torch.tensor(np.array(result), dtype=torch.float32)

def quantize(model):
    model = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

class Engine:
    names = ['TF-IDF + Logistic Regression', 'Word2Vec + Neural Network', 'BERT FP32', 'BERT INT8']
    def __init__(self):
        self.tfidf = joblib.load(ART/'tfidf.joblib')
        self.vectors = KeyedVectors.load(str(ART/'word2vec.kv'))
        self.neural = WordNet().eval()
        self.neural.load_state_dict(torch.load(ART/'neural.pt',map_location='cpu',weights_only=True))
        self.tokenizer = AutoTokenizer.from_pretrained(ART/'bert')
        self.bert = BertForSequenceClassification.from_pretrained(ART/'bert',attn_implementation='eager').eval()
        config = BertConfig.from_pretrained(ART/'bert')
        config._attn_implementation = 'eager'
        self.int8 = quantize(BertForSequenceClassification(config))
        # Only load locally generated, trusted model artifacts.
        self.int8.load_state_dict(torch.load(ART/'bert_int8.pt',map_location='cpu',weights_only=True))
    @torch.inference_mode()
    def probability(self, text, name):
        if name == self.names[0]:
            return float(self.tfidf.predict_proba([normalize(text)])[0,1])
        if name == self.names[1]:
            return float(self.neural(vectorize([text], self.vectors)).softmax(-1)[0,1])
        inputs = self.tokenizer(normalize(text),return_tensors='pt',truncation=True,max_length=96)
        model = self.bert if name == self.names[2] else self.int8
        return float(model(**inputs).logits.softmax(-1)[0,1])
