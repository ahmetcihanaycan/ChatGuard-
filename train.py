"""Fit on training data, select epochs on validation data, never on test data."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from gensim.models import Word2Vec
from transformers import BertTokenizerFast, BertForSequenceClassification
import joblib
from preprocessing import normalize, tokens
from models import ROOT, ART, WordNet, vectorize, quantize

SEED = 42
def read_split(split):
    texts = (ROOT/'data'/f'{split}_text.txt').read_text(encoding='utf-8').splitlines()
    labels = np.loadtxt(ROOT/'data'/f'{split}_labels.txt',dtype=int)
    assert len(texts) == len(labels)
    return texts, labels

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ART.mkdir(exist_ok=True)
    train, y = read_split('train'); val, vy = read_split('val')
    # Remove normalized overlaps and duplicate training examples before all fitting.
    excluded = {normalize(x) for x in val + read_split('test')[0]}
    seen = set(); pairs = []
    for text,label in zip(train,y):
        key = normalize(text)
        if key and key not in excluded and key not in seen:
            pairs.append((text,int(label))); seen.add(key)
    train = [p[0] for p in pairs]; y = np.array([p[1] for p in pairs])
    log = {'seed':SEED,'train_n':len(train),'validation_n':len(val),'test_n':len(read_split('test')[0]),
           'train_class_counts':np.bincount(y).tolist(),'max_length':96,'threshold':.5,
           'latency_budget_ms':50,'epochs':{}}
    print('Training split:',len(train),flush=True)
    baseline = Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)),
                         ('lr',LogisticRegression(max_iter=500,class_weight='balanced',random_state=SEED))])
    baseline.fit([normalize(t) for t in train],y)
    joblib.dump(baseline, ART/'tfidf.joblib')
    print('TF-IDF complete',flush=True)
    w2v = Word2Vec([tokens(t) for t in train],vector_size=100,window=5,min_count=2,
                   sg=1,negative=5,epochs=15,workers=1,seed=SEED)
    w2v.wv.save(str(ART/'word2vec.kv'))
    tx = vectorize(train,w2v.wv); vx = vectorize(val,w2v.wv)
    net = WordNet(); opt = torch.optim.SGD(net.parameters(),lr=.05,momentum=.9)
    weights = torch.tensor(len(y)/(2*np.bincount(y)),dtype=torch.float32)
    lossfn = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(tx,torch.tensor(y)),batch_size=64,shuffle=True)
    best = -1; history=[]
    for epoch in range(25):
        net.train()
        for xb,yb in loader:
            opt.zero_grad(); loss=lossfn(net(xb),yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): score=f1_score(vy,net(vx).argmax(-1).numpy(),average='macro')
        history.append(float(score))
        if score > best:
            best=score; torch.save(net.state_dict(),ART/'neural.pt'); log['neural_best_epoch']=epoch+1
    log['epochs']['neural_validation_macro_f1']=history
    print('Word2Vec neural complete',best,flush=True)
    tokenizer = BertTokenizerFast.from_pretrained(ROOT/'pretrained')
    model = BertForSequenceClassification.from_pretrained(ROOT/'pretrained',num_labels=2,attn_implementation='eager')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('BERT training device:',device,flush=True)
    model.to(device)
    encoded = tokenizer([normalize(t) for t in train],padding='max_length',truncation=True,max_length=96,return_tensors='pt')
    ve = tokenizer([normalize(t) for t in val],padding='max_length',truncation=True,max_length=96,return_tensors='pt')
    loader = DataLoader(TensorDataset(encoded['input_ids'],encoded['attention_mask'],torch.tensor(y)),batch_size=32,shuffle=True)
    opt = torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device)); best=-1; history=[]
    for epoch in range(4):
        model.train(); total=0
        for ids,mask,label in loader:
            opt.zero_grad()
            logits=model(input_ids=ids.to(device),attention_mask=mask.to(device)).logits
            loss=criterion(logits,label.to(device)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()
        model.eval(); pred=[]
        with torch.no_grad():
            for i in range(0,len(val),64):
                pred.extend(model(**{k:v[i:i+64].to(device) for k,v in ve.items()}).logits.argmax(-1).cpu().tolist())
        score=float(f1_score(vy,pred,average='macro')); history.append(score)
        print(f'BERT epoch {epoch+1}: validation macro-F1={score:.4f}',flush=True)
        if score>best:
            best=score; model.save_pretrained(ART/'bert'); tokenizer.save_pretrained(ART/'bert'); log['bert_best_epoch']=epoch+1
    log['epochs']['bert_validation_macro_f1']=history
    model = BertForSequenceClassification.from_pretrained(ART/'bert',attn_implementation='eager')
    quantized = quantize(model)
    torch.save(quantized.state_dict(),ART/'bert_int8.pt')
    log['training_device']=str(device)
    (ART/'training.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    print('Training and INT8 quantization complete.',flush=True)

if __name__ == '__main__': main()
