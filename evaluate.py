"""Same held-out test set and CPU inference path for all four models."""
import csv
import json
import os
import platform
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from models import Engine, ART, ROOT
from preprocessing import rule_predict
from train import read_split

def cpu_name():
    if platform.system() == 'Windows':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    return os.environ.get('PROCESSOR_IDENTIFIER',platform.processor())

def size_mb(name):
    if name == Engine.names[0]: paths=[ART/'tfidf.joblib']
    elif name == Engine.names[1]: paths=[ART/'neural.pt',ART/'word2vec.kv']
    else:
        paths=[p for p in (ART/'bert').iterdir() if p.is_file() and p.suffix != '.safetensors']
        paths += [ART/'bert'/'model.safetensors'] if name == Engine.names[2] else [ART/'bert_int8.pt']
    return sum(p.stat().st_size for p in paths)/1_000_000

def main():
    engine=Engine(); texts,y=read_split('test'); out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[]; failures=[]; predictions={}
    for name in engine.names:
        probs=np.array([engine.probability(t,name) for t in texts]); pred=(probs>=.5).astype(int)
        for t in texts[:20]: engine.probability(t,name)
        times=[]
        for t in texts[:200]:
            start=time.perf_counter(); engine.probability(t,name); times.append((time.perf_counter()-start)*1000)
        row={'model':name,'macro_f1':float(f1_score(y,pred,average='macro')),
             'accuracy':float(accuracy_score(y,pred)),'size_mb':size_mb(name),'latency_mean_ms':float(np.mean(times)),
             'latency_p95_ms':float(np.percentile(times,95)),'latency_runs':len(times),
             'confusion_matrix':confusion_matrix(y,pred).tolist()}
        rows.append(row);predictions[name]=pred.tolist();print(row,flush=True)
        if name==engine.names[-1]:
            failures=[{'text':t,'actual':int(label),'predicted':int(p),'offensive_probability':float(prob)}
                      for t,label,p,prob in zip(texts,y,pred,probs) if label!=p][:10]
    rule=np.array([rule_predict(t) for t in texts])
    result={'hardware':cpu_name(),'platform':platform.platform(),
            'torch_threads':2,'device':'CPU','batch_size':1,'warmup_runs':20,'latency_scope':'normalization + features/tokenizer + model + softmax; excludes UI/network',
            'size_definition':'serialized inference files, decimal MB; includes vocabulary/tokenizer; excludes runtime libraries',
            'test_n':len(texts),'models':rows,'rule_macro_f1':float(f1_score(y,rule,average='macro')),
            'failures':failures,'int8_budget_ms':50,'int8_budget_passed':rows[-1]['latency_mean_ms']<50}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps({'labels':y.tolist(),'predictions':predictions}),encoding='utf-8')
    with (out/'results.csv').open('w',newline='',encoding='utf-8') as f:
        fields=[k for k in rows[0] if k!='confusion_matrix'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:r[k] for k in fields} for r in rows)

if __name__=='__main__':main()
