"""Integration checks after training: actual INT8 modules and finite live predictions."""
import json
import math
import torch
from models import Engine, ROOT

def main():
    engine=Engine()
    count=sum(isinstance(m,torch.ao.nn.quantized.dynamic.Linear) for m in engine.int8.modules())
    assert count>0, 'INT8 model must really contain quantized layers'
    examples=['Thanks for helping our team!', 'You are a stupid idiot.', 'You are a stup1d idiooot.', '']
    results={}
    for text in examples:
        results[text]={name:engine.probability(text,name) for name in engine.names}
        assert all(math.isfinite(p) and 0<=p<=1 for p in results[text].values())
    report={'quantized_linear_layers':count,'live_predictions':results}
    (ROOT/'results'/'integration_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
