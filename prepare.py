"""Download public data and an unfine-tuned compact BERT; record provenance."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')

def main():
    from huggingface_hub import snapshot_download
    data = ROOT/'data'
    data.mkdir(exist_ok=True)
    manifest = {}
    base = 'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/offensive/'
    for split in ('train','val','test'):
        for kind in ('text','labels'):
            filename = f'{split}_{kind}.txt'
            path = data/filename
            if not path.exists():
                urllib.request.urlretrieve(base+filename,path)
            manifest[filename] = {'url':base+filename,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    model = 'google/bert_uncased_L-2_H-128_A-2'
    revision = '30b0a37ccaaa32f332884b96992754e246e48c5f'
    snapshot_download(model,revision=revision,local_dir=str(ROOT/'pretrained'),
                      allow_patterns=['config.json','vocab.txt','model.safetensors'])
    manifest['model'] = {'id':model,'revision':revision}
    (data/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    results=ROOT/'results'
    results.mkdir(exist_ok=True)
    (results/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Data and pretrained BERT downloaded.', flush=True)

if __name__ == '__main__':
    main()
