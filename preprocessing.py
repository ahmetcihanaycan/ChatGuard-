"""Shared deterministic preprocessing; no test-set fitting."""
import re
import unicodedata

LEXICON = ('idiot', 'stupid', 'moron', 'asshole', 'fuck', 'shit', 'bitch', 'bastard')

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    text = re.sub(r'@\w+', ' user ', text)
    text = text.translate(str.maketrans({'ı':'i','ş':'s','ğ':'g','ü':'u','ö':'o','ç':'c'}))
    # Decode leetspeak only within words, preserving ordinary standalone numbers.
    text = re.sub(r'\b\w*[a-z]\w*\b', lambda m: m[0].translate(str.maketrans('013457', 'oieast')), text)
    text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", normalize(text))

def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(ca != cb)))
        row = nxt
    return row[-1]

def rule_predict(text):
    return int(any(w == bad or (len(w) >= 5 and len(bad) >= 5 and edit_distance(w, bad) <= 1)
                   for w in tokens(text) for bad in LEXICON))
