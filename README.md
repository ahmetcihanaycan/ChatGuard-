# ChatGuard — Real-Time Chat Moderation

SWE015, Use Case 4. Ahmet Cihan Aycan, student ID 220911804.

An English chat moderation application with four independently evaluated inference paths:
TF-IDF + logistic regression, a feedforward network using Word2Vec trained on the training messages,
fine-tuned compact BERT, and the same BERT with dynamic INT8 linear layers. The application defaults
to the quantized model and hides messages classified as offensive.

## Setup

Use Python 3.12 (64-bit). Run these commands from this directory:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.

```sh
python -m pip install -r requirements.txt
python prepare.py
python train.py
python evaluate.py
```

`prepare.py` requires Internet access. Training can run on CPU; it uses CUDA if supported by
the installed PyTorch build. The small two-layer model keeps CPU training practical.
Training writes model files to `artifacts/`. Evaluation writes measured results to `results/`.
`requirements-lock.txt` records the exact complete environment used for the reported run.
Use `python -m pip install -r requirements-lock.txt` instead of the shorter requirements file
to reproduce those transitive dependency versions on Python 3.12.
Do not commit data, pretrained models, environment folders, or trained weights.

## Single application run command

```sh
python app.py
```

Open http://127.0.0.1:7860. The app runs locally and does not send chat inputs to a model API.
The Live chat tab supports new messages, model switching, automatic masking and a response-time log.
Compare models evaluates the same new message with all four models. Measured results displays the
saved test evaluation, which is separate from the live timing of a single message.

## Data and experimental protocol

- English OLID / OffensEval 2019, distributed in TweetEval's `offensive` dataset.
- The source's train, validation and test splits are preserved. Before training, normalized duplicate
  training messages and training messages overlapping validation/test are removed. Held-out labels
  are not used in fitting or model selection. Test texts are used only for overlap exclusion.
- The tokenizer and BERT weights are public pretrained assets. Word2Vec is trained from scratch on
  the training messages only. TF-IDF vocabulary is fitted on training messages only.
- Neural/BERT checkpoints are selected by validation macro-F1. Every model uses threshold 0.5.
- All four models are tested on exactly the same complete held-out test split.
- Latency: CPU, batch 1, two PyTorch threads, 20 warm-ups then 200 single-message runs.
  The timer includes normalization, feature extraction/tokenization, forward pass and probability
  extraction. It excludes model loading, network and browser rendering.
- Size: serialized inference files in decimal MB, including word vectors or tokenizer as applicable;
  Python/runtime libraries are excluded. INT8 quantizes linear layers; embeddings remain FP32.
- The budget is a **mean inference latency below 50 ms** on the reported CPU, not a guarantee that
  every request meets 50 ms. p95 is reported separately. Macro-F1 gives each class equal weight.

## Techniques and implementation

`preprocessing.py`: Unicode/case normalization, URL/user replacement, Turkish-character folding,
within-word leetspeak conversion and repeated-letter reduction. The extra rule baseline uses a
small explicit offensive-word lexicon and Levenshtein distance. It is a diagnostic baseline, not
a hidden override of learned-model predictions.

`train.py`: TF-IDF unigrams/bigrams with balanced logistic regression; skip-gram Word2Vec
(100 dimensions, window 5, 15 epochs) and a 100→64→2 ReLU network with dropout (SGD with momentum,
25 epochs); BERT fine-tuning (4 epochs, AdamW, learning rate 3e-5, batch 32, max length 96).
Class-weighted cross-entropy addresses imbalance. Seed 42 is set; complete bitwise reproducibility
can still depend on hardware and library versions.

Compact English BERT replaces the suggested multilingual BERT because this application is English-only
and targets CPU latency. This remains a pretrained BERT encoder fine-tuned for binary classification.
Dynamic INT8 quantization is applied after training with PyTorch, without a calibration dataset.

`models.py`: shared inference code used by both evaluation and the UI.
`evaluate.py`: metrics, confusion matrices, test predictions, actual failure examples and timing.
`app.py`: Gradio interface, score display, masking and model selection.

## Limitations

Social-media training data differs from gaming conversations. Sarcasm, quoted profanity and context
can cause false positives/negatives. Messages are classified independently. BERT truncates at 96
WordPiece tokens. Scores are not calibrated probabilities of harm. English-only validation does not
establish Turkish performance. The hand-written lexicon has limited coverage and can overmatch.

## Sources

- TweetEval dataset and fixed splits: https://github.com/cardiffnlp/tweeteval
- Barbieri et al. (2020), TweetEval: https://aclanthology.org/2020.findings-emnlp.148/
- Zampieri et al. (2019), OffensEval: https://aclanthology.org/S19-2010/
- Google compact BERT: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
- Model revision is pinned in `prepare.py`; downloaded dataset checksums are in `data/sources.json`.
- PyTorch dynamic quantization: https://docs.pytorch.org/docs/stable/quantization.html

## AI usage

OpenAI ChatGPT/Codex assisted with project design, implementation, debugging, experiment execution,
documentation and presentation drafting. Metrics must come from the supplied scripts' actual runs.
The student must review and understand the implementation before submitting and presenting it.

## Submission preparation

Publish this source folder to GitHub/GitLab (excluding ignored weight/data folders). Upload the
trained artifact archive to Hugging Face Hub or Google Drive and put its accessible link here.
After downloading that archive, extract it so `artifacts/bert/` is directly under this project.
Alternatively, the setup above rebuilds every model from public source assets.
Record a narrated live demo no longer than two minutes, publish its link, and add both repository
and video links to the first presentation slide. Verify both links in a private browser window.

**Publication status:** repository, downloadable trained-weight link and narrated video still need
to be published. Local files alone do not satisfy those submission requirements.
