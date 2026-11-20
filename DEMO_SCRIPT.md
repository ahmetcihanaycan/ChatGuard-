# Two-minute narrated live demo

Record the running browser application and your voice. Do not submit a slideshow-only video.
Use the first 120 seconds; rehearse to finish around 1:55. The text below is a narration guide,
not a claim that a recording already exists. Enter messages live while recording.

## 0:00–0:15 — Problem

“Hello, I am Ahmet Cihan Aycan. ChatGuard is my SWE015 real-time chat moderation application.
It labels English messages as clean or offensive, hides flagged messages, and measures CPU latency.”

## 0:15–1:15 — Live application

Show that the selected model is BERT INT8. Type a new clean message such as
“Thanks for helping our team, that was a great match.” Send it and point to the score and latency.

“This message is processed by the quantized model, not a remote language-model API.
The response log records the selected model and the time for each message.”

Enter a difficult noisy message such as “You are a stup1d idiooot.” Read the actual decision.
If it is wrong, acknowledge the failure rather than claim it worked.

“This example contains leetspeak and repeated letters. The shared normalization handles those
patterns, but the classifier can still make mistakes. Flagged messages are hidden here.”

Switch to Compare models, type the same difficult message, and click Compare.

“The same input can be evaluated by all four models. Their predictions may differ.
These are live measurements; the test-set measurements are reported separately.”

## 1:15–1:45 — Architecture and results

Show the architecture slide, then the measured results slide.

“The interface sends text through shared preprocessing into one selected model.
The output becomes a label, score, masked message and latency log. I compared TF-IDF,
a network using my trained Word2Vec embeddings, fine-tuned BERT, and quantized BERT.
All four use the same held-out test set. CPU timing averages two hundred runs after warm-up.”

“INT8 reduced the inference files from 18.49 to 17.29 megabytes. Its macro-F1 was 0.7211.
Its mean CPU time was 3.09 milliseconds, slower than FP32 but within the 50-millisecond budget.”

## 1:45–2:00 — Conclusion

“The comparison shows the measured trade-off between quality, size and latency.
Social-media training data and missing conversational context remain limitations.
I used ChatGPT and Codex for implementation support and disclosed that in my slides. Thank you.”

## Publishing

Upload the narrated recording to YouTube as unlisted or Google Drive with anyone-with-link access.
Check it in a private browser window. Add that exact video URL and the repository URL to the first
slide and to `student.json`. The final first slide must not contain empty or placeholder links.
