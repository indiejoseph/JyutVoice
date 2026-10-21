# Text front end

Order matters, and it is the same in training-data builders, `scripts/synth_compare.py` and inference (`jyutvoice/utils/onnx_infer.py::prepare_text`):

```
raw sentence
  1. normalize_text      numbers, dates, times, money, units, symbols -> spoken form        (jyutvoice/text/text_processor.py, WeTextProcessing fork)
  2. segment             pydips (coarse) on the Chinese parts only; English phrases stay whole (segment_mixed / script_split.split_phrases)
  3. clean_text + g2p    Cantonese: ToJyutping + colloquial corrections (+ optional per-character overrides); English: CMUdict-based g2p; code-mixed text goes through
                         jyutvoice/text/script_split.py                                       (jyutvoice/text/cleaners.py, text/__init__.py::text_to_sequence)
  4. blank interspersing token 0 between all tokens (done by the datamodule, onnx_synth.text_tokens, and synthesise callers)
```

**Normalisation must see raw, unsegmented text.** Spaces break its date/time/number patterns (`2024 年 10 月` -> cardinal, `3 : 30` -> ratio). So: normalise, then segment, then g2p.
Never reuse a segmenter that strips spaces for code-mixed text (it glues English words together).

## Text normalisation (`text_processor.py`)

- Wraps the user's WeTextProcessing fork (`github.com/indiejoseph/WeTextProcessing`, pinned to a commit in `docker/Dockerfile`), which adds Cantonese to the Mandarin and English
  normalisers. Needs `pynini` and `opencc-python-reimplemented`. Without the package the text comes back unchanged with one warning (`JYUTVOICE_TN=required` makes it an error).
- The fork's shipped Cantonese graphs were compiled with `tag_oov` and interjection removal on (they wrap 係/咗/哋 in `<oov>` and delete 啊/喇). `text_processor` rebuilds them
  with `remove_interjections / remove_erhua / tag_oov / full_to_half` all False and caches the graphs in `$JYUTVOICE_TN_CACHE` (default `~/.cache/jyutvoice/tn`) with an options
  fingerprint, so a stale graph is rebuilt. The first English graph build takes about a minute; Cantonese normalisation is 4-18 ms per sentence afterwards.
- `full_to_half=False` keeps `，。！？` full-width because that is what the segmenter saw in the corpus; full-width letters, digits and `＄％` are folded to ASCII in Python.
- Chinese rules mangle some English tokens, so they are hidden behind placeholders (`_protect` / `_restore`): `user-friendly` (was `user減friendly`), `COVID-19`, `80s`.
  Known upstream quirks left alone: a bare year `1984` is read as a cardinal, English `1,200` as "thousand two hundred", `555-1234` as a range.
- A lone lowercase `d` in Cantonese text means 啲 (it is how it is typed in the MCE corpus) and is rewritten.
- Numbers that read well are left to the fork: `170cm` -> 一百七十釐米, `6:30am` -> 上午六點半, `$20` -> 二十蚊, `3個` -> 三個.

## Segmentation

`pydips` (`BertModel().cut(text, mode="coarse")`) cuts Cantonese words. It cuts *inside* English words (`Bad minton`) and doubles spaces, so only non-Latin pieces go to it.
It is only imported by scripts and by `onnx_infer.get_segmenter`, never by the `jyutvoice` training package. Measured: the model ignores word boundaries (flattening
`word_pos` changes nothing), so segmentation matters through **readings only**: g2p splits on spaces and looks up each chunk in ToJyutping, so the chunk is the polyphone context
(cutting at punctuation only changes 0.5% of readings, one character per word changes 3.3% and doubles the error rate). Use the real segmenter when measuring anything about
readings: a stand-in that makes every character its own word biased an earlier study badly.

## Cantonese g2p (`text/cantonese/`)

- `g2p.py`: ToJyutping readings -> initial / final / tone phones with `word2ph`, `word_pos`, `syllable_pos`.
- `colloquial.py`: corrections of the literary readings ToJyutping returns for very common colloquial characters (房, 聲, 年, 到, 咪, 使 ...). Each rule fires only when the current
  reading is the default it was written to replace. Effect with the real segmenter is small (0.16% of teacher syllables, 0.5% of real-speech syllables).
- `readings.py` + `scripts/correct_readings.py`: per-character overrides mined from Jyutping recognisers (a correction needs >= 2 of 3 models to agree, a corpus filter, and not
  undoing a colloquial rule). Only a handful of characters survive (誒 ei6->e6, 國 gwok3->gok3, 到, 呢, 市, 樓). Wired through the dataset's `phone` column; the expected gain is small.
- Many "errors" of the TTS in ASR scoring are really g2p choices that neither the teacher nor the model render (樓 lau2, 劉, 呢 ni1 vs nei1, n/l merger). Score with the
  referee-agreed column (see [evaluation.md](evaluation.md)).

## English and code-mixing

- `text/script_split.py` (pure: g2p functions are injected, so it is testable without pycantonese/nltk) splits a sentence into Chinese and Latin units and assembles one g2p result.
  Rules: a Latin unit is a word or an English phrase; marks and pads get the *matrix* language id (Chinese if any Chinese is present); in English groups a mark is glued to the
  previous word (the English g2p deletes a mark that follows a space); a bare number is read in English only right after an English word (`iPhone 15`), otherwise it is a
  Cantonese number and normalisation has already spelled it out; the English `_` word separator is kept.
- `text/english/g2p.py` is dictionary first (CMUdict + name dictionary + `engdict-hot.rep` overrides), neural guess for unknown words, spelling out only unknown ALL-CAPS tokens
  (so iPhone, YouTube, WhatsApp, OK come from the dictionary). Hot corrections: WhatsApp, Instagram, and exceptions Deliveroo, Lazada, Uniqlo, Zalando.
- **Stress is encoded as its own tone ids** (7 consonant, 8 unstressed, 9 primary, 10 secondary, `text/english/stress.py`). An earlier version wrote stress into the Cantonese tone
  ids 1-3, so an unstressed vowel got the high level tone: that is why `Google` was stressed on "gle". Name rule: dictionary entries are respected; a capitalised word that no
  dictionary knows, longer than three letters, gets primary stress on its first vowel (`initial_stress`). It is **not** a general rule (Japan, Macau, Berlin, Chanel, Toyota,
  Nintendo, IKEA are stressed later).
- The English g2p loads the DeBERTa tokenizer from `./bert/deberta-v3-large` (tracked, 2.4 MB) **relative to the working directory**, and needs NLTK data (`cmudict`,
  `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`). Run from the repository root.
- `cleaners.replace_punctuation` for pure Chinese text keeps only CJK and punctuation. Latin text in a Chinese sentence goes through the code-mixed path; `lang="en"` text is English only.
  The TextMelDataModule's `get_dur_weight` returns all ones for rows with Latin letters (English words have one `word2ph` entry per word, not per character).

## Punctuation

The vocabulary has `! ? … , . ' -`. `cleaners.rep_map` maps `，；：、` -> `,`, `。` -> `.`, `！` -> `!`, `？` -> `?`, quotes and brackets -> `'`, `~—` -> `-`, `…` -> `...`; everything else
that is not CJK or Latin is dropped. Each mark is its own token (tone 0, syllable position 0) and marks are the **pause tokens** of pause-aware alignment and of the duration
sampler's split temperature. What the data teaches: the comma has a clear effect on pauses (+0.2-0.3 s); `？` produces a weak rising end; `！` is essentially untrained (8 in 6,000
teacher sentences, 16 in 20,000 corpus rows) and the teacher's `！` lengthening and higher end pitch are not reproduced. A punctuation-aware ProsodyCFM (`use_mark`) was built and
**failed** to fix this (the mix model stays best). The frozen encoder's output barely depends on the final mark (other tokens move 5-8%).

## BERT features (optional, off)

`text/bert_align.py` maps sub-word features of a Cantonese BERT onto phone positions (word -> [initial, final] -> duplicate; blank layout inferred). `TextEncoder(bert_dim=...)`
adds a zero-initialised projection, so a model without BERT is reproduced exactly. **No measurable gain was found and BERT is not used by any shipped model**; `bert_dim: 0` in all
current configs. `scripts/build_bert_sidefile.py` writes per-character features next to a dataset instead of rewriting it.
