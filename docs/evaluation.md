# Evaluation: what exists, what it can tell, what it cannot

There is no single score. Quality is judged from several partial instruments plus the maintainer's ears. The first rule: **state what you measured.** ASR numbers say words are
*recognised*; they cannot say a voice sounds natural, an accent is right, or stress is on the right syllable.

## Standard loop

1. Render the same sentences with every system you compare, **same speaker, same reference clip, same seeds** (`scripts/synth_compare.py`; it writes `results.jsonl` + `wavs/`).
2. Score the renders (below). Compare against the previous best on the *same* sentences.
3. Build a listening page (`scripts/make_listening_page.py --synth-dir <dir> [--blind]`, where `--blind` hides model names; `scripts/build_listen_page.py --out ... --scores ... --models ...` is the
   variant that shows scores next to the audio) and let the maintainer listen. They decide mixes.

```bash
# render (PyTorch pipeline: full decoder + flow sampler + ProsodyCFM, with the speaker's clip as decoder prompt)
python scripts/synth_compare.py --out tmp/new --dataset-dir <dataset_yue> --config configs/synth_flowdp.yaml --prompt \
    --dp-temperature 0.3 --dp-pause-temperature 1.0 --prosody-ckpt <prosody_cfm_best.pt> --length-scale 1.1 \
    --model "new=<checkpoint.ckpt>:0@cumulative"
# code-mixed set (text already normalised and segmented, no pydips needed):  add  --sentences scripts/synth_sentences_mixed.txt --segmentation given --no-text-normalize
```

`synth_compare.py` options that matter: `--model name=ckpt:bert_dim[@ceil|round|cumulative][!noblank]` (repeatable), `--match-teacher/--prompts-meta` (the teacher's voices),
`--prompt` (clip as decoder prompt), `--prosody-*`, `--dp-*`, `--length-scale`, `--segmentation {pydips,clause,char,given}`, `--seed`. Every model must be fed the same input.

## Instruments

| question | tool | needs | numbers worth knowing |
|---|---|---|---|
| Are English words in code-mixed sentences intelligible? | `scripts/eval_mixed_asr.py --out <synth dir> --base-url <ASR endpoint>` on `scripts/synth_sentences_mixed.txt` (20 code-mixed + 4 pure-Cantonese controls, 2 voices) | a Qwen3-ASR-compatible endpoint, no forced language | per utterance: share of English words (>= 2 letters) found in the transcript, CER of the Chinese characters. **Controls must not get worse.** Before English training: 47% English words, CER 20.3%. After: 97-99%, CER 1.5-2.4%, controls 4.2% both |
| Does the audio say the syllables the model was given? | `scripts/score_synth_readings.py --synth-dir <dir> [--referee-dir <teacher renders>]` | a Jyutping recogniser (separate project) | columns: raw SER; **adjusted** (ignores the filler 誒 and the equivalences nei5/lei5, ni1/nei1: `--ignore-ref`, `--equiv`); **referee-agreed** (only syllables the teacher's own render also said as g2p intended). Use referee-agreed as the TTS-fault metric |
| Is the pacing right? | `scripts/length_vs_teacher.py --teacher <teacher_render.py output dir> --student name=<synth_compare output dir>` (repeatable) | teacher renders of the same sentences (`scripts/teacher_render.py`) | median length ratio vs teacher, std of log ratio, share within +-20%, mean length |
| Are pauses right? | `scripts/pause_stats.py --name <label> --manifest <jsonl of {wav, text}> --audio-root <dir>` (or `--dataset-dir` for corpus rows) | audio | internal pauses >= 0.15 s / >= 0.3 s per clip, median and p90 pause length, vs commas in the text |
| Is the pitch alive? | `scripts/f0_stats.py --synth-dir <dir>` (pyin, semitones) | audio | spread, range, slope, phrase decline, liveliness, reset after pauses. **Voice confounds it** (the same model over five speakers spans spread 2.4-4.9 st): compare the same voice only |
| Does it sound like the reference speaker? | `scripts/spk_sim.py` | audio, CAMPPlus, optionally SpeechBrain ECAPA | CAMPPlus is the conditioning model (circular, flatters); ECAPA is independent. Real different speakers: CAMPPlus 0.13, ECAPA 0.16; the model's same-speaker: 0.78 / 0.58 without a clip prompt, 0.82 / 0.67 with one |
| Does the frame-level model restore detail? | `scripts/diag_mu_smoothing.py`, `scripts/eval_prosody_cfm.py` | cached rows | R2, temporal std ratio, frame-to-frame delta ratio vs real `decoder_h`; good samples have ratios near 1.0 (the encoder mean's are 0.85-0.9) |
| Where do durations go wrong? | `diag_duration.py` (total length), `diag_duration_tokens.py` (phones vs blanks vs punctuation), `diag_token_durations.py` (one sentence), `diag_english_durations.py`, `diag_pause_alignment.py` (alignment bonus sweep) | cached rows / checkpoints | each script's docstring has its usage |
| Is a system better by ear? | listening pages | the maintainer | the only judge of naturalness, accent and stress |

## Reference numbers (2026-10-21, 60 held-out conversational Cantonese sentences, one reference clip, flow sampler T 0.3/1.0, cumulative)

| system | SER | adjusted | referee-agreed | length ratio vs teacher (median) | within 20% |
|---|---|---|---|---|---|
| encoder mean (no ProsodyCFM) | 4.14% | 2.87% | 1.44% | - | - |
| Cantonese-only best (flow sampler ep19 + ProsodyCFM mix1, noise 0.5) | 4.67% | 3.48% | 1.53% | 0.968 | 98% |
| English-capable pipeline (sampler 018 + new ProsodyCFM) | 5.73% | 4.50% | 1.95% | 0.903 | 92% |
| same with `--length-scale 1.1` | 5.52% | 4.30% | 2.18% | 0.992 | 97% |

So the English work costs about +0.8-1.0 point of *adjusted* Cantonese SER (about 20 syllables of 2,462). The old regression duration predictor with `ceil` rounding: ratio 1.216, only 43%
within 20%. Teacher renders themselves misread about 11-12% of syllables on these disfluent ASR sentences (it picks its own readings), so teacher SER is not comparable.
ONNX-path check (24 mixed sentences x 2 voices): English words 97%, CER 2.4%, controls 4.2% (PyTorch path 97% / 1.8% / 4.2%).

## Pitfalls that have already cost time

- **Compare like with like.** An early listening page rendered the student with one speaker and no decoder prompt while the teacher used four prompt voices: they were never in the same voice.
  Since then: `--match-teacher` or one shared reference clip for every system, and check that the served page is the one you just built (a stale server once served an old page).
- **Sampler run-to-run variance** (length ratio 0.865-1.024 across four identical-recipe trainings) means a single trained sampler proves little; decide on several seeds or held-out lengths.
- **Use the real segmenter** (`pydips`) for anything about readings; a stand-in biased a whole study.
- **The recogniser has its own tone mergers and coin-flip particles** (呢 ne1/nei1/ni1, 嚟 lei4/lai4; 3/6, 4/6, 2/5 tone mergers): a unanimous "heard" is not acoustic truth.
- **A float WAV read as int16 is not scaled the same way** (`soundfile.read(dtype="int16")` on a FLOAT file): comparing a float WAV to a PCM16 one this way produces enormous bogus differences.
  Compare float to float, or quantise both identically.
- Pure-Cantonese controls must stay put whenever English or any front-end change is made; the English work changed nothing on them in ASR but moved adjusted SER by about one point on
  conversational text.
- ASR scoring needs the ASR endpoint, which lives on a GPU host the maintainer controls, not in this repo; the render steps run anywhere. Do not hammer a shared production endpoint:
  low concurrency only.

## Tests

`python -m pytest tests --deselect tests/test_number_utils.py -q -p no:warnings` (279 pass, 16 skip without pydips/real WeTextProcessing; with them more run). What is covered:
`test_onnx_export.py` (graphs vs PyTorch, numpy steps vs torch, mutation-checked), `test_api.py` (text splitting, clip handling, cache, endpoints against a stub service, config security), `test_playground.py` (the playground proxy against a stub API: settings whitelist, key handling, dialogue turns, streaming, error mapping; mutation-checked),
`test_text_processor.py`, `test_code_mixed.py`, `test_english_stress.py`, `test_bert_*`, `test_pause_alignment.py`, `test_prosody_train.py`, `test_teacher_rows.py`,
`test_datamodule_mix.py`, `test_dur_weight.py`, `test_correct_readings.py`, `test_build_sd3.py`, `test_model_fixes.py`, `test_flow_prosody.py`. Tests whose names say "real g2p" or need the WeTextProcessing
fork skip on machines without those packages. Several tests were checked by mutation (break the code, see the test fail): keep doing that for new tests.
