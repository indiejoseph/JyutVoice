# Decisions, negative results and known issues

Read this before proposing an experiment. Most entries cost days; the evidence is in the numbers given and in `git log`. **Do not re-run a rejected idea unless something it depended on has changed.**

## Known problems (parked on purpose)

### English words in code-mixed speech: `Zoom`, `interview` (reported 2026-10-21)

- **What the maintainer hears:** `Zoom` sounds like "drone", and `interview` carries wrong stress.
- **What was measured:** ASR recognises English words in the code-mixed set 97-99% of the time, so they are intelligible; the defect is in sound quality and stress, which ASR cannot rate.
  For `Zoom` the model gives the consonants z and m about 1 frame (20 ms) where the aligned reference has 2-3 frames; English phone timing and stress are thinly trained
  (the MCE corpus is volunteers reading, 27.6 h).
- **Decision (the maintainer's reasoning):** Hong Kong speakers do **not** use a native English accent when they code-mix, so *native English recordings are not assumed to be the
  fix* and may even pull the wrong way. Mark it a known problem; **do not start** native-English data, a 2-frame duration floor, or stress work unless the maintainer reopens it.
- If reopened, first ask what accent they want, then collect examples of how Hong Kong speakers say these words (the MCE clips themselves are the closest evidence).

### Reference clips outside the training voices, and no decoder prompt (found 2026-10-21, after the maintainer said the playground's prosody was worse than the demos)

Same 20 held-out sentences, same seeds, one model (the English-capable one), only the voice or the prompt changed (`tmp/proso_diag` while it lasted; the numbers are in `docs/evaluation.md`-style terms):

- **The speaker x-vector drives pacing and pauses, not only timbre**, because the encoder, the duration sampler and ProsodyCFM are all conditioned on it. Clips of corpus speakers (the teacher's prompt clips) give length ratio
  0.88-1.02 against the teacher and almost no pause of 0.3 s or more. The four Common Voice clips first shipped as playground presets (read speech from volunteers, other recording conditions) gave ratios 0.97 / 1.16 / 1.24 / 1.64,
  1.5-3.2 pauses >= 0.3 s per clip, pauses up to 3 s, and for `yue-low-2` a monotone voice (pitch spread 2.1 semitones against 3.6-4.1). **Any uploaded clip that does not look like a corpus speaker can do this.** The playground
  therefore lists clips from `playground/voices.local/` first (git-ignored; the demo voices).
- Taking the *prosody stages'* speaker vector from a well-behaved corpus speaker and only the flow decoder's from the clip (a possible decoupling, not implemented in the API) restores pace and pauses (ratio 0.90, 0.85 long pauses per
  clip) but not the pitch spread of that voice (2.2): the flat pitch comes from the decoder and that voice. Not listened to by the maintainer yet.
- **The ONNX/API path gives the flow decoder no clip prompt, the demos had one.** With the same seeds the prompt leaves durations unchanged and makes the pitch smoother and wider: frame-to-frame pitch change 0.16 -> 0.13 (English model)
  and 0.16 -> 0.11 (Cantonese-only best) semitones per 10 ms, teacher 0.12; spread 3.62 -> 3.82 and 3.53 -> 3.80, teacher 4.14; pauses of 0.3 s or more per clip 0.35 -> 0.10 and 0.70 -> 0.45. Descriptive numbers: the maintainer's ears decide.
  Adding the prompt to the API needs the speech tokenizer and the flow encoder at inference (a few seconds per new clip, cached with the voice) and makes the decoder run on the prompt's frames too (about +80% decoder time for a 4 s prompt and a 5 s
  sentence). Not done; ask before doing it.
- The English-capable model at `length_scale` 1.0 speaks 10% faster than the teacher (4.62 vs 4.19 syllables/s); the API default is now 1.1 (0.99).
- **Without the decoder prompt the output is too loud, and clipped for some voices (found 2026-10-21).** One sentence, seed 1, API settings, HiFT limiter (`audio_limit` 0.99) switched off to see the true level:
  speech RMS of the no-prompt render against the clip's own level (dBFS): demo-1 -9.8 vs -21.9, demo-2 -15.4 vs -26.8, demo-3 -16.8 vs -20.7, demo-4 -16.5 vs -22.5, `yue-high-1` -12.5 vs -18.1, `yue-high-2` -15.7 vs -17.7,
  `yue-low-1` -20.2 vs -25.6, `yue-low-2` -21.7 vs -19.8. So the level swings 12 dB with the voice, and peaks reach +1.8 to +3.8 dBFS for demo-1, `yue-high-1` and `yue-low-1`, which the limiter flattens (0.17% of the samples of a demo-1
  render sit at the ceiling). With the prompt the level follows the clip (within 1-2 dB, `yue-high-1` +4) and nothing clips. The high band (4-12 kHz) is also 4-13 dB hotter without the prompt, so the no-prompt voice is brighter and
  a gain alone will not make the two alike. Levelling the output (limiter off, a fixed gain per request, not per chunk, or the steps between streamed chunks are audible) removes the clipping and the loudness gap; the prompt is what fixes the tone.
  **Done (2026-10-21):** the API levels each request to -20 dBFS speech RMS (`NORMALIZE_LOUDNESS`, `OUTPUT_LEVEL_DBFS`, request field `normalize_loudness`; see inference-and-api.md). Listening check by the maintainer on demo-1, seeds 1-3:
  the no-prompt render scaled to the prompted level sounds like the prompted render, so for demo-1 loudness was the only audible gap (other voices not compared).
- **A "slow" voice is not a sample-rate bug** (checked): the same clip sent at 16, 22.05, 24, 44.1 and 48 kHz gives the same output length (5.52-5.58 s for one sentence); only a band-limited or heavily compressed
  copy moves it (8 kHz 5.96 s, 32 kbps Opus 6.34 s), because that changes the x-vector. Pace per clip (three sentences, API default, speech syllables per second including pauses; pauses >= 0.3 s in brackets;
  teacher 4.19): corpus clips 4.09-4.39 (0-1), `yue-high-1` 4.47 (3), `yue-high-2` 3.68 (3), `yue-low-1` 3.53 (6), `yue-low-2` 2.11 (8). `speed` 1.1 adds about 10% but removes no pause. The remedy that fits is the
  prosody/timbre decoupling above (not implemented).

### Text normalisation gaps (measured 2026-10-21 on the API image, whole-sentence fork normaliser; not fixed)

Found by running the test inputs of a sibling Cantonese TTS service's normaliser through `normalize_text`. Nothing was changed: the model was trained on text produced by this front end, so each fix needs a check against the training text first.

- Plain digits before a unit: `100蚊` -> `一零零蚊` (`100元` and `100塊` are right); phone numbers `9876 5432` -> two cardinals (`九千八百七十六 五千四百三十二`), and `9876-5432` reads the hyphen as `減`.
- A unit written after an amount is said twice: `$100蚊` -> `一百蚊蚊`; `HK$100元` -> `HK十蚊零元` (garbled); `HK$100` keeps the letters `HK`.
- Fractions: `1/2` -> `二分之一` (Cantonese says `二分一`); names and ratios are read as fractions (`7/11` -> `十一分之七`, `24/7`, `50/50` -> `五十分之五十`); `10/21` becomes a date.
- `H:MM:SS` drops the minutes when they are `00` (`12:00:30` -> `十二點三十秒`).
- Right already: `$100`, `$1,250`, `€20`, `£5`, `￥100`, `$13.5`, `20%`, `3.5%`, `10:30`, `9:00`, `2024年10月21日`, `3.14`, the sentence around a symbol (`係 哋 噉 啊` untouched).
- Symbols or digits only go to the Cantonese graph when the sentence has a Chinese character; a bare `$13.5` or `HKD100` goes to the English normaliser (`thirteen point five dollars`).

### Other open items

- The English-capable pipeline is **not** designated "best": `checkpoints/best/` still holds the Cantonese-only best, and the newer pipeline costs about +1 point adjusted Cantonese SER
  (see [evaluation.md](evaluation.md)). Listening confirmation of the new pipeline is the maintainer's call.
- The MCE corpus has no license; trained models derived from it are for internal use until that is settled.
- The sampler renders about 10% short unless `length_scale` 1.1 is used, and sampler selection is noisy (see training facts).
- `！` and `？` are barely trained (see text-frontend.md).
- All ONNX / API / Docker / docs work is uncommitted on branch `prosody-cfm` until the maintainer says otherwise. Branch `dur` was never pushed.
- `tests/test_number_utils.py` is stale (11 failures also on a clean checkout of the base commit) and is deselected in the standard run.

## Decisions and negative results

| topic | outcome | evidence / reason |
|---|---|---|
| Blank tokens | **Keep.** Removing them hurts. | no-blank model: syllable error 8.7% vs 4.0% with blanks (`yue_encoder_noblank.yaml`) |
| Regression duration predictor | Replaced by the flow sampler. | with `ceil` the speech is 18-25% too slow, ratio 1.216 vs teacher, 43% within +-20%, no long pauses ever |
| Distilling teacher durations into the *regression* predictor | No gain. | tried on teacher rows; still deterministic, still too slow, same missing pauses |
| Flow duration sampler | **Shipped**, with split noise (0.3 phones/blanks, 1.0 pause tokens) and cumulative rounding. | matches the teacher's pacing and pauses (length ratio about 0.97, SER 3.74% in the first study); a single global temperature either kills pauses or garbles phones |
| Pause-aware MAS (`pause_bonus` 10) | **Shipped.** | sends about 94% of long silences to pause tokens at negligible prior cost |
| `infer_cfg_scale` of the sampler | Must be 0. | class default 1.5 stretches lengths by 57% |
| ProsodyCFM (frame-level CFM) | **Shipped** (the encoder mean over-smooths: 65-75% of variance, 12% less movement per frame). | one sample at temperature 0.5 matches real roughness (delta ratio 0.93-0.99) |
| ProsodyCFM guidance | Off (cfg 0). | cfg 0.5 raised the std ratio to 1.13 and lowered R2 |
| ProsodyCFM training mix | corpus + teacher rows ("mix1"), noise 0.5. | preferred by ear; teacher-only copies the teacher's reduced readings (+28-35 syllable errors per 2,462); corpus-only loses the phrase pitch decline and reset; the mix costs about +13 |
| Punctuation-aware ProsodyCFM (`use_mark`) | **Failed** (2026-10-21); the mix model stays best. | it did not fix `！`/`？` behaviour |
| BERT features | Built, wired, **off**. | no measurable gain in any comparison (`bert_dim: 0` everywhere) |
| Segmentation (`pydips`) | Matters through readings only; the model itself ignores word boundaries. | flattening `word_pos` changed nothing; one character per word changed 3.3% of readings and doubled the error rate |
| Per-character reading overrides | Kept, tiny effect. | only a handful of characters pass the agreement filter |
| Colloquial reading rules | Kept, small effect. | 0.16% of teacher syllables, 0.5% of real-speech syllables |
| Particle masking in the sampler loss (`teacher_flowdp_masked_cpu.yaml`) | No global effect. | measured; run-to-run variance of the sampler is larger than the difference |
| 噉 vs 咁 | Not a teacher problem. | investigated earlier; the teacher does not confuse them |
| Decoder prompt (clip fed to the flow decoder as `prompt_feat`/`prompt_h`) | Raises speaker similarity (CAMPPlus 0.78 -> 0.82, ECAPA 0.58 -> 0.67) but **not in the ONNX/API path**. | needs the CosyVoice speech tokenizer and flow encoder at inference |
| English stress rule for unknown names (primary on first vowel) | Kept as a *name-only* rule. | not general: Japan, Macau, Berlin, Chanel, Toyota, Nintendo, IKEA stress later |
| English stress encoded in Cantonese tone ids | **Bug, fixed.** Stress now has its own ids 7-10. | made `Google` stressed on "gle" |
| Pitch deficit vs the teacher | **Not evidenced.** | the apparent gap was a voice confound (the same model across five speakers spans 2.4-4.9 st) |
| Mandarin | Dropped from scope (2026-10). Focus is Cantonese + English. | `mandarin/` code stays but is unused |
| Teacher for English / code-mixing | The Cantonese LLM is **not** used. | the maintainer judged it bad on English by ear; code-mixed data is human speech (MCE) |
| English-only fine-tune of the encoder | Done: English words 47% -> 97-99% recognised; Cantonese SER +about 1 point. | `yue_en_encoder` + `yue_en_flowdp` recipe |
| `genkai_flow_prosody.yaml` | Never use. | no-blank + BERT + sampler contradicts the ablations; never run |

## Gotchas that have each cost a session

1. **Import order.** `import torch` before `onnxruntime`, `datasets` or `pyarrow` or you get a `CXXABI` error.
2. **`python -I` cannot import `jyutvoice`** (isolated mode drops the working directory from the import path; verified: `ModuleNotFoundError`). Use it only for code that reads untrusted downloaded data and does not import this package.
3. **RoPE caches are lazy and baked at ONNX trace length**; warm them to `max_tokens` before export (gotcha 1 in [inference-and-api.md](inference-and-api.md)).
4. **`torch.cumsum` accumulates float32 in double on CPU**; numpy must too.
5. **The flow decoder's noise is fixed per process** (`rand_noise` after `set_all_random_seed(0)`). An experiment that "adds seed variation" to the decoder shows nothing. The seed controls
   durations, ProsodyCFM and HiFT noise only.
6. **A float WAV read as int16** gives bogus mismatches (see evaluation.md).
7. **NLTK data permissions** in the Docker image (`chmod -R a+rX`).
8. **`datasets` 3.x vs 4.x:** `List` vs `Sequence` features; run `scripts/downgrade_hf_features.py` before a 3.x machine loads shards written by 4.x.
9. **Dataset parts are joined on common columns**; one part without `silence` turns pause-aware MAS off for all rows.
10. **Configs contain machine-specific absolute paths** (datasets, checkpoints, ASR endpoints). Edit before running; never commit a path that names a private host.
11. **The local development machine lacks `pydips`**; unit tests stub it. Anything that measures readings needs the real package.
12. **Stale tests:** `tests/test_number_utils.py` fails on a clean base commit.
13. **A stale web server** once served an old listening page: check the page you serve is the one you just built.
14. **`ProsodyCFM.forward` defaults** are cfg 0, temperature 0.5; changing them in a script silently changes the sound.

## Working agreements with the maintainer (carry over)

- Naturalness is judged by the maintainer's ears; ASR/SER only measure intelligibility. Produce comparable listening material, do not declare a winner on ASR alone.
- Do not commit, push, delete checkpoints or rewrite history unless asked. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Do not use shared production services heavily; do not touch other people's jobs; do not wait on queued cluster jobs when a local run answers the question.
- Never print or commit secrets (API keys, tokens, license material, `docker/.env`).
- Third-party data or code beyond the declared dependencies: ask first.
- Run installs, builds and tests yourself; if a command is blocked, stop and ask the maintainer to run it. Do not edit settings to grant yourself permission.
- Refer to people with they/them unless told otherwise.
