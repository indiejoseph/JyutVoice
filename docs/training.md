# Training, data and checkpoints

Nothing here is needed to *run* the model (see [inference-and-api.md](inference-and-api.md)). Read it before you train, build data, or touch a checkpoint.

## The recipe that produced the deployed model (English-capable pipeline, 2026-10-21)

Three stages, in this order, each frozen before the next:

| # | stage | config / script | data | result (not in git) |
|---|---|---|---|---|
| 1 | **Text encoder** (+ a throw-away regression duration head). Warm start from the Cantonese-only encoder. 15 epochs, batch 32 x 2 accumulation, one 3090, 3-4 min per epoch. | `python -m jyutvoice.train --config configs/yue_en_encoder.yaml` | 11.2k random Cantonese corpus rows + the 59 MCE shards (11.2k clips, 27.6 h of code-mixed read speech) | `checkpoints_yue_en/checkpoint_epoch=014.ckpt` (val 1.1542) |
| 2 | **Flow duration sampler**, encoder frozen (`freeze_encoder: true`, only `encoder.` and `spk_embed_affine_layer.` warm-started via `init_ckpt_prefixes`), pause-aware MAS (`pause_bonus: 10`). 20 epochs, about 2 min each. | `python -m jyutvoice.train --config configs/yue_en_flowdp.yaml` | 5,979 teacher rows (Cantonese LLM teacher, no audio) + MCE rows with the `silence` column added | `checkpoints_yue_en_flowdp/checkpoint_epoch=018.ckpt` (val 1.5148) |
| 3 | **ProsodyCFM**, trained on cached frozen-encoder features, never end to end. 60 epochs, about 52 s each, EMA 0.999. | `scripts/build_prosody_cache.py` once per kind of row, then `scripts/train_prosody_cfm.py --cache <teacher> <mce> <corpus> --repeat 1 1 2 --epochs 60 --out-dir <dir> --device cuda` | teacher + MCE + corpus (corpus counted twice) | `prosody_cfm_best.pt` (EMA weights, val 0.1939) |

Then `python scripts/export_onnx.py` (configs/export_onnx.yaml names these three files). `checkpoints_yue_en_flowdp_t2/` is a variant with the teacher rows listed twice; it did not fix
pacing and was not adopted (it can be deleted, with the maintainer's consent). `checkpoints/init/yue_nobert_epoch009.ckpt` is the Cantonese-only encoder and the "before" baseline.

The earlier **Cantonese-only best** (`checkpoints/best/`, see its README) used: encoder `checkpoints_yue_nobert` epoch 9 (no BERT, blanks kept; trained on the 260k-row Cantonese
corpus) + flow sampler `teacher_flowdp.yaml` epoch 19 + ProsodyCFM "mix1" (20,000 corpus rows + 5,381 teacher rows, noise 0.5). `configs/teacher_flowdp.yaml` is its recipe.

Why this order: ProsodyCFM and the sampler both consume the encoder's states, so any encoder retrain invalidates both (and the ProsodyCFM caches); caches do not depend on the
sampler, so the two can train in parallel.

## Data

### Row format (what `TextMelDataModule` reads)

A Hugging Face `datasets` directory (or a glob of shard directories) with columns:

| column | meaning |
|---|---|
| `text` | word-segmented text (spaces between words; English phrases whole) |
| `lang` | `yue`, `zh` or `en` (code-mixed rows are `yue`) |
| `phone` | optional Jyutping override (g2p is used when empty); `phone_override.json` can correct labels by row id |
| `spk_emb` | 192-d CAMPPlus x-vector |
| `decoder_h` | `(T_mel, 80)` flow-encoder states of the row's speech tokens at 50 Hz: the alignment and prior target |
| `audio` | optional (teacher rows have none); only needed for mel conditioning (prompt) experiments |
| `silence` | optional uint8 per frame, 1 where silent (see `utils/silence_probe.py`); enables pause-aware alignment |
| `text_valid`, `id` | filter flag, row id |
| `bert` | optional per-character features (off) |

`dataset_path` may be a list: parts are joined on their **common columns**. That bites: a part without `silence` switches pause-aware alignment off for *all* rows, so MCE needed
`scripts/add_silence_column.py`. Teacher rows set `teacher_prior_weight: 0` (they train durations only).

`datasets` 4.x writes `Sequence` features as `List`, which 3.x cannot read (also inside the Arrow schema metadata). Shards written with 4.x must go through
`scripts/downgrade_hf_features.py '<glob>'` before a 3.x machine can load them (verified: tables equal).

### Where the data comes from (none of it is in git)

- **Corpus** (about 478 h, 447k rows: Cantonese 262k rows / 160 h, English 91k / 159 h, Mandarin 95k / 159 h): built with the `scripts/prepare_dataset*.py` pipeline (word segmentation,
  CAMPPlus, CosyVoice2 speech tokenizer and flow encoder -> `decoder_h`) and lives on an HPC cluster, not on development machines. It has **no digits, no Latin letters in
  Cantonese/Mandarin rows, therefore no code-mixed rows**. Local copies are tiny smoke sets.
- **Speech-dataset3** (conversational Cantonese turns, ASR-transcribed): `scripts/build_sd3_dataset.py` (`plan` / `build` / `verify`, worker in `data/sd3_build.py`).
  Its transcripts come from an ASR system, so they use ASR spellings (噉, 喇, 囉 are far denser than in the teacher's training text).
- **MCE** (`AlienKevin/mixed_cantonese_and_english_speech`, 14,051 clips, 34.9 h, volunteers *reading* code-mixed scripts, 16 kHz): `scripts/asr_transcribe_parquet.py`
  re-transcribes every clip, `scripts/build_mce_dataset.py plan` keeps clips whose CER between the normalised reference and the normalised ASR transcript is <= 0.10 (and 2-30 s,
  valid text), `build` writes shards, `verify` checks them. Result: 11,193 clips / 27.6 h, 10,707 with English (46,677 English words). Training text = the segmented ASR
  transcript. **The corpus card has no license: internal use only.**
- **Teacher rows**: `scripts/teacher_gen.py` runs the user's Cantonese-fine-tuned CosyVoice2 LLM over a pool of sentences (`scripts/build_teacher_pool.py` makes the pool) and
  writes **speech tokens only**; `scripts/teacher_rows.py` turns them into rows (text + `decoder_h` of the teacher's tokens, no audio, `silence` from the probe).
  Pitfalls: the `hit_max` runaway guard (speech tokens > 8 x text tokens) is too tight for English; sentences the teacher misreads (12.5% of syllables on disfluent ASR text)
  add noise to the alignment targets; teacher wavs of the 6,000 sentences were not kept, so rows cannot be filtered by a recogniser afterwards.
  `scripts/teacher_render.py` renders token files to audio for comparisons.

### Teachers and what they are good for

- The **fine-tuned Cantonese CosyVoice2** is the teacher for Cantonese (and was intelligible on code-mixed text in an ASR trial), but the maintainer judged it bad on
  English/Mandarin by ear. **Do not use it for English or code-mixed targets** unless they say so after listening. For English/Mandarin the teacher would be the *original*
  CosyVoice2 weights, and never with a Cantonese prompt (zero-shot with a Cantonese prompt garbles English and Mandarin).
- Code-mixed speech therefore comes from human recordings (MCE), not from a teacher.

## Configs

All are `hyperpyyaml`. **Absolute paths inside (datasets, checkpoints, output dirs) are machine-specific: edit them for your machine.** The header comment of each says what it is for.

| config | use |
|---|---|
| `yue_en_encoder.yaml`, `yue_en_flowdp.yaml` | stages 1 and 2 above (the deployed model) |
| `synth_flowdp.yaml` | inference model definition (full decoder + flow sampler) for `scripts/synth_compare.py` and `scripts/export_onnx.py` |
| `export_onnx.yaml` | defaults of `scripts/export_onnx.py` (checkpoint, ProsodyCFM weights, `max_tokens`, `euler_steps`, `atol`) |
| `onnx_infer.yaml` | only the frozen decoder and HiFT, for `infer_onnx.py` and the API |
| `teacher_flowdp.yaml` | recipe of the Cantonese-only sampler; `teacher_flowdp_masked_cpu.yaml` is its CPU variant with particle masking |
| `yue_encoder_nobert.yaml` | the Cantonese-only encoder fine-tune (the baseline of every later comparison) |
| `base.yaml`, `encoder_only.yaml`, `sd3_encoder.yaml`, `yue_encoder.yaml` | earlier encoder/regression-duration trainings (some with BERT) |
| `yue_encoder_noblank.yaml` | the no-blank ablation (worse; see decisions) |
| `slp.yaml` | legacy speech-length predictor (`n_tone: 7` on purpose: its own tone table) |
| `genkai_flow_prosody.yaml` | **do not use**: no-blank + BERT + sampler, contradicts the ablation results, never run |

Rules that have bitten: `FlowDurationPredictor.infer_cfg_scale` must be **0** (its class default 1.5 stretches lengths by 57%; the configs set 0); `ProsodyCFM.forward` defaults are
cfg 0 and temperature 0.5 (guidance made it worse: cfg 0.5 -> std ratio 1.13, R2 down); checkpoint `monitor` is `loss/val_epoch` (the old `val_loss` never existed);
`init_ckpt` is weights only, `ckpt_path` resumes everything and fails if the model gained parameters; an older 7-row `tone_emb` is padded to 11 on load (`n_tone: 11` everywhere
except the legacy SLP config).

## Training-time facts worth knowing

- **Sampler run-to-run variance is large.** Four runs of the same recipe gave total length ratios vs the teacher of 0.865, 0.977, 1.009, 1.024 and long-pause rates of 0.25-0.67 per
  clip. The English-capable sampler renders about 10% short (median ratio 0.90); `length_scale 1.1` calibrates it (0.99) but also stretches pauses and English. Any shipped sampler
  needs model selection on held-out lengths, not one lucky seed.
- Token-level R2 of the duration model is low (about 0.4; 0.25 on teacher rows) because utterance-level speaking rate is not determined by text; summing neighbours does not help.
  R2 of a *sampler* is negative by construction, ignore it.
- Encoder over-smoothing is the reason for ProsodyCFM: the deterministic encoder explains only 65-75% of the variance of real `decoder_h` and moves 12% less per frame. A single
  ProsodyCFM sample (temperature 0.5-0.7) matches the real roughness (delta ratio 0.93-0.99) and explains *more* variance than the mean.
- Teacher-only prosody training costs pronunciation (+28-35 syllable errors per 2,462) because it copies the teacher's reduced readings (e.g. 即係); corpus-only data removes that cost but
  does not bring the teacher's phrase pitch decline/reset. The shipped mix (corpus + teacher) was preferred by ear and costs about +13 errors.
- Decoder noise is fixed per process (see architecture); do not expect `seed` to change the flow decoder.
- Local training on a 3090 works with the base Python (torch 2.6, lightning); a data copy of the MCE and teacher rows must be downgraded first (see above). Long training should
  run detached with a log; checkpoints are written every epoch (`save_top_k: 3`).

## Checkpoint hygiene

Checkpoint directories `checkpoints_*` and `checkpoints/` are git-ignored; keep the best epoch of each stage only if the maintainer asks you to clean up, and **never delete without
confirmation**. After any encoder retrain, the sampler, the ProsodyCFM caches and ProsodyCFM weights are all obsolete.
