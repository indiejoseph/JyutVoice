"""Speaker similarity of synthesised speech, with two speaker-verification models.

    python scripts/spk_sim.py --dataset-dir ~/jyutvoice_data/dataset_yue --n-speakers 5 --max-sid 20 \\
        --campplus ~/sd3_build/pretrained_models/campplus.onnx --ecapa speechbrain/spkrec-ecapa-voxceleb \\
        --system old=~/teacher/synth_spk:base --system flow=~/teacher/synth_spk5_flow:flow \\
        --teacher-manifest ~/teacher/render_pilot/manifest.jsonl --prompts ~/teacher/prompts/meta.json

Targets are the speakers synth_compare.py conditions on (`pick_speakers` with the same dataset and count: row ids and their stored CampPlus embedding)
and the REAL recording each of those embeddings came from. Per system (`--system name=<synth_compare output dir>:<model name>`, utterances with sid < --max-sid):

    CampPlus   cosine between the utterance's CampPlus embedding and the conditioning embedding of its speaker. This is the model the conditioning comes from, so it
               flatters the system; it says how well the frozen decoder follows the embedding it was given.
    ECAPA      cosine between the utterance's ECAPA-VoxCeleb embedding and the ECAPA embedding of the speaker's real clip: an independent judge.
    held-out   the same ECAPA cosine against the mean ECAPA embedding of OTHER corpus clips whose stored CampPlus embedding is close to the speaker's
               (`--heldout` nearest neighbours, at least `--heldout-min-cos`; the corpus has no speaker ids, so this is a proxy for "same speaker"). It matters
               for systems that were prompted with the speaker's clip: the output may copy that clip's recording conditions, which flatters the other columns.
    same / other   mean cosine to the utterance's own speaker / to the other speakers, and top-1 identification: the share of utterances whose most similar
               real clip (ECAPA) or conditioning embedding (CampPlus) is the intended speaker's. Chance is 1 / n_speakers.

Also printed: the real clips against each other (the floor for "different speaker") and, with --teacher-manifest, the teacher's renders against the prompt speaker
they were conditioned on.
"""
import argparse
import collections
import glob
import io
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def to16(y, sr):
    import torchaudio

    y = torch.from_numpy(np.asarray(y, dtype=np.float32))
    if y.ndim > 1:
        y = y.mean(-1)
    return torchaudio.functional.resample(y, sr, 16000).numpy() if sr != 16000 else y.numpy()


def real_clips(dataset_dir, wanted):
    """{row id: (16 kHz audio, stored CampPlus embedding)} for the first row of the spread-out shards `pick_speakers` looks at."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    files = sorted(glob.glob(os.path.join(dataset_dir, "data-*.arrow")))
    out = {}
    for fn in files[:: max(1, len(files) // 40)][:40]:
        b = next(iter(ipc.open_stream(pa.memory_map(fn)))).select(["id", "spk_emb", "audio"])
        rid = b.column("id")[0].as_py()
        if rid in wanted:
            audio = b.column("audio")[0].as_py()
            y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            out[rid] = (to16(y, sr), np.asarray(b.column("spk_emb")[0].as_py(), dtype=np.float32))
    return out


def heldout_clips(dataset_dir, cond, ids, k, min_cos):
    """Per speaker: ([16 kHz audio of up to k other corpus clips], [their CampPlus cosines to the speaker]), neighbours by cosine of the stored CampPlus embeddings."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    files = sorted(glob.glob(os.path.join(dataset_dir, "data-*.arrow")))
    keys, embs, rids = [], [], []
    for fi, fn in enumerate(files):
        for bi, b in enumerate(ipc.open_stream(pa.memory_map(fn))):
            col = b.column("spk_emb")
            embs.append(np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(col), -1))
            keys += [(fi, bi, r) for r in range(len(col))]
            rids += b.column("id").to_pylist()
    E = np.concatenate(embs)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    out = []
    for j, c in enumerate(cond):
        sims = E @ (c / np.linalg.norm(c))
        order = [i for i in np.argsort(-sims) if rids[i] != ids[j]][:k]
        order = [i for i in order if sims[i] >= min_cos]
        clips = []
        for i in order:
            fi, bi, r = keys[i]
            for bj, b in enumerate(ipc.open_stream(pa.memory_map(files[fi]))):
                if bj == bi:
                    y, sr = sf.read(io.BytesIO(b.column("audio")[r].as_py()["bytes"]), dtype="float32")
                    clips.append(to16(y, sr))
                    break
        out.append((clips, [float(sims[i]) for i in order]))
    return out


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--n-speakers", type=int, default=5)
    ap.add_argument("--max-sid", type=int, default=20)
    ap.add_argument("--campplus", required=True)
    ap.add_argument("--ecapa", default="speechbrain/spkrec-ecapa-voxceleb", help="a directory, or the hub id (HF_HUB_OFFLINE=1 reads it from the local cache)")
    ap.add_argument("--system", action="append", default=[], help="name=dir:model")
    ap.add_argument("--heldout", type=int, default=8, help="other clips per speaker for the held-out target (0 = skip)")
    ap.add_argument("--heldout-min-cos", type=float, default=0.55)
    ap.add_argument("--matched", action="append", default=[], help="name=dir:model of a synth_compare.py --match-teacher output (spk = the teacher's prompt index): "
                    "scored against that prompt speaker like the teacher renders (needs --prompts)")
    ap.add_argument("--teacher-manifest")
    ap.add_argument("--prompts")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/spk_sim_ecapa"))
    a = ap.parse_args()

    from speechbrain.inference.speaker import EncoderClassifier

    from jyutvoice.utils import flow_features as ff
    from synth_compare import pick_speakers

    cp = ff.load_spk_embedding_model(a.campplus)
    ec = EncoderClassifier.from_hparams(source=a.ecapa, savedir=a.cache, run_opts={"device": "cpu"})

    def emb_cp(y16):
        return np.asarray(ff.get_spk_embedding(y16, cp), dtype=np.float32)

    def emb_ec(y16):
        with torch.no_grad():
            return ec.encode_batch(torch.from_numpy(y16).float().unsqueeze(0)).squeeze().numpy()

    speakers = pick_speakers(a.dataset_dir, a.n_speakers)
    ids = [s for s, _ in speakers]
    cond = [e.numpy().ravel() for _, e in speakers]  # normalised stored CampPlus embeddings, what synth_compare feeds the model
    real = real_clips(a.dataset_dir, set(ids))
    assert set(real) == set(ids), (sorted(real), ids)
    R_cp = [emb_cp(real[i][0]) for i in ids]
    R_ec = [emb_ec(real[i][0]) for i in ids]
    n = len(ids)
    print(f"speakers {ids}")
    print("sanity: CampPlus of the real clip vs its stored embedding:", [round(cos(R_cp[i], real[ids[i]][1]), 3) for i in range(n)])

    def offdiag(m):
        return float(np.mean([m[i][j] for i in range(n) for j in range(n) if i != j]))

    m_cp = [[cos(R_cp[i], R_cp[j]) for j in range(n)] for i in range(n)]
    m_ec = [[cos(R_ec[i], R_ec[j]) for j in range(n)] for i in range(n)]
    print(f"real clips against each other (different speakers): CampPlus {offdiag(m_cp):.3f}, ECAPA {offdiag(m_ec):.3f}\n")

    H_ec = None
    if a.heldout > 0:
        found = heldout_clips(a.dataset_dir, cond, ids, a.heldout, a.heldout_min_cos)
        H_ec = []
        for i, (clips, sims) in enumerate(found):
            print(f"held-out clips for {ids[i]}: {len(clips)} (CampPlus cosine to the speaker {min(sims):.2f}-{max(sims):.2f})" if sims else f"held-out clips for {ids[i]}: none")
            embs = [emb_ec(c) for c in clips]
            H_ec.append(np.mean([e / np.linalg.norm(e) for e in embs], axis=0) if embs else None)
        if any(h is None for h in H_ec):
            print("a speaker has no held-out clip: held-out columns skipped")
            H_ec = None
        else:
            hm = [[cos(R_ec[i], H_ec[j]) for j in range(n)] for i in range(n)]
            print(f"real clip against the held-out centroids: same speaker {np.mean([hm[i][i] for i in range(n)]):.3f}, other speakers {offdiag(hm):.3f} (the ceiling for the held-out column)")
    header = f"{'system':8s} {'utts':>4s} | {'CampPlus: same':>14s} {'other':>6s} {'top1':>5s} | {'ECAPA: same':>11s} {'other':>6s} {'top1':>5s}" + (
        f" | {'held-out: same':>14s} {'other':>6s} {'top1':>5s}" if H_ec else "")
    print(header)
    per_spk = {}
    for spec in a.system:
        name, rest = spec.split("=", 1)
        d, model = rest.rsplit(":", 1)
        d = os.path.expanduser(d)
        rows = [json.loads(line) for line in open(os.path.join(d, "results.jsonl"), encoding="utf-8")]
        rows = [r for r in rows if r["model"] == model and r["sid"] < a.max_sid and r["spk"] < n]
        sc = collections.defaultdict(list)
        for r in rows:
            y, sr = sf.read(os.path.join(d, r["wav"]), dtype="float32")
            y16 = to16(y, sr)
            e_cp, e_ec = emb_cp(y16), emb_ec(y16)
            s = r["spk"]
            c_cp = [cos(e_cp, cond[j]) for j in range(n)]
            c_ec = [cos(e_ec, R_ec[j]) for j in range(n)]
            sc["cp_same"].append(c_cp[s])
            sc["cp_other"].append(np.mean([c_cp[j] for j in range(n) if j != s]))
            sc["cp_top1"].append(int(np.argmax(c_cp) == s))
            sc["ec_same"].append(c_ec[s])
            sc["ec_other"].append(np.mean([c_ec[j] for j in range(n) if j != s]))
            sc["ec_top1"].append(int(np.argmax(c_ec) == s))
            if H_ec:
                c_h = [cos(e_ec, H_ec[j]) for j in range(n)]
                sc["h_same"].append(c_h[s])
                sc["h_other"].append(np.mean([c_h[j] for j in range(n) if j != s]))
                sc["h_top1"].append(int(np.argmax(c_h) == s))
            per_spk.setdefault(name, collections.defaultdict(list))[s].append((c_cp[s], c_ec[s]))
        m = {k: float(np.mean(v)) for k, v in sc.items()}
        print(f"{name:8s} {len(rows):4d} | {m['cp_same']:14.3f} {m['cp_other']:6.3f} {100 * m['cp_top1']:4.0f}% | {m['ec_same']:11.3f} {m['ec_other']:6.3f} {100 * m['ec_top1']:4.0f}%"
              + (f" | {m['h_same']:14.3f} {m['h_other']:6.3f} {100 * m['h_top1']:4.0f}%" if H_ec else ""))
    print("\nper speaker (CampPlus same / ECAPA same):")
    for name, by in per_spk.items():
        print(f"  {name:8s} " + "  ".join(f"{ids[s]}: {np.mean([x[0] for x in v]):.2f}/{np.mean([x[1] for x in v]):.2f}" for s, v in sorted(by.items())))

    if a.teacher_manifest:
        metas = json.load(open(a.prompts, encoding="utf-8"))["prompts"]
        pdir = os.path.dirname(os.path.abspath(a.prompts))
        P = []
        for m in metas:
            y, sr = sf.read(os.path.join(pdir, os.path.basename(m["wav"])), dtype="float32")
            y16 = to16(y, sr)
            P.append((emb_cp(y16), emb_ec(y16)))
        by_mode = collections.defaultdict(list)
        tdir = os.path.dirname(os.path.abspath(a.teacher_manifest))
        for line in open(a.teacher_manifest, encoding="utf-8"):
            r = json.loads(line)
            y, sr = sf.read(os.path.join(tdir, r["wav"]), dtype="float32")
            y16 = to16(y, sr)
            k = r["prompt"]
            e_cp, e_ec = emb_cp(y16), emb_ec(y16)
            others_ec = np.mean([cos(e_ec, P[j][1]) for j in range(len(P)) if j != k])
            by_mode[r["mode"]].append((cos(e_cp, P[k][0]), cos(e_ec, P[k][1]), others_ec, int(np.argmax([cos(e_ec, p[1]) for p in P]) == k)))
        print("\nteacher renders against the prompt speaker they were conditioned on:")
        for mode, v in by_mode.items():
            v = np.array(v)
            print(f"  {mode:10s} n={len(v):3d}  CampPlus same {v[:, 0].mean():.3f}  ECAPA same {v[:, 1].mean():.3f} (other prompts {v[:, 2].mean():.3f}, top-1 {100 * v[:, 3].mean():.0f}% of {len(P)})")


    if a.matched:
        metas = json.load(open(a.prompts, encoding="utf-8"))["prompts"]
        pdir = os.path.dirname(os.path.abspath(a.prompts))
        P = []
        for m in metas:
            y, sr = sf.read(os.path.join(pdir, os.path.basename(m["wav"])), dtype="float32")
            y16 = to16(y, sr)
            P.append((emb_cp(y16), emb_ec(y16)))
        print(f"\nsame-voice systems against the prompt speaker of each sentence ({len(P)} prompt speakers, chance top-1 {100 / len(P):.0f}%):")
        print(f"{'system':14s} {'utts':>4s} | {'CampPlus same':>13s} {'other':>6s} {'top1':>5s} | {'ECAPA same':>10s} {'other':>6s} {'top1':>5s}")
        entries = [("teacher", os.path.dirname(os.path.abspath(a.teacher_manifest)), None)] if a.teacher_manifest else []
        for spec in a.matched:
            name, rest = spec.split("=", 1)
            d, model = rest.rsplit(":", 1)
            entries.append((name, os.path.expanduser(d), model))
        for name, d, model in entries:
            if model is None:
                items = [(json.loads(l)["wav"], json.loads(l)["prompt"]) for l in open(a.teacher_manifest, encoding="utf-8")]
            else:
                items = [(r["wav"], r["spk"]) for r in (json.loads(l) for l in open(os.path.join(d, "results.jsonl"), encoding="utf-8")) if r["model"] == model]
            sc = collections.defaultdict(list)
            for wav, k in items:
                y, sr = sf.read(os.path.join(d, wav), dtype="float32")
                y16 = to16(y, sr)
                e_cp, e_ec = emb_cp(y16), emb_ec(y16)
                c_cp = [cos(e_cp, p[0]) for p in P]
                c_ec = [cos(e_ec, p[1]) for p in P]
                sc["cp"].append(c_cp[k]); sc["cpo"].append(np.mean([c for j, c in enumerate(c_cp) if j != k])); sc["cpt"].append(int(np.argmax(c_cp) == k))
                sc["ec"].append(c_ec[k]); sc["eco"].append(np.mean([c for j, c in enumerate(c_ec) if j != k])); sc["ect"].append(int(np.argmax(c_ec) == k))
            m = {key: float(np.mean(v)) for key, v in sc.items()}
            print(f"{name:14s} {len(items):4d} | {m['cp']:13.3f} {m['cpo']:6.3f} {100 * m['cpt']:4.0f}% | {m['ec']:10.3f} {m['eco']:6.3f} {100 * m['ect']:4.0f}%")


if __name__ == "__main__":
    main()
