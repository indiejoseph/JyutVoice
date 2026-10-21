"""Silence detector for flow-encoder states: a logistic regression on the 80-d `decoder_h` frame.

S3 speech tokens have no single silence code, so a pause cannot be read off the token sequence, but the flow encoder's output for a silent stretch is easy to tell from speech.
The probe is fitted on real corpus clips whose frames are labelled silent from the AUDIO (a frame is silent when it lies in a gap found by librosa.effects.split), and can
then label the frames of sequences that have no audio, such as the teacher's. `SilenceProbe.fit` picks the decision threshold on held-out frames for a target precision,
because the alignment trusts the flags: a false 'silent' frame inside a word should not pull a punctuation token there.
"""
import numpy as np
import torch


class SilenceProbe:
    def __init__(self, mean, std, w, b, threshold=0.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.w = np.asarray(w, dtype=np.float32)
        self.b = float(b)
        self.threshold = float(threshold)

    def score(self, h):
        """(T, 80) states -> (T,) logits (larger = more silent)."""
        return ((np.asarray(h, dtype=np.float32) - self.mean) / self.std) @ self.w + self.b

    def mask(self, h):
        return self.score(h) > self.threshold

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, w=self.w, b=self.b, threshold=self.threshold)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["std"], z["w"], z["b"], z["threshold"])

    @classmethod
    def fit(cls, frames, silent, holdout=0.3, precision_target=0.85, seed=0):
        """frames (N, 80), silent (N,) bool -> (probe, report dict). The hold-out is a random split of frames (pass whole clips in order and set
        holdout=0 to fit on everything with the threshold at 0)."""
        X = torch.from_numpy(np.asarray(frames, dtype=np.float32))
        Y = np.asarray(silent, dtype=bool)
        idx = np.random.default_rng(seed).permutation(len(X))
        n_te = int(len(X) * holdout)
        te, tr = idx[:n_te], idx[n_te:]
        mean, std = X[tr].mean(0), X[tr].std(0).clamp_min(1e-4)
        Xtr = (X[tr] - mean) / std
        Ytr = torch.from_numpy(Y[tr].astype(np.float32))
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        pos_w = (1 - Ytr.mean()) / Ytr.mean().clamp_min(1e-6)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, Ytr, pos_weight=pos_w) + 1e-3 * (w**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        probe = cls(mean.numpy(), std.numpy(), w.detach().numpy(), b.item(), 0.0)
        report = {"train_frames": int(len(tr)), "silent_share": float(Y[tr].mean())}
        if n_te:
            s = probe.score(X[te].numpy())
            order = np.argsort(-s)
            tp = np.cumsum(Y[te][order])
            precision = tp / np.arange(1, len(order) + 1)
            ok = np.where(precision >= precision_target)[0]
            k = ok.max() if len(ok) else 0
            probe.threshold = float(s[order][k])
            pred = s > probe.threshold
            from sklearn.metrics import roc_auc_score

            report.update(holdout_frames=int(n_te), auc=float(roc_auc_score(Y[te], s)), threshold=probe.threshold, precision=float(Y[te][pred].mean()) if pred.any() else float("nan"),
                          recall=float((pred & Y[te]).sum() / max(1, Y[te].sum())))  # fmt: skip
        return probe, report
