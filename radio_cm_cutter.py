import argparse
import csv
import glob
import json
import math
import os
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fftpack import dct
from scipy.signal import medfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Segment:
    start: float
    end: float
    score: float = 0.0


def print_ui_header(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n{title}\n{line}")


def print_ui_step(message: str) -> None:
    print(f"[STEP] {message}")


def print_ui_ok(message: str) -> None:
    print(f"[ OK ] {message}")


def print_ui_ng(message: str) -> None:
    print(f"[ NG ] {message}")


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + p.stdout
            + "\n\nSTDERR:\n"
            + p.stderr
        )


def ensure_commands(commands: list[str]) -> None:
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError(
            "Required command not found: "
            + ", ".join(missing)
            + "\nPlease install FFmpeg and ensure ffmpeg/ffprobe are on PATH."
        )


def ffprobe_duration(input_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("ffprobe failed:\n" + p.stderr)
    return float(p.stdout.strip())


def decode_to_wav(input_path: Path, wav_path: Path, sample_rate: int) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    run(cmd)


def read_wav_pcm16(wav_path: Path) -> np.ndarray:
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        if n_channels != 1 or sampwidth != 2:
            raise ValueError("Expected mono 16-bit PCM WAV.")
        frames = wf.readframes(n_frames)
    x = np.frombuffer(frames, dtype=np.int16)
    return x


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    xf = x.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(xf * xf) + 1e-12))
    db = 20.0 * math.log10(rms + 1e-12)
    return db


def compute_frame_rms_db(x: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> np.ndarray:
    win = max(1, int(round(window_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    n = x.size
    n_frames = max(0, (n - win) // hop + 1)
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a = i * hop
        b = a + win
        out[i] = rms_dbfs(x[a:b])
    return out


def _make_mel_filterbank(sr: int, n_fft: int, n_mels: int = 20) -> np.ndarray:
    f_min = 0.0
    f_max = sr / 2.0

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if right <= left:
            continue
        for k in range(left, center):
            if 0 <= k < fb.shape[1] and center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < fb.shape[1] and right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def compute_frame_features(x_pcm16: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(round(window_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    n = x_pcm16.size
    n_frames = max(0, (n - win) // hop + 1)

    if n_frames == 0:
        return np.empty((0, 11), dtype=np.float32), np.empty((0,), dtype=np.float32)

    x = x_pcm16.astype(np.float32) / 32768.0
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr).astype(np.float32)
    mel_fb = _make_mel_filterbank(sr, win, n_mels=20)

    feats = np.empty((n_frames, 11), dtype=np.float32)
    frame_times = np.empty((n_frames,), dtype=np.float32)

    for i in range(n_frames):
        a = i * hop
        b = a + win
        frame = x[a:b]
        frame_times[i] = (a + win / 2) / sr

        rms = np.sqrt(np.mean(frame * frame) + 1e-12)
        rms_db = 20.0 * np.log10(rms + 1e-12)

        zcr = np.mean(np.abs(np.diff(np.signbit(frame).astype(np.float32))))

        spec = np.abs(np.fft.rfft(frame * window)) + 1e-12
        spec_sum = float(np.sum(spec))
        centroid = float(np.sum(freqs * spec) / spec_sum)
        flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))

        half = spec.shape[0] // 2
        low_e = float(np.sum(spec[:half])) + 1e-12
        high_e = float(np.sum(spec[half:])) + 1e-12
        high_low_ratio = math.log10(high_e / low_e)

        mel = mel_fb @ spec
        log_mel = np.log(mel + 1e-9)
        mfcc = dct(log_mel, type=2, norm="ortho")[:6]

        feats[i, 0] = rms_db
        feats[i, 1] = zcr
        feats[i, 2] = centroid
        feats[i, 3] = flatness
        feats[i, 4] = high_low_ratio
        feats[i, 5:] = mfcc

    return feats, frame_times


def segments_from_threshold(
    scores: np.ndarray,
    hop_sec: float,
    start_threshold: float,
    end_threshold: float,
    min_cm_sec: float,
    merge_gap_sec: float,
    pad_sec: float,
) -> list[Segment]:
    segments: list[Segment] = []
    in_seg = False
    seg_start_idx = 0

    for i, score in enumerate(scores):
        if not in_seg:
            if score >= start_threshold:
                in_seg = True
                seg_start_idx = i
        else:
            if score < end_threshold:
                seg_end_idx = i
                in_seg = False
                segments.append(_mk_segment(seg_start_idx, seg_end_idx, scores, hop_sec))

    if in_seg:
        segments.append(_mk_segment(seg_start_idx, len(scores) - 1, scores, hop_sec))

    segments = [s for s in segments if (s.end - s.start) >= min_cm_sec]

    if segments:
        merged = [segments[0]]
        for s in segments[1:]:
            prev = merged[-1]
            if s.start - prev.end <= merge_gap_sec:
                prev.end = max(prev.end, s.end)
                prev.score = max(prev.score, s.score)
            else:
                merged.append(s)
        segments = merged

    for s in segments:
        s.start = max(0.0, s.start - pad_sec)
        s.end = max(s.start, s.end + pad_sec)

    return segments


def segments_from_delta(
    delta_db: np.ndarray,
    hop_sec: float,
    start_delta_db: float,
    end_delta_db: float,
    min_cm_sec: float,
    merge_gap_sec: float,
    pad_sec: float,
) -> list[Segment]:
    return segments_from_threshold(
        scores=delta_db,
        hop_sec=hop_sec,
        start_threshold=start_delta_db,
        end_threshold=end_delta_db,
        min_cm_sec=min_cm_sec,
        merge_gap_sec=merge_gap_sec,
        pad_sec=pad_sec,
    )


def _mk_segment(start_idx: int, end_idx: int, values: np.ndarray, hop_sec: float) -> Segment:
    start = start_idx * hop_sec
    end = end_idx * hop_sec
    score = float(np.mean(values[start_idx:end_idx + 1])) if end_idx >= start_idx else 0.0
    return Segment(start=start, end=end, score=score)


def save_segments_csv(segments: list[Segment], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_sec", "end_sec", "score"])
        for s in segments:
            w.writerow([f"{s.start:.3f}", f"{s.end:.3f}", f"{s.score:.3f}"])


def load_segments_csv(csv_path: Path) -> list[Segment]:
    segs: list[Segment] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            segs.append(Segment(float(row["start_sec"]), float(row["end_sec"]), float(row.get("score", 0.0))))
    return segs


def write_report_html(duration: float, segments: list[Segment], out_html: Path, mode: str = "heuristic") -> None:
    def pct(t: float) -> float:
        return 100.0 * t / max(duration, 1e-6)

    bars = []
    for s in segments:
        left = pct(s.start)
        width = max(0.2, pct(s.end) - pct(s.start))
        bars.append(f'<div class="seg" style="left:{left:.4f}%;width:{width:.4f}%;"></div>')

    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{s.start:.2f}</td><td>{s.end:.2f}</td><td>{(s.end-s.start):.2f}</td><td>{s.score:.3f}</td></tr>"
        for i, s in enumerate(segments)
    )

    html = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>CM Detection Report</title>
<style>
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 20px; }}
.timeline {{ position: relative; height: 24px; background: #eee; border-radius: 8px; overflow: hidden; }}
.seg {{ position: absolute; top: 0; bottom: 0; background: #ff6b6b; opacity: 0.85; }}
small {{ color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
th {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>CM Detection Report</h1>
<p><small>Mode: {mode} / Duration: {duration:.2f} sec / Segments: {len(segments)}</small></p>
<div class="timeline">
{''.join(bars)}
</div>
<table>
<thead><tr><th>#</th><th>start(s)</th><th>end(s)</th><th>len(s)</th><th>score</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def complement_segments(duration: float, cut_segments: list[Segment]) -> list[Segment]:
    cut_segments = sorted(cut_segments, key=lambda s: s.start)
    keep: list[Segment] = []
    cur = 0.0
    for s in cut_segments:
        if s.start > cur:
            keep.append(Segment(cur, min(s.start, duration), 0.0))
        cur = max(cur, s.end)
    if cur < duration:
        keep.append(Segment(cur, duration, 0.0))
    keep = [k for k in keep if (k.end - k.start) >= 0.5]
    return keep


def cut_mp3(input_path: Path, keep_segments: list[Segment], out_path: Path, q: int, keep_parts: bool = False) -> None:
    if keep_parts:
        work = Path("parts_") / (input_path.stem + "_parts")
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        _cut_mp3_in_workdir(input_path, keep_segments, out_path, q, work)
        return

    with tempfile.TemporaryDirectory(prefix="radio_parts_") as tmp:
        work = Path(tmp)
        _cut_mp3_in_workdir(input_path, keep_segments, out_path, q, work)


def _cut_mp3_in_workdir(input_path: Path, keep_segments: list[Segment], out_path: Path, q: int, work: Path) -> None:
    part_files = []
    for i, s in enumerate(keep_segments):
        part = work / f"part_{i:03d}.mp3"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-ss", f"{s.start:.3f}",
            "-to", f"{s.end:.3f}",
            "-c:a", "libmp3lame",
            "-q:a", str(q),
            str(part),
        ]
        run(cmd)
        part_files.append(part)

    concat_txt = work / "concat.txt"
    with open(concat_txt, "w", encoding="utf-8", newline="\n") as f:
        for p in part_files:
            f.write(f"file '{p.as_posix()}'\n")

    cmd2 = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(out_path),
    ]
    run(cmd2)


def _detect_segments_heuristic(x: np.ndarray, cfg: dict) -> tuple[list[Segment], str]:
    sr = int(cfg["sample_rate"])
    rms_db = compute_frame_rms_db(x, sr, float(cfg["window_sec"]), float(cfg["hop_sec"]))
    hop_sec = float(cfg["hop_sec"])

    baseline_frames = max(3, int(round(float(cfg["baseline_window_sec"]) / hop_sec)))
    if baseline_frames % 2 == 0:
        baseline_frames += 1
    baseline = medfilt(rms_db, kernel_size=baseline_frames)
    delta = rms_db - baseline

    segments = segments_from_delta(
        delta_db=delta,
        hop_sec=hop_sec,
        start_delta_db=float(cfg["start_delta_db"]),
        end_delta_db=float(cfg["end_delta_db"]),
        min_cm_sec=float(cfg["min_cm_sec"]),
        merge_gap_sec=float(cfg["merge_gap_sec"]),
        pad_sec=float(cfg["pad_sec"]),
    )
    min_score = float(cfg.get("min_score", 0.0))
    if min_score > 0:
        segments = [s for s in segments if s.score >= min_score]

    return segments, "heuristic"


def _load_model(model_path: Path) -> dict:
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    if "model" not in obj:
        raise ValueError("Invalid model file: missing model key")
    return obj


def _detect_segments_ml(x: np.ndarray, cfg: dict, model_path: Path) -> tuple[list[Segment], str]:
    model_obj = _load_model(model_path)
    model_cfg = model_obj.get("config", {})
    model = model_obj["model"]

    sr = int(model_cfg.get("sample_rate", cfg["sample_rate"]))
    window_sec = float(model_cfg.get("window_sec", cfg["window_sec"]))
    hop_sec = float(model_cfg.get("hop_sec", cfg["hop_sec"]))

    feats, _ = compute_frame_features(x, sr, window_sec, hop_sec)
    if feats.shape[0] == 0:
        return [], "ml"

    probs = model.predict_proba(feats)[:, 1]

    start_th = float(cfg.get("ml_start_prob", model_cfg.get("threshold", 0.55)))
    end_th = float(cfg.get("ml_end_prob", start_th * 0.9))

    segments = segments_from_threshold(
        scores=probs,
        hop_sec=hop_sec,
        start_threshold=start_th,
        end_threshold=end_th,
        min_cm_sec=float(cfg["min_cm_sec"]),
        merge_gap_sec=float(cfg["merge_gap_sec"]),
        pad_sec=float(cfg["pad_sec"]),
    )
    return segments, "ml"


def _resolve_model_path(args: argparse.Namespace, cfg: dict) -> Path:
    if getattr(args, "model", None):
        return Path(args.model)
    return Path(cfg.get("model_path", "model/model.pkl"))


def _detect_with_fallback(x: np.ndarray, cfg: dict, model_path: Path, require_ml: bool = False) -> tuple[list[Segment], str]:
    if model_path.exists():
        try:
            return _detect_segments_ml(x, cfg, model_path)
        except Exception as e:
            if require_ml:
                raise
            print_ui_ng(f"ML model failed, fallback to heuristic: {e}")
    elif require_ml:
        raise FileNotFoundError(f"Model not found: {model_path}")

    return _detect_segments_heuristic(x, cfg)


def cmd_detect(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    input_path = Path(args.input)
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    model_path = _resolve_model_path(args, cfg)

    wav_path = cache_dir / (input_path.stem + "_16k_mono.wav")
    print(f"[1/4] Decoding to WAV: {wav_path}")
    decode_to_wav(input_path, wav_path, int(cfg["sample_rate"]))

    print("[2/4] Loading WAV PCM...")
    x = read_wav_pcm16(wav_path)
    sr = int(cfg["sample_rate"])
    duration = x.size / sr

    print("[3/4] Detecting CM segments...")
    segments, mode = _detect_with_fallback(x, cfg, model_path)

    out_csv = Path(args.out_csv or (input_path.stem + "_segments.csv"))
    out_html = Path(args.out_html or (input_path.stem + "_report.html"))

    print("[4/4] Saving outputs...")
    save_segments_csv(segments, out_csv)
    write_report_html(duration, segments, out_html, mode=mode)

    total_cm = sum((s.end - s.start) for s in segments)
    print(f"Saved: {out_csv} / {out_html} (mode={mode})")
    print(f"CM segments: {len(segments)} total_cm_sec={total_cm:.1f} duration_sec={duration:.1f}")


def cmd_detect_ml(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    input_path = Path(args.input)
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    model_path = _resolve_model_path(args, cfg)

    wav_path = cache_dir / (input_path.stem + "_16k_mono.wav")
    decode_to_wav(input_path, wav_path, int(cfg["sample_rate"]))
    x = read_wav_pcm16(wav_path)
    sr = int(cfg["sample_rate"])
    duration = x.size / sr

    segments, mode = _detect_with_fallback(x, cfg, model_path, require_ml=True)

    out_csv = Path(args.out_csv or (input_path.stem + "_segments.csv"))
    out_html = Path(args.out_html or (input_path.stem + "_report.html"))
    save_segments_csv(segments, out_csv)
    write_report_html(duration, segments, out_html, mode=mode)

    print(f"Saved: {out_csv} / {out_html} (mode={mode})")


def _parse_label_value(v: str) -> int:
    val = str(v).strip().lower()
    if val in {"cm", "ad", "1", "true", "yes"}:
        return 1
    if val in {"program", "show", "0", "false", "no"}:
        return 0
    raise ValueError(f"Unknown label value: {v} (use cm/program)")


def _load_label_rows(label_patterns: list[str]) -> list[dict]:
    paths: list[Path] = []
    for pattern in label_patterns:
        for p in glob.glob(pattern):
            paths.append(Path(p))
    uniq = sorted({p.resolve() for p in paths})
    if not uniq:
        raise FileNotFoundError("No label files matched patterns")

    rows: list[dict] = []
    for p in uniq:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            req = {"audio_path", "start_sec", "end_sec", "label"}
            if not req.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Label CSV missing required columns in {p}: {sorted(req)}")
            for row in reader:
                if not row.get("audio_path"):
                    continue
                rows.append(
                    {
                        "audio_path": Path(row["audio_path"]),
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "label": _parse_label_value(row["label"]),
                    }
                )
    return rows


def _frame_labels(frame_times: np.ndarray, intervals: list[tuple[float, float, int]]) -> np.ndarray:
    y = np.full(frame_times.shape[0], -1, dtype=np.int8)
    for start, end, label in intervals:
        mask = (frame_times >= start) & (frame_times <= end)
        y[mask] = label
    return y


def cmd_train(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    label_rows = _load_label_rows(args.labels)

    by_audio: dict[Path, list[tuple[float, float, int]]] = {}
    for row in label_rows:
        audio = row["audio_path"]
        by_audio.setdefault(audio, []).append((row["start_sec"], row["end_sec"], row["label"]))

    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    sr = int(cfg["sample_rate"])
    window_sec = float(cfg["window_sec"])
    hop_sec = float(cfg["hop_sec"])

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    print_ui_header("Training CM model")
    print_ui_step(f"Audio files with labels: {len(by_audio)}")

    for i, (audio_path, intervals) in enumerate(sorted(by_audio.items()), 1):
        if not audio_path.exists():
            raise FileNotFoundError(f"Labeled audio not found: {audio_path}")

        safe = f"train_{audio_path.stem}_{abs(hash(str(audio_path))) % 1000000}"
        wav_path = cache_dir / f"{safe}_16k_mono.wav"
        print(f"[{i}/{len(by_audio)}] {audio_path.name}")
        decode_to_wav(audio_path, wav_path, sr)
        x = read_wav_pcm16(wav_path)
        feats, frame_times = compute_frame_features(x, sr, window_sec, hop_sec)
        y = _frame_labels(frame_times, intervals)
        mask = y >= 0

        if np.any(mask):
            xs.append(feats[mask])
            ys.append(y[mask])

    if not xs:
        raise RuntimeError("No usable labeled frames found")

    x_train = np.vstack(xs)
    y_train = np.concatenate(ys)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Need both cm and program labels for training")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)

    probs = model.predict_proba(x_train)[:, 1]
    thresholds = np.linspace(0.35, 0.75, 17)
    best_th = 0.55
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    for th in thresholds:
        pred = (probs >= th).astype(np.int8)
        p, r, f1, _ = precision_recall_fscore_support(y_train, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)
            best_p = float(p)
            best_r = float(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "model": model,
        "config": {
            "sample_rate": sr,
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "threshold": best_th,
        },
        "metrics": {
            "train_precision": best_p,
            "train_recall": best_r,
            "train_f1": best_f1,
            "samples": int(x_train.shape[0]),
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)

    print_ui_ok(f"Model saved: {out_path}")
    print_ui_ok(f"Train metrics (frame): precision={best_p:.3f} recall={best_r:.3f} f1={best_f1:.3f} th={best_th:.2f}")


def cmd_init_label_template(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [["audio_path", "start_sec", "end_sec", "label", "note"]]
    if args.audio_path:
        rows.extend(
            [
                [args.audio_path, "0.0", "15.0", "program", "例: 番組本編"],
                [args.audio_path, "120.0", "150.0", "cm", "例: CM"],
            ]
        )

    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print_ui_ok(f"Template created: {out}")


def cmd_cut(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    input_path = Path(args.input)
    segs = load_segments_csv(Path(args.segments))
    duration = ffprobe_duration(input_path)

    keep = complement_segments(duration, segs)
    out_path = Path(args.out or (input_path.stem + "_cut.mp3"))

    print(f"Cutting: keep_segments={len(keep)} -> {out_path}")
    cut_mp3(input_path, keep, out_path, int(cfg.get("mp3_quality_q", 2)), keep_parts=bool(args.keep_parts))
    print("Done.")


def cmd_process_folder(args: argparse.Namespace) -> None:
    print_ui_header("Radio CM Cutter - Folder Batch")
    print_ui_step("Checking external tools...")
    ensure_commands(["ffmpeg", "ffprobe"])
    print_ui_ok("ffmpeg / ffprobe detected")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_path = _resolve_model_path(args, cfg)

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    out_dir = Path(args.out_dir) if args.out_dir else (folder / "cut_output")
    out_mp3_dir = out_dir / "cut"
    seg_dir = out_dir / "segments"
    rep_dir = out_dir / "reports"

    out_mp3_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    files = sorted(folder.rglob("*.mp3") if args.recursive else folder.glob("*.mp3"))
    if not files:
        print(f"No mp3 files found in: {folder}")
        return

    ml_enabled = model_path.exists()
    print_ui_step(f"Target folder: {folder}")
    print_ui_step(f"Detected files: {len(files)}")
    print_ui_step(f"Output dir   : {out_dir}")
    print_ui_step(f"Detection mode: {'ML優先' if ml_enabled else '従来ルール(フォールバック)'}")

    ok = 0
    ng = 0
    total_cm = 0.0

    for i, input_path in enumerate(files, 1):
        try:
            stem = input_path.stem
            out_mp3 = out_mp3_dir / f"{stem}_cut.mp3"
            out_csv = seg_dir / f"{stem}_segments.csv"
            out_html = rep_dir / f"{stem}_report.html"

            if out_mp3.exists() and not args.overwrite:
                print(f"[{i:>3}/{len(files):>3}] SKIP  {input_path.name} (exists)")
                continue

            safe = f"{stem}_{abs(hash(str(input_path))) % 1000000}"
            wav_path = cache_dir / f"{safe}_16k_mono.wav"

            print(f"[{i:>3}/{len(files):>3}] RUN   {input_path.name}")
            decode_to_wav(input_path, wav_path, int(cfg["sample_rate"]))
            x = read_wav_pcm16(wav_path)
            sr = int(cfg["sample_rate"])
            duration = x.size / sr

            segments, mode = _detect_with_fallback(x, cfg, model_path)

            save_segments_csv(segments, out_csv)
            write_report_html(duration, segments, out_html, mode=mode)

            cm_sec = sum((s.end - s.start) for s in segments)
            total_cm += cm_sec
            print(f"      -> mode={mode} segments={len(segments)} cm_sec={cm_sec:.1f}")

            keep = complement_segments(ffprobe_duration(input_path), segments)
            cut_mp3(
                input_path,
                keep,
                out_mp3,
                int(cfg.get("mp3_quality_q", 2)),
                keep_parts=bool(args.keep_parts),
            )
            ok += 1
            print(f"      -> output={out_mp3.name}")

        except Exception as e:
            ng += 1
            print_ui_ng(f"{input_path.name} -> {e}")

    print_ui_header("Summary")
    print(f"files       : {len(files)}")
    print(f"success     : {ok}")
    print(f"failed      : {ng}")
    print(f"total_cm_sec: {total_cm:.1f}")
    print(f"output      : {out_dir}")
    if ng == 0:
        print_ui_ok("Batch process completed")
    else:
        print_ui_ng("Batch process completed with errors")


def main() -> int:
    p = argparse.ArgumentParser(prog="radio_cm_cutter")
    p.add_argument("--config", default="config.json", help="path to config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_detect = sub.add_parser("detect", help="detect CM segments and write CSV/HTML (ML auto fallback)")
    p_detect.add_argument("input", help="input MP3 path")
    p_detect.add_argument("--out-csv", default=None)
    p_detect.add_argument("--out-html", default=None)
    p_detect.add_argument("--model", default=None, help="path to ML model (default: config model_path)")
    p_detect.set_defaults(func=cmd_detect)

    p_detect_ml = sub.add_parser("detect-ml", help="detect using ML model only")
    p_detect_ml.add_argument("input", help="input MP3 path")
    p_detect_ml.add_argument("--out-csv", default=None)
    p_detect_ml.add_argument("--out-html", default=None)
    p_detect_ml.add_argument("--model", default=None, help="path to ML model (default: config model_path)")
    p_detect_ml.set_defaults(func=cmd_detect_ml)

    p_train = sub.add_parser("train", help="train ML model from labeled CSVs")
    p_train.add_argument("--labels", nargs="+", required=True, help="label CSV patterns (ex: data/labels/*.csv)")
    p_train.add_argument("--out", default="model/model.pkl", help="output model path")
    p_train.set_defaults(func=cmd_train)

    p_template = sub.add_parser("init-label-template", help="create label template CSV")
    p_template.add_argument("--out", default="data/labels/template.csv", help="output template path")
    p_template.add_argument("--audio-path", default="", help="optional audio path to prefill sample rows")
    p_template.set_defaults(func=cmd_init_label_template)

    p_cut = sub.add_parser("cut", help="cut CM segments using CSV")
    p_cut.add_argument("input", help="input MP3 path")
    p_cut.add_argument("--segments", required=True, help="segments CSV from detect")
    p_cut.add_argument("--out", default=None, help="output MP3 path")
    p_cut.add_argument("--keep-parts", action="store_true", help="keep intermediate part_*.mp3 files")
    p_cut.set_defaults(func=cmd_cut)

    p_pf = sub.add_parser("process-folder", help="batch process all MP3 files in a folder")
    p_pf.add_argument("folder", help="folder path containing mp3 files")
    p_pf.add_argument("--out-dir", default=None, help="output directory (default: <folder>/cut_output)")
    p_pf.add_argument("--recursive", action="store_true", help="search subfolders too")
    p_pf.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    p_pf.add_argument("--keep-parts", action="store_true", help="keep intermediate part_*.mp3 files")
    p_pf.add_argument("--model", default=None, help="path to ML model (default: config model_path)")
    p_pf.set_defaults(func=cmd_process_folder)

    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
