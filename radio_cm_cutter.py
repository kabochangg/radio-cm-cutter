import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import medfilt


@dataclass
class Segment:
    start: float
    end: float
    score: float = 0.0  # 平均deltaなど


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
    # 解析用に軽量化（mono/16kHz/PCM）
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
    # WAVのPCM16を生で読む（巨大ファイルでも比較的安定）
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
    # int16 -> float
    xf = x.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(xf * xf) + 1e-12))
    db = 20.0 * math.log10(rms + 1e-12)
    return db


def compute_frame_rms_db(x: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> np.ndarray:
    win = max(1, int(round(window_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    n = x.size
    # フレーム数（ホップ基準）
    n_frames = max(0, (n - win) // hop + 1)
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a = i * hop
        b = a + win
        out[i] = rms_dbfs(x[a:b])
    return out


def segments_from_delta(
    delta_db: np.ndarray,
    hop_sec: float,
    start_delta_db: float,
    end_delta_db: float,
    min_cm_sec: float,
    merge_gap_sec: float,
    pad_sec: float,
) -> list[Segment]:
    segments: list[Segment] = []
    in_seg = False
    seg_start_idx = 0

    for i, d in enumerate(delta_db):
        if not in_seg:
            if d >= start_delta_db:
                in_seg = True
                seg_start_idx = i
        else:
            if d < end_delta_db:
                # end at i
                seg_end_idx = i
                in_seg = False
                segments.append(_mk_segment(seg_start_idx, seg_end_idx, delta_db, hop_sec))

    if in_seg:
        segments.append(_mk_segment(seg_start_idx, len(delta_db) - 1, delta_db, hop_sec))

    # 後処理：短すぎるもの削除
    segments = [s for s in segments if (s.end - s.start) >= min_cm_sec]

    # 隙間が小さければ結合
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

    # pad適用（前後を少し広げる）
    for s in segments:
        s.start = max(0.0, s.start - pad_sec)
        s.end = max(s.start, s.end + pad_sec)

    return segments


def _mk_segment(start_idx: int, end_idx: int, delta_db: np.ndarray, hop_sec: float) -> Segment:
    start = start_idx * hop_sec
    end = (end_idx * hop_sec)
    score = float(np.mean(delta_db[start_idx:end_idx + 1])) if end_idx >= start_idx else 0.0
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


def write_report_html(duration: float, segments: list[Segment], out_html: Path) -> None:
    # 超簡易：タイムラインバー＋表
    def pct(t: float) -> float:
        return 100.0 * t / max(duration, 1e-6)

    bars = []
    for s in segments:
        left = pct(s.start)
        width = max(0.2, pct(s.end) - pct(s.start))
        bars.append(f'<div class="seg" style="left:{left:.4f}%;width:{width:.4f}%;"></div>')

    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{s.start:.2f}</td><td>{s.end:.2f}</td><td>{(s.end-s.start):.2f}</td><td>{s.score:.2f}</td></tr>"
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
<p><small>Duration: {duration:.2f} sec / Segments: {len(segments)}</small></p>
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
    # CMを除去したいので、残す区間（keep）を作る
    cut_segments = sorted(cut_segments, key=lambda s: s.start)
    keep: list[Segment] = []
    cur = 0.0
    for s in cut_segments:
        if s.start > cur:
            keep.append(Segment(cur, min(s.start, duration), 0.0))
        cur = max(cur, s.end)
    if cur < duration:
        keep.append(Segment(cur, duration, 0.0))
    # 極端に短いkeepは捨てる（クリックノイズ防止）
    keep = [k for k in keep if (k.end - k.start) >= 0.5]
    return keep


def cut_mp3(input_path: Path, keep_segments: list[Segment], out_path: Path, q: int, keep_parts: bool = False) -> None:
    # keep_parts=True のときだけ parts_ に残す。通常は一時フォルダで自動削除。
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

def cmd_detect(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    input_path = Path(args.input)
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    wav_path = cache_dir / (input_path.stem + "_16k_mono.wav")
    print(f"[1/4] Decoding to WAV: {wav_path}")
    decode_to_wav(input_path, wav_path, int(cfg["sample_rate"]))

    print("[2/4] Loading WAV PCM...")
    x = read_wav_pcm16(wav_path)
    sr = int(cfg["sample_rate"])
    duration = x.size / sr

    print("[3/4] Computing frame RMS (dBFS)...")
    rms_db = compute_frame_rms_db(
        x, sr,
        float(cfg["window_sec"]),
        float(cfg["hop_sec"]),
    )

    hop_sec = float(cfg["hop_sec"])
    baseline_frames = max(3, int(round(float(cfg["baseline_window_sec"]) / hop_sec)))
    if baseline_frames % 2 == 0:
        baseline_frames += 1  # medfiltは奇数
    baseline = medfilt(rms_db, kernel_size=baseline_frames)

    delta = rms_db - baseline

    print("delta_db max:", float(np.max(delta)), "p99:", float(np.percentile(delta, 99)), "p95:", float(np.percentile(delta, 95)))

    print("[4/4] Detecting segments...")
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
    segments = [s for s in segments if s.score >= min_score]

    out_csv = Path(args.out_csv or (input_path.stem + "_segments.csv"))
    out_html = Path(args.out_html or (input_path.stem + "_report.html"))

    save_segments_csv(segments, out_csv)
    write_report_html(duration, segments, out_html)

    total_cm = sum((s.end - s.start) for s in segments)
    print(f"Saved: {out_csv} / {out_html}")
    print(f"CM segments: {len(segments)}  total_cm_sec={total_cm:.1f}  duration_sec={duration:.1f}")


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
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

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

    min_score = float(cfg.get("min_score", 0.0))

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
                print(f"[{i}/{len(files)}] SKIP (exists): {input_path.name}")
                continue

            # --- detect ---
            safe = f"{stem}_{abs(hash(str(input_path))) % 1000000}"
            wav_path = cache_dir / f"{safe}_16k_mono.wav"

            print(f"[{i}/{len(files)}] Detect: {input_path.name}")
            decode_to_wav(input_path, wav_path, int(cfg["sample_rate"]))
            x = read_wav_pcm16(wav_path)
            sr = int(cfg["sample_rate"])
            duration = x.size / sr

            rms_db = compute_frame_rms_db(
                x, sr,
                float(cfg["window_sec"]),
                float(cfg["hop_sec"]),
            )

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

            # score フィルタ（あなたが入れたのと同じ思想）
            if min_score > 0:
                segments = [s for s in segments if s.score >= min_score]

            save_segments_csv(segments, out_csv)
            write_report_html(duration, segments, out_html)

            cm_sec = sum((s.end - s.start) for s in segments)
            total_cm += cm_sec
            print(f"    segments={len(segments)} cm_sec={cm_sec:.1f}")

            # --- cut ---
            keep = complement_segments(ffprobe_duration(input_path), segments)
            cut_mp3(
                input_path,
                keep,
                out_mp3,
                int(cfg.get("mp3_quality_q", 2)),
                keep_parts=bool(args.keep_parts),
            )
            ok += 1

        except Exception as e:
            ng += 1
            print(f"ERROR: {input_path} -> {e}")

    print("\n=== Summary ===")
    print(f"files={len(files)} ok={ok} ng={ng} total_cm_sec={total_cm:.1f}")
    print(f"output: {out_dir}")

def main() -> int:
    p = argparse.ArgumentParser(prog="radio_cm_cutter")
    p.add_argument("--config", default="config.json", help="path to config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_detect = sub.add_parser("detect", help="detect CM segments and write CSV/HTML")
    p_detect.add_argument("input", help="input MP3 path")
    p_detect.add_argument("--out-csv", default=None)
    p_detect.add_argument("--out-html", default=None)
    p_detect.set_defaults(func=cmd_detect)

    p_cut = sub.add_parser("cut", help="cut CM segments using CSV")
    p_cut.add_argument("input", help="input MP3 path")
    p_cut.add_argument("--segments", required=True, help="segments CSV from detect")
    p_cut.add_argument("--out", default=None, help="output MP3 path")
    p_cut.set_defaults(func=cmd_cut)
    p_cut.add_argument("--keep-parts", action="store_true", help="keep intermediate part_*.mp3 files")

    p_pf = sub.add_parser("process-folder", help="batch process all MP3 files in a folder")
    p_pf.add_argument("folder", help="folder path containing mp3 files")
    p_pf.add_argument("--out-dir", default=None, help="output directory (default: <folder>/cut_output)")
    p_pf.add_argument("--recursive", action="store_true", help="search subfolders too")
    p_pf.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    p_pf.add_argument("--keep-parts", action="store_true", help="keep intermediate part_*.mp3 files")
    p_pf.set_defaults(func=cmd_process_folder)

    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
