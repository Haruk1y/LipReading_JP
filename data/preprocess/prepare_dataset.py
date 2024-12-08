import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import torch
from typing import List, Dict, Tuple

def process_video(video_path: Path, output_dir: Path) -> List[np.ndarray]:
    """動画をフレームに分割して保存"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 正規化
        normalized = gray.astype(np.float32) / 255.0
        
        frames.append(normalized)
    
    cap.release()
    return frames

def parse_lab_file(lab_path: Path) -> List[Dict[str, int]]:
    """LABファイルを解析して音素情報を抽出"""
    phonemes = []
    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 空行をスキップ
                continue
            start, end, phoneme = line.split()
            phonemes.append({
                'start': int(start),
                'end': int(end),
                'phoneme': phoneme
            })
    return phonemes

def align_phonemes_to_frames(phonemes: List[Dict], num_frames: int, fps: float = 29.95) -> List[str]:
    """音素情報をフレームに合わせてアライメント"""
    frame_phonemes = []
    ns_per_frame = int(1e9 / fps)
    
    for frame_idx in range(num_frames):
        time_ns = frame_idx * ns_per_frame
        
        # デフォルトは無音
        current_phoneme = 'sil'
        
        # 現在のフレーム時刻に対応する音素を探す
        for p in phonemes:
            if p['start'] <= time_ns < p['end']:
                current_phoneme = p['phoneme']
                break
                
        frame_phonemes.append(current_phoneme)
    
    return frame_phonemes

def process_landmarks(landmarks_path: Path) -> np.ndarray:
    """ランドマークCSVを読み込んで処理"""
    df = pd.read_csv(landmarks_path)
    # 必要な座標のみを抽出 (口周りのランドマーク)
    lip_indices = [
        *range(0, 17),      # Jaw line
        *range(48, 68),     # Mouth
    ]
    landmarks = df.iloc[:, lip_indices].values
    return landmarks

def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    split: str,
    max_seq_length: int = 459  # ROHANの最大フレーム数
):
    """データセットの準備"""
    videos_dir = data_dir / split / "videos"
    lab_dir = data_dir / split / "lab"
    landmarks_dir = data_dir / split / "landmarks"
    
    # 出力ディレクトリの作成
    processed_dir = output_dir / split
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 各ビデオを処理
    video_files = sorted(list(videos_dir.glob("*.mp4")))
    
    for video_path in tqdm(video_files, desc=f"Processing {split} data"):
        video_id = video_path.stem.replace("LFROI_", "")
        
        # 関連ファイルのパスを構築
        lab_path = lab_dir / f"{video_id}.lab"
        landmarks_path = landmarks_dir / f"{video_id}_points.csv"
        
        # 各データの処理
        frames = process_video(video_path, processed_dir)
        phonemes = parse_lab_file(lab_path)
        landmarks = process_landmarks(landmarks_path)
        
        # フレームと音素のアライメント
        frame_phonemes = align_phonemes_to_frames(phonemes, len(frames))
        
        # データの保存
        save_path = processed_dir / f"{video_id}.npz"
        np.savez_compressed(
            save_path,
            frames=np.array(frames),
            phonemes=np.array(frame_phonemes),
            landmarks=landmarks
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    
    # 訓練データと検証データの準備
    prepare_dataset(args.data_dir, args.output_dir, "train")
    prepare_dataset(args.data_dir, args.output_dir, "val")
    prepare_dataset(args.data_dir, args.output_dir, "test")