import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def is_valid_landmarks(landmarks, min_range=30):
    """
    ランドマークデータが有効かどうかをチェック
    
    Args:
        landmarks: numpy array of shape (478, 3)
    Returns:
        bool: データが有効かどうか
    """
    try:
        # X, Y座標の範囲をチェック
        x_range = landmarks[:, 0].max() - landmarks[:, 0].min()
        y_range = landmarks[:, 1].max() - landmarks[:, 1].min()
        
        # 基本的な範囲チェック
        valid_range = x_range > min_range and y_range > min_range
        
        # 分散のチェック
        x_std = landmarks[:, 0].std()
        y_std = landmarks[:, 1].std()
        valid_std = x_std > min_range/10 and y_std > min_range/10
        
        # 外れ値のチェック
        x_iqr = np.percentile(landmarks[:, 0], 75) - np.percentile(landmarks[:, 0], 25)
        y_iqr = np.percentile(landmarks[:, 1], 75) - np.percentile(landmarks[:, 1], 25)
        valid_distribution = x_iqr > min_range/4 and y_iqr > min_range/4
        
        return valid_range and valid_std and valid_distribution
    
    except Exception as e:
        print(f"Error in landmark validation: {e}")
        return False

def extract_lip_roi(frame, landmarks, target_size=(96, 96)):
    """
    Extract lip ROI from a frame using facial landmarks
    
    Args:
        frame: numpy array of shape (H, W, C)
        landmarks: numpy array of shape (478, 3) containing facial landmarks
        target_size: tuple of (height, width) for output size
    
    Returns:
        lip_roi: numpy array of shape (target_size[0], target_size[1], C)
    """
    # デバッグ用に最初のフレームのランドマークの範囲を確認  
    if not hasattr(extract_lip_roi, 'debug_printed'):
        points_y = landmarks[:, 1]  # Y座標
        sorted_indices = np.argsort(points_y)
        lip_region_start = int(len(sorted_indices) * 0.6)  # 下半分付近から探索
        potential_lip_points = sorted_indices[lip_region_start:]
        print("Potential lip landmark indices:", potential_lip_points.tolist())
        extract_lip_roi.debug_printed = True

    # ROHANデータセット用の口周辺のランドマークインデックス
    # 注意: 以下のインデックスは仮の値です。実際のデータセットに合わせて修正が必要です
    LIPS_INDICES = list(range(71, 79)) + list(range(58, 70))  # この値は要確認
    
    # Get lip landmarks
    lip_points = landmarks[LIPS_INDICES]
    
    # Calculate bounding box
    x_min = int(np.min(lip_points[:, 0]))
    x_max = int(np.max(lip_points[:, 0]))
    y_min = int(np.min(lip_points[:, 1]))
    y_max = int(np.max(lip_points[:, 1]))
    
    # Add margin
    margin = int((x_max - x_min) * 0.2)
    x_min = max(0, x_min - margin)
    x_max = min(frame.shape[1], x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(frame.shape[0], y_max + margin)
    
    # Extract ROI
    lip_roi = frame[y_min:y_max, x_min:x_max]
    
    # Resize to target size
    lip_roi = cv2.resize(lip_roi, target_size)
    
    return lip_roi

def process_video(video_path, landmarks_path):
    """
    Process a video file to extract lip ROIs for each frame
    
    Args:
        video_path: path to input video file
        landmarks_path: path to landmarks CSV file
        skip_validation: 検証をスキップするかどうか
    Returns:
        frames: numpy array of shape (T, H, W, C) containing lip ROIs
    """

    # CSVファイルの基本情報を確認
    landmarks_df = pd.read_csv(landmarks_path, header=None)

    # 最後の列を除外
    landmarks_df = landmarks_df.iloc[:, :-1]

    try:
        num_frames = len(landmarks_df)
        # numpyアレイに変換して reshape
        landmarks = landmarks_df.values.reshape(num_frames, 478, 3)

        # すべてのフレームでランドマークをチェック
        for frame_idx in range(num_frames):
            if not is_valid_landmarks(landmarks[frame_idx]):
                raise ValueError(f"Invalid landmark ranges detected in frame {frame_idx}")

        # Read video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Extract lip ROI
                lip_roi = extract_lip_roi(frame, landmarks[frame_idx])
                frames.append(lip_roi)
                frame_idx += 1
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
                print(f"Landmarks shape for this frame: {landmarks[frame_idx].shape}")
                break
        
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames were processed successfully for video: {video_path}")

        return np.array(frames)

    except Exception as e:
        print(f"Error processing landmarks from {landmarks_path}: {e}")
        print(f"Landmark shape: {landmarks_df.shape}")
        raise

class LabelProcessor:
    """音素ラベルの処理を行うクラス"""
    def __init__(self):
        # 基本の音素リスト (論文より)
        self.phonemes = ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 
                        'e', 'f', 'fy', 'g', 'gw', 'gy', 'h', 'hy', 'i', 'j', 
                        'k', 'kw', 'ky', 'm', 'my', 'n', 'ny', 'o', 'p', 'pau',
                        'py', 'r', 'ry', 's', 'sh', 'sil', 't', 'ts', 'ty', 'u',
                        'v', 'w', 'y', 'z']
        
        # 特殊な音素の追加 (silB, silE, sp)
        self.special_phonemes = ['silB', 'silE', 'sp']
        self.all_phonemes = self.phonemes + self.special_phonemes
        
        # 音素とインデックスの対応辞書を作成
        self.phoneme2idx = {phoneme: idx for idx, phoneme in enumerate(self.all_phonemes)}
        self.idx2phoneme = {idx: phoneme for idx, phoneme in enumerate(self.all_phonemes)}
        
    def convert_time_to_frame_idx(self, time_value, fps=29.95):
        """時間値(1.0e-7秒単位)をフレームインデックスに変換"""
        seconds = time_value * 1.0e-7
        return int(seconds * fps)
    
    def process_lab_file(self, lab_path, max_frames=459):
        """
        .labファイルを処理してフレームごとの音素ラベルを生成
        
        Args:
            lab_path: .labファイルのパス
            max_frames: 最大フレーム数 (デフォルト: 459)
            
        Returns:
            frame_labels: 各フレームに対応する音素インデックスの配列
            phoneme_sequence: 音素シーケンス（重複なし）のインデックス配列
        """
        # フレームごとのラベルを初期化
        frame_labels = np.zeros(max_frames, dtype=np.int64)
        phoneme_sequence = []
        
        with open(lab_path, 'r') as f:
            lines = f.readlines()
            
        prev_phoneme = None
        for line in lines:
            # 時間とラベルを分割
            start_time, end_time, phoneme = line.strip().split()
            start_time, end_time = int(start_time), int(end_time)
            
            # 最初のsilをsilBに、最後のsilをsilEに変換
            if phoneme == 'sil':
                if prev_phoneme is None:
                    phoneme = 'silB'
                elif end_time == int(lines[-1].split()[1]):
                    phoneme = 'silE'
            
            # 時間をフレームインデックスに変換
            start_frame = self.convert_time_to_frame_idx(start_time)
            end_frame = self.convert_time_to_frame_idx(end_time)
            end_frame = min(end_frame, max_frames)
            
            # 音素インデックスを取得
            phoneme_idx = self.phoneme2idx[phoneme]
            
            # フレームラベルを設定
            frame_labels[start_frame:end_frame] = phoneme_idx
            
            # 音素シーケンスに追加（重複を除く）
            if prev_phoneme != phoneme:
                phoneme_sequence.append(phoneme_idx)
            
            prev_phoneme = phoneme
        
        return frame_labels, np.array(phoneme_sequence)

def process_labels(lab_path, processor):
    """
    Process label file to get frame labels and phoneme sequence
    
    Args:
        lab_path: path to .lab file
        processor: LabelProcessor instance
        
    Returns:
        frame_labels: numpy array of frame-level labels
        phoneme_sequence: numpy array of phoneme sequence
    """
    try:
        with open(lab_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]  # 空行を除去
        
        if not lines:
            raise ValueError(f"Empty label file: {lab_path}")
            
        frame_labels = []
        phoneme_sequence = []
        prev_phoneme = None
        
        for line in lines:
            try:
                # Split time and label
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid line format in {lab_path}: {line}")
                    
                start_time, end_time, phoneme = parts
                start_time, end_time = int(start_time), int(end_time)
                
                # Convert sil to silB/silE
                if phoneme == 'sil':
                    if prev_phoneme is None:
                        phoneme = 'silB'
                    elif lines[-1].split()[1] == str(end_time):  # 最後の行のend_timeと比較
                        phoneme = 'silE'
                
                # Convert time to frame index (assuming 29.97 fps)
                start_frame = int(start_time * 29.97 * 1e-7)
                end_frame = int(end_time * 29.97 * 1e-7)
                
                try:
                    # Get phoneme index
                    phoneme_idx = processor.phoneme2idx[phoneme]
                except KeyError:
                    print(f"Warning: Unknown phoneme '{phoneme}' in {lab_path}")
                    continue
                
                # Assign labels to frames
                frame_labels.extend([phoneme_idx] * (end_frame - start_frame))
                
                # Add to phoneme sequence
                if prev_phoneme != phoneme:
                    phoneme_sequence.append(phoneme_idx)
                
                prev_phoneme = phoneme
                
            except Exception as e:
                print(f"Error processing line in {lab_path}: {line}")
                print(f"Error details: {str(e)}")
                continue
        
        if not frame_labels or not phoneme_sequence:
            raise ValueError(f"No valid labels processed from {lab_path}")
        
        return np.array(frame_labels), np.array(phoneme_sequence)
        
    except Exception as e:
        print(f"Error processing label file {lab_path}: {str(e)}")
        raise

# ROHANDatasetクラスの__getitem__メソッドを更新
class ROHANDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.label_processor = LabelProcessor()

        # Read split list
        split_file = self.data_root / 'processed' / f'{split}_list.txt'
        with open(split_file) as f:
            self.video_files = [line.strip() for line in f if line.strip()]

        # 異常データを記録するリスト
        self.invalid_data = []

        # すべてのデータに対して同じ基準で検証を行う
        print(f"Validating {split} dataset...")
        self.validate_data()

    def validate_data(self):
        """
        データセット内の異常データを検出し、除外する
        """
        valid_files = []
        print(f"Validating {self.split} dataset...")
        total_files = len(self.video_files)
        
        for idx, video_name in enumerate(tqdm(self.video_files)):
            landmarks_path = self.data_root / 'raw' / 'landmarks' / f'{video_name}_points.csv'
            
        for video_name in tqdm(self.video_files):
            landmarks_path = self.data_root / 'raw' / 'landmarks' / f'{video_name}_points.csv'
            label_path = self.data_root / 'raw' / 'lab' / f'{video_name}.lab'
            
            try:
                # ラベルファイルの検証
                with open(label_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if not lines:
                    raise ValueError(f"Empty label file: {label_path}")
                
                # ランドマークデータの検証
                landmarks_df = pd.read_csv(landmarks_path, header=None)
                landmarks_df = landmarks_df.iloc[:, :-1]
                num_frames = len(landmarks_df)
                landmarks = landmarks_df.values.reshape(num_frames, 478, 3)
                
                if is_valid_landmarks(landmarks[0]):
                    valid_files.append(video_name)
                else:
                    self.invalid_data.append((video_name, "Invalid landmark ranges"))
                    print(f"Warning: Removing {video_name} due to invalid landmark ranges")
                    
            except Exception as e:
                self.invalid_data.append((video_name, str(e)))
                print(f"Warning: Removing {video_name} due to error: {e}")
                continue
        
        # 有効なファイルのみを保持
        self.video_files = valid_files
        print(f"Validation complete for {self.split}:")
        print(f"Original files: {total_files}")
        print(f"Valid files: {len(valid_files)}")
        print(f"Invalid files: {len(self.invalid_data)}")
        
        # 異常データの情報を保存（オプション）
        if self.invalid_data:
            invalid_data_file = self.data_root / 'processed' / f'invalid_data_{self.split}.txt'
            with open(invalid_data_file, 'w') as f:
                for name, reason in self.invalid_data:
                    f.write(f"{name}\t{reason}\n")

    def __len__(self):
        return len(self.video_files)
            
    def __getitem__(self, idx):
        if idx >= len(self.video_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.video_files)}")
            
        video_name = self.video_files[idx]
        
        # Get paths
        video_path = self.data_root / 'raw' / 'videos' / f'LFROI_{video_name}.mp4'
        landmarks_path = self.data_root / 'raw' / 'landmarks' / f'{video_name}_points.csv'
        label_path = self.data_root / 'raw' / 'lab' / f'{video_name}.lab'
        
        # Check if all required files exist
        if not all(p.exists() for p in [video_path, landmarks_path, label_path]):
            raise FileNotFoundError(f"Required files not found for video {video_name}")
        
        try:
            # データの読み込みと処理
            frames = process_video(video_path, landmarks_path)
            frame_labels, phoneme_sequence = process_labels(label_path, self.label_processor)
            
            # データの検証
            if len(frames) == 0:
                raise ValueError(f"No frames processed for video {video_name}")
            if len(frame_labels) == 0 or len(phoneme_sequence) == 0:
                raise ValueError(f"No labels processed for video {video_name}")
            
            # Convert to torch tensors
            frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
            frame_labels = torch.LongTensor(frame_labels)
            phoneme_sequence = torch.LongTensor(phoneme_sequence)
            
            return frames, frame_labels, phoneme_sequence
            
        except Exception as e:
            print(f"Error processing item {idx} ({video_name}): {str(e)}")
            # 次の有効なインデックスを試す
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

def collate_fn(batch):
    """DataLoaderのcollate_fn
    
    Args:
        batch: list of tuples (frames, frame_labels, phoneme_sequence)
        
    Returns:
        tuple of padded tensors
    """

    # データ拡張のパラメータ
    noise_scale = 0.01
    brightness_scale = 0.1
    contrast_scale = 0.1

    # バッチ内の各サンプルのサイズを取得
    frames_lengths = [frames.size(1) for frames, _, _ in batch]
    phonemes_lengths = [phonemes.size(0) for _, _, phonemes in batch]
    
    # バッチ内の最大長を取得
    max_frames = max(frames_lengths)
    max_phonemes = max(phonemes_lengths)
    batch_size = len(batch)
    
    # パディング済みのテンソルを準備
    padded_frames = torch.zeros(batch_size, 3, max_frames, 96, 96)
    padded_frame_labels = torch.ones(batch_size, max_frames, dtype=torch.long) * -1  # padding_idx = -1
    padded_phonemes = torch.ones(batch_size, max_phonemes, dtype=torch.long) * -1    # padding_idx = -1
    
    # 実際の長さを保存
    frames_lengths = torch.tensor(frames_lengths)
    phonemes_lengths = torch.tensor(phonemes_lengths)
    
    # バッチ内の各サンプルをパディング
    for i, (frames, frame_labels, phonemes) in enumerate(batch):
        T = frames.size(1)
        P = phonemes.size(0)

        # フレームの拡張処理
        augmented_frames = frames.clone()
        
        # 1. ガウシアンノイズの追加
        noise = torch.randn_like(augmented_frames) * noise_scale
        augmented_frames = augmented_frames + noise
        
        # 2. ランダムな明るさ変更
        brightness_factor = 1.0 + (torch.rand(1) * 2 - 1) * brightness_scale
        augmented_frames = augmented_frames * brightness_factor
        
        # 3. ランダムなコントラスト変更
        contrast_factor = 1.0 + (torch.rand(1) * 2 - 1) * contrast_scale
        mean = augmented_frames.mean(dim=(2, 3), keepdim=True)
        augmented_frames = (augmented_frames - mean) * contrast_factor + mean
        # 4. 値の範囲を[0, 1]に制限
        augmented_frames = torch.clamp(augmented_frames, 0, 1)
        
        # シーケンス長を合わせる
        T = min(T, max_frames)  # フレーム数を最大長に制限

        # フレームのパディング（拡張済みフレームを使用）
        padded_frames[i, :, :T, :, :] = augmented_frames[:, :T, :, :]
        
        # フレームラベルのパディング
        if frame_labels.size(0) > T:
            padded_frame_labels[i, :T] = frame_labels[:T]
        else:
            padded_frame_labels[i, :frame_labels.size(0)] = frame_labels
            if frame_labels.size(0) < T:
                # 最後のラベルで残りを埋める
                padded_frame_labels[i, frame_labels.size(0):T] = frame_labels[-1]
        
        # 音素シーケンスのパディング
        P = min(P, max_phonemes)  # 音素シーケンス長を最大長に制限
        padded_phonemes[i, :P] = phonemes[:P]
    
    return {
        'frames': padded_frames,
        'frame_labels': padded_frame_labels,
        'phoneme_sequence': padded_phonemes,
        'frames_lengths': frames_lengths,
        'phonemes_lengths': phonemes_lengths
    }