

import os
import json
import glob
import math
from datetime import datetime
import pdb
import pandas as pd

import cv2
import numpy as np

import torch
from PIL import Image, ImageDraw, ImageFont

import random

import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

def sliding_window_indices(num_frames: int,
                           window_size: int,
                           interval: int,
                           stride: int) -> list[list[int]]:
    """
    num_frames 길이의 시퀀스에 대해
    window_size, interval, stride 조건으로
    각 윈도우의 (유효한) 프레임 인덱스 리스트를 반환.

    음수이거나 num_frames 이상인 인덱스는 리스트에 포함되지 않음.
    """
    half = window_size // 2
    offset = - half * interval
    max_i = (num_frames - 1) // stride

    windows: list[list[int]] = []
    for i in range(max_i + 1):
        start = i * stride + offset
        idxs: list[int] = []
        for j in range(window_size):
            idx = start + j * interval
            if 0 <= idx < num_frames:
                idxs.append(idx)
        windows.append(idxs)
    return windows


def sliding_window_features(feats: torch.Tensor,
                            window_size: int,
                            interval: int,
                            stride: int
                           ):
    """
    video_feats (num_frames × feat_dim)에 sliding window를 적용해
    out[k][j]는 j번째 슬롯의 feature 혹은 None
    """
    num_frames, dim = feats.shape
    idx_lists = sliding_window_indices(num_frames, window_size, interval, stride)

    windows: List[List[Optional[torch.Tensor]]] = []
    for idxs in idx_lists:
        slot_feats: List[Optional[torch.Tensor]] = []
        for i in idxs:
            if 0 <= i < num_frames:
                slot_feats.append(feats[i])
            else:
                slot_feats.append(None)
        windows.append(slot_feats)

    return windows


def build_video_info(
    video_path: str,
    clip_seconds: int | None = None,   # 예: 4 ⇒ 앞에서 4초만 사용
    random_clip: bool = False          # True면 무작위 구간 추출
):
    """
    video_path만 가지고 (video_path, max_frames, s, e, None) 튜플을 만든다.

    Parameters
    ----------
    video_path : str
        영상 파일 경로
    clip_seconds : int | None
        앞에서 몇 초만 사용할지 결정. None이면 전체 길이를 사용
    random_clip : bool
        무작위 구간을 뽑을지 여부 (clip_seconds가 None이면 의미 없음)

    Returns
    -------
    tuple
        (video_path, max_frames, s, e, None)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30  # fps 정보가 없으면 30으로 가정
    cap.release()

    # 사용할 구간의 프레임 수 결정
    if clip_seconds is None:
        s = 0
        e = total_frames - 1
    else:
        clip_frames = min(total_frames, int(clip_seconds * fps))
        if random_clip and clip_frames < total_frames:
            s = random.randint(0, total_frames - clip_frames)
            e = s + clip_frames - 1
        else:
            s = 0
            e = clip_frames - 1

    max_frames = e - s + 1
    return (video_path, max_frames, s, e, fps)


def save_video(tensor: torch.Tensor, output_path: str, fps: int = 25):
    """
    주어진 비디오 텐서를 영상 파일로 저장합니다.

    Args:
        tensor (torch.Tensor): (T, 3, H, W) 형태의 비디오 텐서. 값은 0~255 범위여야 함.
        output_path (str): 저장할 .mp4 또는 .avi 파일 경로.
        fps (int): 초당 프레임 수 (기본값: 25).
    """
    assert tensor.ndim == 4 and tensor.shape[1] == 3, "입력은 (T, 3, H, W) 텐서여야 합니다."
    
    T, C, H, W = tensor.shape

    # torch.Tensor → numpy.ndarray, (T, H, W, C), uint8 타입으로 변환
    video_np = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    # OpenCV VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID'
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in video_np:
        # OpenCV는 BGR을 사용하므로 RGB → BGR 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"영상이 저장되었습니다: {output_path}")
    
    
    
class OpenCVVideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_avg_fps(self):
        return self.fps

    def read_frame_at_time(self, sec):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    
    def __len__(self):
        return self.frame_count

    def release(self):
        self.cap.release()



class VideoFromPIL:
    def __init__(self, output_path: str, fps: int = 30):
        """
        PIL 이미지를 누적하여 비디오로 저장하는 클래스

        Parameters:
        - output_path (str): 저장할 비디오 파일 경로 (예: 'output.mp4')
        - fps (int): 비디오의 초당 프레임 수
        """
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        self.size = None  # (width, height)

    def add_frame(self, pil_image: Image.Image):
        """
        PIL 이미지를 프레임으로 추가합니다.
        이미지 크기는 첫 프레임과 동일하게 자동 조정됩니다.
        """
        if self.size is None:
            self.size = pil_image.size
        elif pil_image.size != self.size:
            pil_image = pil_image.resize(self.size)

        self.frames.append(pil_image)

    def save(self):
        """
        누적된 프레임을 비디오로 저장합니다.
        """
        if not self.frames:
            raise RuntimeError("저장할 프레임이 없습니다.")

        width, height = self.size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        for img in self.frames:
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print(f"비디오가 저장되었습니다: {self.output_path}")



def save_config(config: dict, output_path: str):
    folder_path = os.path.dirname(output_path)
    file_name = os.path.basename(output_path)

    config_save_path = output_path+'.json'

    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)  # 디렉토리 생성 (존재하지 않으면)
    
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)  # JSON 파일로 저장



def save_model_output(output: dict, output_path: str):
    config_save_path = output_path+'.csv'
    
    # Ensure the save path exists
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    
    # Convert output dictionary to DataFrame
    df = pd.DataFrame.from_dict(data=output, orient='columns')

    # Save DataFrame to CSV
    df.to_csv(config_save_path, index=False)
    
    print(f"Results saved to {config_save_path}")



def get_video_files(folder_path, extensions=(".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".MOV")):
    """특정 폴더 내의 모든 영상 파일 경로를 반환"""
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    return video_files



def interpolate_missing_frames(data, fps, max_frame):
    # DataFrame 생성
    df = pd.DataFrame.from_dict(data)
    
    all_frames = pd.DataFrame({"frame": range(df["frame"].min(), df["frame"].max() + int(fps))})
    df_filled = all_frames.merge(df, on="frame", how="left")#.interpolate(method='nearest')

    df_interpolated = df_filled.fillna(method='ffill', limit=int(fps/2)).fillna(method='bfill', limit=int(fps/2))
    df_interpolated = df_interpolated.fillna(method='ffill').fillna(method='bfill')
    df_interpolated = df_interpolated.astype(int)
    df_interpolated = df_interpolated[:max_frame]
    
    result_dict = df_interpolated.to_dict(orient='dict')
    
    return result_dict


def video_transform(frames, transform):
    h, w = frames.shape[-2:]  # Image shape (C, H, W) or (N, C, H, W)
    frames = F.resize(
        frames, size=(transform.size, transform.size), interpolation=InterpolationMode.BICUBIC
    )
    frames = (
        frames.to(torch.float32) / 255.0
    )  # Convert to float and scale to [0, 1] range
    frames = transform.normalize(frames)
    
    return frames, (w, h)