import time
import statistics
import os
import pdb
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
# from torch.nn.functional import pad

import cv2
from PIL import Image, ImageDraw
import numpy as np


import math

import utils.utils as utils


import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import core.transforms.video_transform as video_transform


import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def main(
        video_path,
        prompt_list,
        categories,
        window_size,
        interval,
        stride,
        sampling_fps,
        output_path,
        split,
        show_video
    ):

    video_files = utils.get_video_files(video_path)
    
    # model setting
    print("CLIP configs:", pe.CLIP.available_configs())
    model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True).cuda()  # Downloads from HF
    VideoTransform = video_transform.VideoTransform(model.image_size, normalize_img=True)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    

    if split is not None:
        # 각 구간의 범위 계산
        total_files = len(video_files)
        chunk_size = math.ceil(total_files / split[0])

        start_idx = (split[1] - 1) * chunk_size
        end_idx = min(split[1] * chunk_size, total_files)

        # 해당 범위의 파일 선택
        video_files = video_files[start_idx:end_idx]
    
    pred_path = os.path.join(output_path, 'pred')
    # pred_path = os.path.join(output_path, 'pred', datetime.now().strftime('%y%m%d-%X'))
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)  
    
    for video_file in video_files:

        file_name = os.path.splitext(os.path.basename(video_file))[0]
        
        
        # visual/text 전처리
        video_path, max_frames, s, e, fps = utils.build_video_info(video_file)
        # 0.1초 간격으로 뽑을 것이니까 sampling_fps=10
        frames = VideoTransform.load_video(video_path, max_frames, sampling_fps, s, e)[0]
        # video_input = VideoTransform((video_path, max_frames, s, e, None), sampling_fps=sampling_fps)[0].cuda()   
        
        print("[System] Start preprocessing!!")
        video_input, (w, h) = utils.video_transform(frames, VideoTransform)
        print("[System] Finished preprocessing!!")
        video_input = video_input.cuda()
        text = tokenizer(prompt_list).cuda()


        # inferencing
        time_sequence = np.arange(0, max_frames)
        for t in time_sequence:
            image_input = video_input[t]
            image_input = image_input.unsqueeze(0)
            if image_input is None:
                print(f"Frame at {t}s could not be read.")
                continue
            
            with torch.no_grad(), torch.autocast("cuda"):
                image_features, _, logit_scale = model(image=image_input)
                _, text_features, _ = model(text=text)

                indices = utils.sliding_window_indices(len(image_features), window_size, interval, stride)
                pdb.set_trace()
                windows = [image_features[idx] for idx in indices]
                avr_pool = torch.stack([x.mean(dim=0) for x in windows])
                text_probs = (logit_scale * avr_pool @ text_features.T).softmax(dim=-1)
            
            pass
        
        # 결과 저장용 변수 초기화
        output_dict = defaultdict(list)
        ratio_dict = defaultdict(list)
        inference_time = []
        preprocess_time = []
        total_time = []
        
        # 비디오 로딩 및 설정
        vr = OpenCVVideoReader(video_file)
        max_frame = len(vr)# - 1
        fps = float(vr.get_avg_fps())
        video_length = max_frame / fps
        video_sec = round(video_length)
        
        
        
        
        utils.save_config(config, os.path.join(pred_path, file_name))
        utils.save_model_output(output_dict, os.path.join(pred_path, file_name))


if __name__ == "__main__":
    
    video_path = "/workspace/nas_192/Coupang/smoke/CV/videos"
    prompt_list = ["normal", "a smoky scene", "a photo of falldown", "a photo of fire"]

    window_size = 3     # 윈도우 사이즈(초)
    interval = 1    # 윈도우 내에서 샘플링 수
    stride = 1 # stride
    sampling_fps = 10
    split = (1,1)
    categories = ["smoke", "normal"]
    output_path = '/workspace/sangrak/benchmarking/results/SmokingCls_v0.3.0_Foreigner'
    show_video = True   # 비디오 시각화 여부, True시 time_interval과 무관하게 원본 fps값으로 고정

    
    main(
        video_path=video_path,
        prompt_list=prompt_list,
        categories=categories,
        window_size = window_size,
        interval = interval,
        stride = stride,
        sampling_fps = sampling_fps,
        output_path = output_path,
        split = split,
        show_video = show_video
    )

        
        
    