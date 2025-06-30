import torch
from PIL import Image
import utils.utils as utils
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import core.transforms.video_transform as video_transform
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


import json
import pdb


window_size = 3
interval = 1
stride = 1

prompt_list = ["normal", "a smoky scene", "a photo of falldown", "a photo of fire"]

sampling_fps = 10

file_name = 'CCTV_footage__fixed_camera__high_resolution__inside_a_tunnel__A_fire_breaks_in_the_tunnel_with_smoke_ccf2eb'
# file_name = 'a_walljump_smoke'
# file_name = '99-4_cam03_swoon01_place03_day_winter'
# file_name = 'a_walljump_195745_195966'

print("CLIP configs:", pe.CLIP.available_configs())
# CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)  # Downloads from HF
# model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
model = model.cuda()

VideoTransform = video_transform.VideoTransform(model.image_size, normalize_img=True)
tokenizer = transforms.get_text_tokenizer(model.context_length)

video_path, max_frames, s, e, fps = utils.build_video_info(f"/workspace/nas_192/Coupang/smoke/videos/{file_name}.mp4")

# 0.1초 간격으로 뽑을 것이니까 sampling_fps=10
frames = VideoTransform.load_video(video_path, max_frames, sampling_fps, s, e)[0]
video_input = VideoTransform((video_path, max_frames, s, e, None), sampling_fps=sampling_fps)[0].cuda()   


    # h, w = image.shape[-2:]  # Image shape (C, H, W) or (N, C, H, W)
    # image = F.resize(
    #     image, size=(self.size, self.size), interpolation=InterpolationMode.BICUBIC
    # )
    # image = (
    #     image.to(torch.float32) / 255.0
    # )  # Convert to float and scale to [0, 1] range
    # image = self.normalize(image)
    # return image.cuda(), (w, h)
 
text = tokenizer(prompt_list).cuda()


# your texts and threshold
class_names = ["normal", "smoke", "falldown", "fire"]
target_text = "smoke"
threshold = 0.5  # 예: dog 확률 0.5 이상일 때 붉은 테두리

# 1) 프레임과 프로브 계산 (기존 코드)
with torch.no_grad(), torch.autocast("cuda"):
    image_features, _, logit_scale = model(image=video_input)
    _, text_features, _ = model(text=text)

    indices = utils.sliding_window_indices(len(image_features), window_size, interval, stride)
    windows = [image_features[idx] for idx in indices]
    avr_pool = torch.stack([x.mean(dim=0) for x in windows])
    text_probs = (logit_scale * avr_pool @ text_features.T).softmax(dim=-1)
# text_probs.shape == (num_windows, num_texts)

# 2) 프레임 텐서를 numpy 이미지 리스트로 변환
frames_cpu = frames.cpu()                            # (T, C, H, W)
frames_np   = (frames_cpu.permute(0,2,3,1).numpy()).astype(np.uint8)

pil_frames  = [Image.fromarray(f) for f in frames_np]

# 준비: 폰트 설정 (시스템에 따라 경로 조정)
try:
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 48)
except IOError:
    font = ImageFont.load_default()

# 3) 각 윈도우별로 문턱 넘은 프레임에 테두리 그리고,
#    각 프레임 우측 상단에 전체 확률 순위와 값 표시
target_idx = class_names.index(target_text)
for win_idx, idxs in enumerate(indices):
    probs = text_probs[win_idx]
    if (probs >= 0.7).any():
        # idx = probs.argmax()
        img = pil_frames[win_idx]
        
        draw = ImageDraw.Draw(img)
        w, h = img.size
        draw.rectangle(
            [0, 0, w-1, h-1],
            outline="red", width=4
        )

    sorted_idxs = torch.argsort(probs, descending=True).tolist()
    # for frame_i in idxs:
    img = pil_frames[win_idx]
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x0 = w - 200    # 오른쪽으로부터 200px 위치
    y0 = 10         # 위쪽에서 10px 내려온 위치
    for rank, cls_i in enumerate(sorted_idxs):
        prob_val = probs[cls_i].item()
        txt = f"{rank+1}. {class_names[cls_i]}: {prob_val:.2f}"
        draw.text((x0, y0 + rank*18), txt, fill="white", font=font)

# 4) OpenCV로 비디오 저장
h, w = pil_frames[0].size[1], pil_frames[0].size[0]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'./results/{file_name}.mp4', fourcc, sampling_fps, (w, h))
for img in pil_frames:
    img = np.array(img)
    img = img[..., [2, 1, 0]]
    out.write(np.array(img))
out.release()

print(f"annotated video saved to ./results/{file_name}.mp4")
