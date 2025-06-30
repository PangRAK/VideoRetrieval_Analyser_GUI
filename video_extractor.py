import torch
from PIL import Image
import utils.utils as utils
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import core.transforms.video_transform as video_transform

import json
import pdb


sampling_fps = 10

print("CLIP configs:", pe.CLIP.available_configs()) # CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)  # Downloads from HF
model = model.cuda()

preprocess = video_transform.get_video_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

video_path, max_frames, s, e, fps = utils.build_video_info("./apps/pe/docs/assets/dog.mp4")
frames = preprocess((video_path, max_frames, s, e, None), sampling_fps=sampling_fps)[0].cuda()    # 0.1초 간격으로 뽑을 것이니까 sampling_fps=10


with torch.no_grad(), torch.autocast("cuda"):
    
    image_features, _, logit_scale = model(image = frames)
    
    # # Prompt에 대해서 바로 연산하실 거면 아래의 주석을 제거하시고 사용하면 됩니다.
    # text = tokenizer(["a diagram", "a dog", "a cat"]).cuda()
    # _, text_features, logit_scale = model(text = text)
    # text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
    # print("Label probs:", text_probs)  # prints: [[0.0, 0.0, 1.0]]
    
data = {'file_path': video_path,
        'max_frame': max_frames,
        'fps': fps,
        'sampling_fps': sampling_fps,
        'data': image_features.tolist()}

with open("./results/data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    
print("Visual feature maps are saved")

