import torch
import json
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import utils.utils as utils

import pdb

def load_video_features(json_path, device):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # assume 'data' key holds a list of feature vectors
    feats = torch.tensor(data['data'], dtype=torch.float32, device=device)
    return feats

def main():
    window_size = 3
    interval = 1
    stride = 1
    
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load clip model (same config you used before)
    config_name = "PE-Core-B16-224"
    model = pe.CLIP.from_config(config_name, pretrained=True)
    model = model.to(device).eval()

    # load pre-extracted video features
    video_feats = load_video_features("./results/data.json", device)
    # video_feats shape: [num_frames, feature_dim]

    # define your texts
    texts = ["a diagram", "a dog", "a cat"]
    
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    text_tokens = tokenizer(texts).to(device)

    # encode texts and compute logits
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda"):
                _, text_feats, logit_scale = model(text=text_tokens)
        else:
            _, text_feats, logit_scale = model(text=text_tokens)

        # text_feats shape: [num_texts, feature_dim]
        # compute similarity: [num_frames, num_texts]
        indices = utils.sliding_window_indices(len(video_feats), window_size, interval, stride)
        windows = [video_feats[idx] for idx in indices]
        avr_pool = torch.stack([x.mean(dim=0) for x in windows])
        
        logits = logit_scale * (avr_pool @ text_feats.T)
        probs = logits.softmax(dim=-1)

    print("similarity probabilities per frame:", probs)

if __name__ == "__main__":
    main()