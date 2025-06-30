import gradio as gr
import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import core.transforms.video_transform as video_transform
import utils.utils as utils

# 전역 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_name = "PE-Core-B16-224"
model = pe.CLIP.from_config(config_name, pretrained=True).to(device).eval()
preprocess = video_transform.get_video_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

def extract_and_cache(video_path):
    sampling_fps = 10
    video_path, max_frames, s, e, fps = utils.build_video_info(video_path)
    frames = preprocess((video_path, max_frames, s, e, None), sampling_fps=sampling_fps)[0].to(device)
    with torch.no_grad(), torch.autocast("cuda" if device.type=="cuda" else "cpu"):
        image_feats, _, logit_scale = model(image=frames)
        texts = ["a diagram", "a dog", "a cat"]
        text_tokens = tokenizer(texts).to(device)
        _, text_feats, _ = model(text=text_tokens)
    return image_feats.cpu(), text_feats.cpu(), logit_scale.cpu()

def pool_and_predict(image_feats, text_feats, logit_scale, window_size, interval, stride):
    image_feats = image_feats.to(device)
    text_feats = text_feats.to(device)
    logit_scale = logit_scale.to(device)
    indices = utils.sliding_window_indices(image_feats.shape[0], window_size, interval, stride)
    pooled = torch.stack([image_feats[idx].mean(dim=0) for idx in indices])
    logits = logit_scale * (pooled @ text_feats.T)
    probs = logits.softmax(dim=-1).cpu().tolist()
    return "\n".join([f"window {i}: {p}" for i, p in enumerate(probs)])

# 새로 추가: state inspect 함수
def inspect_state(image_feats):
    shape_str = f"image_feats shape: "
    return shape_str, preview

with gr.Blocks() as demo:
    gr.Markdown("## 비디오 업로드 → feature 캐싱 → pooling & 텍스트 유사도 계산")

    with gr.Row():
        video_input        = gr.Video(label="MP4 영상 업로드")
        with gr.Column():
            window_slider   = gr.Slider(1, 20, value=3, step=1, label="window_size")
            interval_slider = gr.Slider(1, 20, value=1, step=1, label="interval")
            stride_slider   = gr.Slider(1, 20, value=1, step=1, label="stride")
            submit_btn      = gr.Button("Submit")
            inspect_btn     = gr.Button("Inspect Features")   # ← 추가

    result_box         = gr.Textbox(label="각 윈도우별 텍스트 유사도 (softmax)")
    image_state        = gr.State()
    text_state         = gr.State()
    scale_state        = gr.State()

    # inspect 용 출력 컴포넌트
    shape_box          = gr.Textbox(label="Feature Shape")
    preview_table      = gr.Dataframe(label="First 5 Frame Features")

    # 이벤트 바인딩
    video_input.upload(fn=extract_and_cache,
                       inputs=[video_input],
                       outputs=[image_state, text_state, scale_state])

    submit_btn.click(fn=pool_and_predict,
                     inputs=[image_state, text_state, scale_state,
                             window_slider, interval_slider, stride_slider],
                     outputs=[result_box])

    inspect_btn.click(fn=inspect_state,
                      inputs=[image_state],
                      outputs=[shape_box, preview_table])

if __name__ == "__main__":
    demo.launch()