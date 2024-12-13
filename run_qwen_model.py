import gradio as gr
from transformers import TextStreamer
import torch
import argparse
import os
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info

# Use a relative path or environment variable for the model path
model_path = os.environ.get("QWEN_MODEL_PATH", "thanhhuynhk17/qwen2-vl-2b-ft-freeze-vit")
MAX_PIXELS = os.environ.get("MAX_PIXELS", 1280)

def load_model(use_flash_attention=False):
    model_kwargs = {
        "max_seq_length": 2048,
        "dtype": "auto",
        "load_in_4bit":True
    }
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    model, processor = FastVisionModel.from_pretrained(
        model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
        **model_kwargs

    )
    FastVisionModel.for_inference(model) # Enable native 2x faster inference
        
    return model, processor


def process_input(image, prompt, temperature=1.5, min_p=0.1, max_tokens=256):
    if image is not None:
        media_type = "image"
        media = image
    else:
        return "Please upload an image"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": media_type, media_type: media, "max_pixels": MAX_PIXELS*28*28},
                {"type": "text", "text": prompt},
            ],
        }
    ]    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    # streaming text
    text_streamer = TextStreamer(processor)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        do_sample=True,
        streamer = text_streamer,
        use_cache=True,
        max_new_tokens=max_tokens,
        temperature=temperature,
        min_p=min_p
    )
    
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    response = output_text[0].split("assistant\n")[-1].strip()
    return response

def create_interface():
    interface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Image(type="filepath", label="Upload Image"),
            gr.Textbox(label="Text Prompt", value="Nhận diện văn bản xuất hiện trong ảnh với định dạng markdown"),
            gr.Slider(0.1, 2, value=1.5, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.1, label="min-p"),
            gr.Slider(2, 1024, value=512, step=4, label="Max Tokens")
        ],
        outputs=gr.Textbox(label="Generated Description"),
        title=f"Model name: {model_path}",
        description="Upload an image and enter a prompt to generate a description.",
    )
    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2-VL model with optional Flash Attention 2")
    parser.add_argument("--flash-attn2", action="store_true", help="Use Flash Attention 2")
    args = parser.parse_args()
    
    model, processor = load_model(use_flash_attention=args.flash_attn2)
    interface = create_interface()
    interface.launch(share=True)
