import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import argparse
import os

# Use a relative path or environment variable for the model path
model_path = os.environ.get("QWEN_MODEL_PATH", "path/to/model")

def load_model(use_flash_attention=False):
    model_kwargs = {
        "torch_dtype": torch.float16,  # Use float16 for AWQ compatibility
        "device_map": "auto",
    }
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    return model

max_pixels=256*28*28
processor = AutoProcessor.from_pretrained(model_path,max_pixels=max_pixels)

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
                {"type": media_type, media_type: media},
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
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
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
            gr.Image(type="filepath", label="Upload Image (optional)"),
            gr.Textbox(label="Text Prompt", value="Mô tả nội dung hình ảnh"),
            gr.Slider(0.1, 2, value=1.5, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.1, label="min-p"),
            gr.Slider(2, 512, value=256, step=2, label="Max Tokens")
        ],
        outputs=gr.Textbox(label="Generated Description"),
        title="Qwen2-VL-2B Finetuned",
        description="Upload an image and enter a prompt to generate a description.",
    )
    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2-VL model with optional Flash Attention 2")
    parser.add_argument("--flash-attn2", action="store_true", help="Use Flash Attention 2")
    args = parser.parse_args()
    
    model = load_model(use_flash_attention=args.flash_attn2)
    interface = create_interface()
    interface.launch(share=True)
