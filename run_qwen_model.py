import gradio as gr
from transformers import TextStreamer
import torch
import argparse
from unsloth import FastVisionModel
import os

from qwen_vl_utils import process_vision_info

# Use a relative path or environment variable for the model path
model_path = os.environ.get(
    "QWEN_MODEL_PATH", "thanhhuynhk17/qwen2-vl-2b-ft-freeze-vit")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 1280))


def load_model(use_flash_attention=False):
    model_kwargs = {
        "max_seq_length": 2048,
        "dtype": "auto",
        "load_in_4bit": True
    }
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_path,  # YOUR MODEL YOU USED FOR TRAINING
        **model_kwargs

    )
    FastVisionModel.for_inference(model)  # Enable native 2x faster inference

    return model, processor


def greet(name):
    return "Hello " + name + "!"


def process_question(image, prompt, max_tokens=256):
    if image is not None:
        media_type = "image"
        media = image
    else:
        return "Ảnh không tìm thấy!"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": media_type, media_type: media,
                    "max_pixels": MAX_PIXELS*28*28},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
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
        streamer=text_streamer,
        use_cache=True,
        max_new_tokens=max_tokens,
        temperature=1.5,
        min_p=0.1
    )

    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    response = output_text[0].split("assistant\n")[-1].strip()
    return response


def process_input(image1, image2, max_tokens):
    return "not implemented"

with gr.Blocks() as demo:
    gr.Markdown("# Demo nhận diện văn bản từ tài liệu về GCN quyền sử dụng đất")
    with gr.Row():
        image_input_front = gr.Image(label="Mặt trước GCN quyền sử dụng đất", width=100)
        image_input_back = gr.Image(label="Mặt sau GCN quyền sử dụng đất", width=100)
    scan_button = gr.Button("Scan")
    # Editor
    gr.Markdown("""
        ## Thông tin mặt trước
                """)
    with gr.Row():
        gr.Markdown("""
            ### Người thứ nhất
                    """)
        # Front side
        user_name_output = gr.Textbox(label="Họ và tên")
        dob_output = gr.Textbox(label="Năm sinh")
        cccd_output = gr.Textbox(label="Số CCCD")
        address_output = gr.Textbox(label="Địa chỉ thường trú")
    with gr.Row():
        gr.Markdown("""
            ### Người thứ hai (nếu có)
                    """)
        # Front side
        user_name_output = gr.Textbox(label="Họ và tên")
        dob_output = gr.Textbox(label="Năm sinh")
        cccd_output = gr.Textbox(label="Số CCCD")
        address_output = gr.Textbox(label="Địa chỉ thường trú")

    # gr.Markdown("""
    #     ## Thông tin mặt sau
    #             """)
    # with gr.Row():
    #     # Front side
    #     user_name_output = gr.Textbox(label="Họ và tên")
    #     dob_output = gr.Textbox(label="Năm sinh")
    #     cccd_output = gr.Textbox(label="Số CCCD")
    #     address_output = gr.Textbox(label="Địa chỉ thường trú")
    #     # Back side


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Run {model_path} model with optional Flash Attention 2")
    parser.add_argument("--flash-attn2", action="store_true",
                        help="Use Flash Attention 2")
    args = parser.parse_args()


    model, processor = load_model(use_flash_attention=args.flash_attn2)

        
    demo.launch()
