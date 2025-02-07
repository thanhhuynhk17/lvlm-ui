import gradio as gr
from transformers import TextStreamer
import torch
import argparse
from unsloth import FastVisionModel
import os
import time

from qwen_vl_utils import process_vision_info

# Use a relative path or environment variable for the model path
model_path = os.environ.get(
    "QWEN_MODEL_PATH", "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 1280))


def load_model(use_flash_attention=False):
    model_kwargs = {
        "max_seq_length": 2048,
        "dtype": torch.bfloat16,
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

def process_question(model, processor, promt=None, images=None, chat_history=None, max_tokens=1024):

    msg_content = [
        {'type': 'text', 'text': promt},
    ]
    if images != None:
        for image in images:
            msg_content.append(
                {'type': 'image', 'image': image, 'max_pixels': 800*28*28})

    msg = [
        {
            'role': 'system',
            'content':[
                {'type': 'text', 'text': "Hỗ trợ nhận diện văn bản tiếng Việt từ hình ảnh."}
            ]
        },
    ]
    if chat_history != None:
        msg.extend(chat_history)
    msg.append({'role': 'user', 'content': msg_content})

    text = processor.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True)

    try:
        image_inputs, video_inputs = process_vision_info(msg)
    except AttributeError as atrr_err:
        print(atrr_err)
        image_inputs = None
        video_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # streaming text
    text_streamer = TextStreamer(processor)

    generated_ids = model.generate(
        **inputs,
        streamer=text_streamer,
        do_sample=True,
        use_cache=True,
        max_new_tokens=max_tokens,
        temperature=0.6,
        min_p=0.1,
        top_p=0.95,
        # repetition_penalty=1.1  # if output begins repeating increase
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Empty the cache after inference to free up memory
    for i in range(3):
        torch.cuda.empty_cache()  # Fallback to empty the entire CUDA cache
        time.sleep(0.2)

    # add chat history
    if chat_history == None:
        chat_history = []
    chat_history.append({'role': 'user', 'content': msg_content})
    chat_history.append({'role': 'assistant', 'content': [
                        {'type': 'text', 'text': output_texts[0]}]})

    return output_texts, chat_history


def process_input(image1, image2, max_tokens):
    import time

    start = time.time()

    global_outputs = []
    chat_history = None
    ocr_promt = "Nhận diện văn bản từ hình ảnh\n"

    # infor extraction promt
    info_promt = """Trả lời ngắn gọn và chính xác thông tin được hỏi.
Nếu không tìm thấy hãy trả lời "<|not_found|>".
1. Chủ sở hữu đất trong giấy chứng nhận.
2. Trả lời thông tin về địa chỉ.
3. Trả lời thông tin về số CCCD hoặc CMND đối với cá nhân, mã công ty đối với công ty.
4. Số thửa đất và số tờ bản đồ.
5. Diện tích."""

    # query promt
    for promt in [ocr_promt, info_promt]:
        if chat_history == None:
            output, chat_history = process_question(model, processor,
                                                    promt=promt,
                                                    images=[image1, image2],
                                                    max_tokens=max_tokens)
        else:
            output, chat_history = process_question(model, processor,
                                                    promt=promt,
                                                    chat_history=chat_history,
                                                    max_tokens=max_tokens)
        global_outputs += output

    end = time.time()
    print('Time taken: ', end - start)

    return global_outputs


with gr.Blocks() as demo:
    gr.Markdown(f"""
    # Demo nhận diện văn bản từ tài liệu về GCN quyền sử dụng đất
    ## Mô hình LVLM: `{model_path}`
    """)


    with gr.Row():
        image_input_front = gr.Image(
            type="filepath", label="Mặt trước GCN quyền sử dụng đất", width=100)
        image_input_back = gr.Image(
            type="filepath", label="Mặt sau GCN quyền sử dụng đất", width=100)
    max_tokens = gr.Slider(2, 2048, value=1024, step=2, label="Max Tokens")
    scan_button = gr.Button("Scan")
    # Editor
    gr.Markdown("""## Nội dung OCR""")
    content = gr.Textbox(label="Nội dung OCR")

    gr.Markdown("""## Trích xuất thông tin""")
    info = gr.Textbox(label="Thông tin được trích xuất")

    with gr.Row():
        gr.Markdown("""
            ### Tổ chức
                    """)
        # Front side
        gr_name_output = gr.Textbox(label="Tên tổ chức")
        gr_id_output = gr.Textbox(label="Mã số doanh nghiệp")
        gr_address_output = gr.Textbox(label="Địa chỉ")
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
        user_name_output2 = gr.Textbox(label="Họ và tên")
        dob_output2 = gr.Textbox(label="Năm sinh")
        cccd_output2 = gr.Textbox(label="Số CCCD")
        address_output2 = gr.Textbox(label="Địa chỉ thường trú")

    scan_button.click(process_input,
                        inputs=[
                            image_input_front,
                            image_input_back,
                            max_tokens],
                        outputs=[
                            content, info
                        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Run {model_path} model with optional Flash Attention 2")
    parser.add_argument("--flash-attn2", action="store_true",
                        help="Use Flash Attention 2")
    args = parser.parse_args()

    model, processor = load_model(use_flash_attention=args.flash_attn2)

    demo.launch(share=True)
