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
        "dtype":None,
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

def convert_msg(image=None, promt=None, context_str=None, context_msgs=None):
  if context_msgs is not None:
    messages = context_msgs
  else:
    messages = []
  # Provide context for every promt
  if context_str is not None:
    messages.append(
      {
          'role': 'assistant',
          'content':[
              {'type': 'text', 'text': context_str}
          ]
      }
    )

  # Prepare messages template
  if image is not None:  # 1st msg include image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image,
            "max_pixels": MAX_PIXELS*28*28},
            {"type": "text", "text": promt},
        ],
    })
  else:
    # text promt only
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": promt}],
    })

  return messages

def process_question(messages=None, max_tokens=2048):
  texts = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)

  try:
    image_inputs, video_inputs = process_vision_info(messages)
  except AttributeError as atrr_err:
    print(atrr_err)
    image_inputs = None
    video_inputs = None

  inputs = processor(
      text=texts,
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
      do_sample=False,
      use_cache=True,
      max_new_tokens=max_tokens,
      temperature=1.5,
      min_p=0.1,
      repetition_penalty=1.1  # if output begins repeating increase
  )         # streamer=text_streamer,

  generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  ]
  output_texts = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
  )
  return output_texts


def process_input(image1, image2, max_tokens):
  import time

  start = time.time()

  global_outputs = []
  ocr_promt = "Nhận dạng văn bản xuất hiện trong bức ảnh"
  # image front
  promts_gr = [
      "Trả lời ngắn gọn thông tin của tổ chức nếu có: tên tổ chức. ",
      "Trả lời ngắn gọn thông tin của tổ chức nếu có: mã số doanh nghiệp",
      "Trả lời ngắn gọn thông tin của tổ chức nếu có: địa chỉ doanh nghiệp",
  ]
  promts1 = [
    "Trả lời ngắn gọn thông tin của người thứ nhất: họ và tên là gì",
    "Trả lời ngắn gọn thông tin của người thứ nhất: năm sinh là bao nhiêu",
    "Trả lời ngắn gọn thông tin của người thứ nhất: số cccd là bao nhiêu",
    "Trả lời ngắn gọn thông tin của người thứ nhất: địa chỉ ở đâu",
  ]
  promts2 = [
    "Trả lời ngắn gọn thông tin của người thứ hai nếu có: họ và tên là gì",
    "Trả lời ngắn gọn thông tin của người thứ hai nếu có: năm sinh là bao nhiêu",
    "Trả lời ngắn gọn thông tin của người thứ hai nếu có: số cccd là bao nhiêu",
    "Trả lời ngắn gọn thông tin của người thứ hai nếu có: địa chỉ ở đâu",
  ]

  messages_ctx = None
  ctx_str = None
  for idx, promt in enumerate([ocr_promt] + promts_gr + promts1 + promts2):
    if idx==0: # promt includes image
      messages = convert_msg(image=image1, promt=promt)
      ocr_outputs = process_question(messages)
      # store context
      messages_ctx = messages
      ctx_str = ocr_outputs[0]
      # global output
      global_outputs = global_outputs+ocr_outputs
      continue
    # text only promt
    messages = convert_msg(image=None, promt=promt, context_str=ctx_str, context_msgs=messages_ctx)
    outputs = process_question(messages)
    # store context
    messages_ctx = messages
    ctx_str = outputs[0]
    # global output
    global_outputs = global_outputs + outputs
  
  end = time.time()
  print('Time taken: ', end - start)

  return global_outputs

with gr.Blocks() as demo:
    gr.Markdown(f"""
    # Demo nhận diện văn bản từ tài liệu về GCN quyền sử dụng đất
    ## Mô hình LVLM: `{model_path}`
    """)
    with gr.Row():
        image_input_front = gr.Image(type="filepath", label="Mặt trước GCN quyền sử dụng đất", width=100)
        image_input_back = gr.Image(type="filepath", label="Mặt sau GCN quyền sử dụng đất", width=100)
    max_tokens = gr.Slider(2, 1024, value=1024, step=2, label="Max Tokens")
    scan_button = gr.Button("Scan")
    # Editor
    gr.Markdown("""
        ## Thông tin mặt trước
                """)
    content = gr.Textbox(label="Nội dung OCR")
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

    scan_button.click(process_input,\
                      inputs=[
                        image_input_front,
                        image_input_back,
                        max_tokens],\
                      outputs=[
                        content,
                        gr_name_output, gr_id_output, gr_address_output,
                        user_name_output, dob_output, cccd_output, address_output,
                        user_name_output2, dob_output2, cccd_output2, address_output2,
                        ])
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

    demo.launch(share=True)
