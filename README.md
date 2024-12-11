
# **Qwen2-VL Gradio UI**


## **Requirements**

- Python 3.8+
- CUDA-compatible GPU (recommended)

---

## **Setup**

1. **Clone this repository**:
    ```bash
    git clone https://github.com/Kaszebe/Large-Vision-Language-Model-UI.git
    cd Large-Vision-Language-Model-UI
    ```
   
2. **Create a virtual environment**:
    ```bash
    python3 -m venv qwen_venv
    ```

3. **Activate the virtual environment**:
    ```bash
    source qwen_venv/bin/activate
    ```

4. **Install the required packages**:
    ```bash
    pip install transformers accelerate qwen-vl-utils gradio

    pip install flash-attn --no-build-isolation
    ```

5. **Download the Qwen2-VL model**:
    - Visit [Hugging Face Model](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ)
    - Download the model files
    - Place the files in a directory on your local machine
    - Set the `QWEN_MODEL_PATH` environment variable:
      ```bash
      export QWEN_MODEL_PATH="/path/to/your/model"
      ```

---

## **Running the Model**

1. **Activate the virtual environment** (if not already activated):
    ```bash
    source qwen_venv/bin/activate
    ```

2. **Run the script**:
    ```bash
    python run_qwen_model.py --flash-attn2
    ```
    This will launch the Gradio interface, allowing you to interact with the model through a web browser. You can upload images to generate descriptions.
---

## **Usage**

1. Open the provided URL in your web browser.
2. **Upload** an image or video.
3. Enter a **text prompt**.
4. Adjust **generation parameters** if needed.
5. Click **"Submit"** to generate a description.

---

## **Notes**

- The model performs best with a **CUDA-compatible GPU**.
- Processing time may vary depending on your hardware and the complexity of the input.

--Enjoy!
---
 
