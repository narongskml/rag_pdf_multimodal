# ✅ Multimodal RAG Chatbot: People + Food, Image + Text, Answered by Ollama LLM (Chat UI)
# ---------------------------------------------------------
# Stack: Python + Gradio + Qdrant + CLIP + PyMuPDF + Ollama
# Requirements:
# pip install gradio transformers torch qdrant-client pymupdf pillow ollama

import gradio as gr
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import pymupdf as fitz  # PyMuPDF
import uuid
import ollama
import shutil
import io

# Init Qdrant
client = QdrantClient(url="http://localhost:6333")
COLLECTION = "multimodal_kb"

if COLLECTION not in [c.name for c in client.get_collections().collections]:    
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

UPLOAD_DIR = "data/kb"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Clear Knowledge Base

def clear_knowledge_base():
    client.delete_collection(collection_name=COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return "ล้างข้อมูลใน Knowledge Base เรียบร้อยแล้ว ✅"

# --- PDF Processing (Assume exists as `process_pdf_upload`)
def process_pdf_upload(pdf_file, data_type):
    import fitz  # ใช้ใน local scope เผื่อระบบฝั่ง Production แยก module
    
    doc = fitz.open(pdf_file.name)
    results = []
    id_base = str(uuid.uuid4())[:8]
    index = 0

    for page in doc:
        images = page.get_images(full=True)
        text = page.get_text().strip()

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            
            image = Image.open(io.BytesIO(image_bytes))
            file_name = f"{id_base}_{index}.{img_ext}"
            file_path = os.path.join(UPLOAD_DIR, file_name)
            image.save(file_path)

            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                vector = clip_model.get_image_features(**inputs)[0].numpy()

            # Text processing
            lines = text.split("\n")
            if data_type == "food":
                title = lines[0] if lines else "Unknown Dish"
                description = ""
                ingredients = ""
                steps = ""
                for line in lines[1:]:
                    if "ingredient" in line.lower():
                        ingredients += line + "\n"
                    elif "step" in line.lower() or "วิธี" in line.lower():
                        steps += line + "\n"
                    else:
                        description += line + " "

                payload = {
                    "type": "food",
                    "menu_name": title.strip(),
                    "description": description.strip(),
                    "ingredients": ingredients.strip(),
                    "steps": steps.strip(),
                    "image_path": file_path
                }
            else:
                name = lines[0] if lines else "Unknown Person"
                bio = " ".join(lines[1:])
                payload = {
                    "type": "person",
                    "name": name.strip(),
                    "bio": bio.strip(),
                    "image_path": file_path
                }
            client.upsert(
                collection_name=COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        payload=payload,
                        vector=vector.tolist(),
                    ),
                ],
            )
           
            results.append((payload.get("name", payload.get("menu_name", "Unknown")), file_path))
            index += 1

    return f"อัปโหลดและบันทึกข้อมูลจำนวน {len(results)} รายการ เรียบร้อยแล้ว ✅"

# --- Streaming LLM Response with Conversation Memory

def stream_llm_response_with_history(history, question, context, model_name):
    model_map = {
        "llama3": "llama3",
        "mistral": "mistral"      
    }
    model = model_map.get(model_name, "llama3")

    messages = [
        {"role": "system", "content": "คุณคือผู้ช่วยที่สามารถตอบคำถามจากข้อมูลที่มีให้เท่านั้น"},
    ]
    for user_input, ai_response in history[:-1]:
        if user_input:
            messages.append({"role": "user", "content": user_input})
        if ai_response:
            messages.append({"role": "assistant", "content": ai_response})

    user_prompt = f"คำถาม: {question}\n\nข้อมูลที่มี:\n{context}"
    messages.append({"role": "user", "content": user_prompt})

    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']
    except Exception as e:
        yield f"เกิดข้อผิดพลาดกับ Ollama: {str(e)}"

# --- Chat Handler for ChatInterface with memory

def handle_chat(history, image, mode, chat_model):
    question = history[-1][0] if history else "ภาพนี้คืออะไร"

    if image is not None:
        pil_img = Image.fromarray(image)
        inputs = clip_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            vector = clip_model.get_image_features(**inputs)[0].numpy()

        filter_type = "food" if mode == "อาหาร" else "person"
        result = client.search(
            collection_name=COLLECTION,
            query_vector=vector.tolist(),
            limit=1,
            query_filter={"must": [{"key": "type", "match": {"value": filter_type}}]}
        )
        if not result:
            return history + [[question, "ไม่พบข้อมูลในระบบ"]]

        match = result[0].payload
        context = match.get("bio") if filter_type == "person" else f"{match['description']}\n{match['ingredients']}\n{match['steps']}"
    else:
        context = "(ไม่พบภาพประกอบ กรุณาอัปโหลดรูปเพื่อค้นหา)"

    stream = stream_llm_response_with_history(history, question, context, chat_model)
    full = ""
    for chunk in stream:
        full += chunk
        yield history[:-1] + [[question, full]]

# --- Gradio UI
with gr.Blocks(title="🧠 Multimodal RAG Chatbot") as demo:
    gr.Markdown("""# 🧠 Chatbot AI รู้จำภาพ + เอกสาร สำหรับบุคคลและเมนูอาหาร
คุณสามารถพิมพ์คำถามและอัปโหลดรูปภาพ AI จะค้นหาข้อมูลจากฐานความรู้เอกสารและตอบกลับแบบ Streaming พร้อมจำบทสนทนาเดิม
""")

    with gr.Tab("📥 Admin Upload PDF"):
        pdf_file = gr.File(file_types=[".pdf"], label="อัปโหลด PDF")
        data_type = gr.Radio(choices=["food", "person"], label="ประเภทข้อมูล")
        upload_btn = gr.Button("อัปโหลด")
        clear_kb_btn = gr.Button("🗑️ ล้าง Knowledge Base")
        upload_result = gr.Textbox()
        upload_btn.click(fn=process_pdf_upload, inputs=[pdf_file, data_type], outputs=upload_result)
        clear_kb_btn.click(fn=clear_knowledge_base, outputs=upload_result)

    with gr.Tab("💬 Chat Interface"):
        with gr.Row():
            chatbot = gr.Chatbot()
            with gr.Column():
                img = gr.Image(type="numpy", label="📸 รูปภาพที่ต้องการสอบถาม")
                mode = gr.Radio(["อาหาร", "บุคคล"], value="อาหาร", label="ประเภทข้อมูล")
                model = gr.Dropdown(["llama3", "mistral", "gpt-3.5", "gpt-4"], value="llama3", label="LLM Model")
        txt = gr.Textbox(placeholder="พิมพ์คำถามแล้วกด Enter...")
        clear_chat_btn = gr.Button("🔄 ล้างบทสนทนา")

        def add_text(history, user_text):
            return history + [[user_text, None]], ""

        def clear_history():
            return []

        txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
            handle_chat, [chatbot, img, mode, model], chatbot
        )
        clear_chat_btn.click(fn=clear_history, outputs=chatbot)

# --- Run app

demo.launch()
