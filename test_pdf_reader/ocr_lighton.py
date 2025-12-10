import torch
import os
import re
from PIL import Image
from transformers import AutoProcessor, LightOnOCRForConditionalGeneration
from pdf2image import convert_from_path

PDF_PATH = r"vladick.pdf"
CACHE_PATH = r"ocr_cache_version2.txt"
POPPLER_PATH = r"C:\Users\sanya\AppData\Roaming\Python\Python313\Scripts\poppler-25.11.0\Library\bin"

model_id = "lightonai/LightOnOCR-1B-1025"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = LightOnOCRForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device,
    attn_implementation="sdpa",
)
model.eval()


def ocr_pil_image(image: Image.Image) -> str:
    image = image.convert("RGB")

    max_width = 1200
    w, h = image.size
    if w > max_width:
        new_h = int(h * max_width / w)
        image = image.resize((max_width, new_h), Image.BILINEAR)

    messages = [{"role": "user", "content": [{"type": "image"}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            torch.bfloat16 if device == "cuda" else torch.float32
        )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_text = processor.tokenizer.decode(
        outputs[0, input_length:], skip_special_tokens=True
    )
    return generated_text


def postprocess_ocr_text(text: str) -> str:
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        if not line.strip():
            continue
            
        if re.search(r'Группа\s+\d+\.\d+.*Группа\s+\d+\.\d+', line):
            processed_lines.append(f"[ОБЪЕДИНЁННАЯ ПАРА] {line}")
            continue
        
        if re.search(r'^\s*[-–—]\s*$', line) or line.strip() == '':
            processed_lines.append("[ОКНО В РАСПИСАНИИ]")
            continue
            
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def ocr_pdf_to_text(pdf_path: str) -> str:
    print(" Запускаю OCR ...")
    pages = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH,
        dpi=150,
    )
    
    texts = []
    for i, page in enumerate(pages, start=1):
        print(f"Обработка страницы {i}/{len(pages)}...")
        raw_text = ocr_pil_image(page)
        
        processed_text = postprocess_ocr_text(raw_text)
        
        texts.append(f"=== СТРАНИЦА {i} ===\n{processed_text}")
    
    result = "\n\n".join(texts)
    
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"💾 Текст сохранён в {CACHE_PATH}")
    
    return result


if __name__ == "__main__":
    final_text = ocr_pdf_to_text(PDF_PATH)
    print("\n" + "="*50)
    print(final_text[:2000])
