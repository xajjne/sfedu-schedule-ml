import re
import json
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageOps
from pdf2image import convert_from_path
from transformers import AutoProcessor, LightOnOCRForConditionalGeneration


# ================= НАСТРОЙКИ =================
PDF_PATH = "filevlad.pdf"
POPPLER_PATH = r"C:\Users\sanya\AppData\Roaming\Python\Python313\Scripts\poppler-25.11.0\Library\bin"

OCR_MODEL = "lightonai/LightOnOCR-1B-1025"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_TXT = "ocr_result.txt"
OUT_TXT_CLEAN = "ocr_result_clean.txt"
OUT_JSON = "schedule_output.json"

DPI = 300


# ================= OCR =================
processor = AutoProcessor.from_pretrained(OCR_MODEL)
model = LightOnOCRForConditionalGeneration.from_pretrained(
    OCR_MODEL,
    device_map=DEVICE,
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    attn_implementation="sdpa",
)
model.eval()


# ================= PREPROCESS =================
def normalize_page(img: Image.Image) -> Image.Image:
    """
    Оставляем максимально безопасную предобработку.
    Важно: не ломаем таблицу агрессивными фильтрами.
    """
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    return img.convert("RGB")


# ================= OCR PAGE =================
def ocr_page(page: Image.Image) -> str:
    page = normalize_page(page)

    # ВАЖНО: как в твоём рабочем варианте — только image, без текстовой инструкции
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[page],
        return_tensors="pt"
    ).to(DEVICE)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            torch.bfloat16 if DEVICE == "cuda" else torch.float32
        )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )

    input_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(
        output[0, input_len:], skip_special_tokens=True
    )


# ================= OCR PDF =================
def ocr_pdf(pdf_path: str) -> str:
    pages = convert_from_path(
        pdf_path,
        dpi=DPI,
        poppler_path=POPPLER_PATH
    )

    all_pages: List[str] = []
    for i, page in enumerate(pages, start=1):
        print(f"📄 OCR страница {i}/{len(pages)}")
        text = ocr_page(page)
        all_pages.append(f"=== СТРАНИЦА {i} ===\n{text}")

    return "\n\n".join(all_pages)


# ================= POSTPROCESS: CLEAN =================
PLACEHOLDER_PATTERNS = [
    r"^\|\s*Column\s+1\s*\|",                         # markdown-плейсхолдеры
    r"^\|\s*Row\s+\d+\s*\|\s*Data",
    r"^Note:\s+The provided image contains a blank table",
]

def _looks_like_placeholder_table(line: str) -> bool:
    for p in PLACEHOLDER_PATTERNS:
        if re.search(p, line.strip(), flags=re.IGNORECASE):
            return True
    return False

def clean_ocr_text(raw: str) -> str:
    """
    Убираем мусор, который появился из-за генеративных "шаблонов"
    (на случай если где-то всё равно вылезет).
    """
    lines = raw.splitlines()
    out: List[str] = []
    skip_block = False

    for ln in lines:
        s = ln.strip()

        # скипаем блоки placeholder-table
        if _looks_like_placeholder_table(s):
            skip_block = True

        if skip_block:
            # заканчиваем пропуск, когда таблица закончилась (пустая строка после таблицы)
            if not s:
                skip_block = False
            continue

        # убираем маркдаун-картинки вида ![image](...)
        if s.startswith("![") and "](" in s:
            continue

        # иногда модель пишет много "===" или мусорные заголовки
        if s.lower().startswith("втревпало"):
            continue

        out.append(ln)

    # чуть подчистим многократные пустые строки
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ================= POSTPROCESS: STRUCTURE (days + time slots) =================
DAY_RE = re.compile(r"^(ПОНЕДЕЛЬНИК|ВТОРНИК|СРЕДА|ЧЕТВЕРГ|ПЯТНИЦА|СУББОТА)\b", re.IGNORECASE)

# ловим формы:
# 8 00 -8 45 / 9 50 -1035 / 1155 - 1240 / 8:00-8:45 / 8.00-8.45
TIME_RE = re.compile(
    r"(?P<h1>\d{1,2})\s*[:.]?\s*(?P<m1>\d{2})\s*[-–—]\s*(?P<h2>\d{1,2})\s*[:.]?\s*(?P<m2>\d{2})"
)

def normalize_time_range_in_line(line: str) -> Tuple[str, Optional[str]]:
    t = line
    t = t.replace("—", "-").replace("–", "-")
    t = t.replace("О", "0").replace("о", "0")

    m = TIME_RE.search(t)
    if not m:
        return t, None

    h1 = int(m.group("h1")); m1 = int(m.group("m1"))
    h2 = int(m.group("h2")); m2 = int(m.group("m2"))
    norm = f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
    t2 = TIME_RE.sub(norm, t, count=1)
    return t2, norm

def build_structured_schedule(page_text: str) -> Dict:
    """
    Не пытаемся идеальной “табличной” реконструкции (это сложно без детектора ячеек),
    но делаем полезную структуру:
    day -> time_slot -> lines
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    result: Dict[str, Dict[str, List[str]]] = {}
    cur_day: Optional[str] = None
    cur_time: Optional[str] = None

    for ln in lines:
        # нормализуем время внутри строки
        ln2, tr = normalize_time_range_in_line(ln)

        # день недели
        dm = DAY_RE.match(ln2)
        if dm:
            cur_day = dm.group(1).upper()
            result.setdefault(cur_day, {})
            cur_time = None
            continue

        # таймслот (если найден)
        if tr:
            cur_time = tr
            if cur_day is None:
                cur_day = "_NO_DAY_"
                result.setdefault(cur_day, {})
            result[cur_day].setdefault(cur_time, [])
            # если в строке кроме времени есть текст — добавим
            rest = ln2.replace(tr, "").strip(" |:-")
            if rest:
                result[cur_day][cur_time].append(rest)
            continue

        # обычная строка контента
        if cur_day is None:
            cur_day = "_NO_DAY_"
            result.setdefault(cur_day, {})
        if cur_time is None:
            cur_time = "_NO_TIME_"
            result[cur_day].setdefault(cur_time, [])
        result[cur_day][cur_time].append(ln2)

    return result


def split_pages_from_text(text: str) -> Dict[int, str]:
    pages: Dict[int, str] = {}
    chunks = text.split("=== СТРАНИЦА ")
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        m = re.match(r"(\d+)\s*===\s*\n(.*)$", ch, re.S)
        if not m:
            continue
        pnum = int(m.group(1))
        raw_page = m.group(2).strip()
        pages[pnum] = raw_page
    return pages


# ================= MAIN =================
if __name__ == "__main__":
    # 1) OCR
    text = ocr_pdf(PDF_PATH)

    # 2) raw txt
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)

    # 3) clean txt
    cleaned_pages: List[str] = []
    pages_map = split_pages_from_text(text)
    for pnum in sorted(pages_map.keys()):
        cleaned = clean_ocr_text(pages_map[pnum])
        cleaned_pages.append(f"=== СТРАНИЦА {pnum} ===\n{cleaned}")

    cleaned_text = "\n\n".join(cleaned_pages)
    with open(OUT_TXT_CLEAN, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # 4) structured json
    structured: Dict[int, Dict] = {}
    for pnum in sorted(pages_map.keys()):
        structured[pnum] = build_structured_schedule(clean_ocr_text(pages_map[pnum]))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print("✅ OCR завершён")
    print(f"📝 raw:   {OUT_TXT}")
    print(f"🧹 clean: {OUT_TXT_CLEAN}")
    print(f"🧩 json:  {OUT_JSON}")
