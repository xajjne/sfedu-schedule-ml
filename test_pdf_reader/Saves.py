# ocr_lighton.py
import torch
import os
from PIL import Image
from transformers import AutoProcessor, LightOnOCRForConditionalGeneration
from pdf2image import convert_from_path

PDF_PATH = r"vladick.pdf"
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


def ocr_pdf_to_text(pdf_path: str) -> str:
    pages = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH,
        dpi=150,
    )
    texts = []
    for page in pages:
        texts.append(ocr_pil_image(page))
    return "\n\n".join(texts)

    with open("vladick_ocr_output.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("💾 Текст сохранён в vladick_ocr_output.txt")
    
    return result


if __name__ == "__main__":
    print(ocr_pdf_to_text(PDF_PATH))






# ocr_lighton.py v2
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







# ocr_lighton.py v3 by deepsick
import torch
import os
import re
from PIL import Image
from transformers import AutoProcessor, LightOnOCRForConditionalGeneration
from pdf2image import convert_from_path
import pandas as pd
from collections import defaultdict

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


def preprocess_image(image: Image.Image) -> Image.Image:
    """Предобработка изображения для улучшения OCR"""
    image = image.convert("RGB")
    
    # Увеличиваем контрастность и яркость
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    # Масштабируем
    max_width = 1800
    w, h = image.size
    if w > max_width:
        new_h = int(h * max_width / w)
        image = image.resize((max_width, new_h), Image.LANCZOS)
    
    return image


def ocr_pil_image(image: Image.Image) -> str:
    """Выполнение OCR на изображении"""
    image = preprocess_image(image)

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
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_text = processor.tokenizer.decode(
        outputs[0, input_length:], skip_special_tokens=True
    )
    return generated_text


def parse_time_interval(time_str):
    """Парсинг временного интервала из строки"""
    time_str = re.sub(r'[^\d\s\-–:.]', '', time_str)  # Удаляем все кроме цифр и разделителей
    time_str = re.sub(r'\s+', ' ', time_str).strip()
    
    # Ищем паттерны времени
    patterns = [
        r'(\d{1,2})[.:]?(\d{2})\s*[-–]\s*(\d{1,2})[.:]?(\d{2})',  # 8:00-9:35
        r'(\d{1,2})\s*[-–]\s*(\d{1,2})[.:]?(\d{2})',  # 8-9:35
        r'(\d{1,2})\s+(\d{1,2})\s*[-–]\s*(\d{1,2})\s+(\d{1,2})',  # 8 00 9 35
    ]
    
    for pattern in patterns:
        match = re.search(pattern, time_str)
        if match:
            groups = match.groups()
            if len(groups) == 4:
                # Преобразуем в стандартный формат
                start_hour = groups[0].zfill(2)
                start_min = groups[1] if len(groups[1]) == 2 else groups[1].zfill(2)
                end_hour = groups[2].zfill(2)
                end_min = groups[3] if len(groups[3]) == 2 else groups[3].zfill(2)
                return f"{start_hour}:{start_min}-{end_hour}:{end_min}"
            elif len(groups) == 3:
                start_hour = groups[0].zfill(2)
                end_hour = groups[1].zfill(2)
                end_min = groups[2].zfill(2)
                return f"{start_hour}:00-{end_hour}:{end_min}"
    
    return None


def extract_groups_from_line(line):
    """Извлечение информации о группах из строки"""
    groups_info = {}
    
    # Шаблоны для поиска групп
    group_patterns = [
        r'Группа\s*(\d\.\d)',  # Группа 2.1
        r'гр\.?\s*(\d\.\d)',   # гр.2.1
        r'\b(\d\.\d)\b',       # 2.1
    ]
    
    # Шаблон для поиска предметов с преподавателями и аудиториями
    subject_pattern = r'([А-ЯЁа-яё\s\-]+?)\s*\((л|с)\)\s*([А-ЯЁа-яё\.\s\-]+?)\s*(?:ауд\.?)?\s*(\d+[А-Яа-я]*)'
    
    # Сначала пытаемся найти группы в начале строки
    for pattern in group_patterns:
        matches = re.findall(pattern, line)
        if matches:
            # Если нашли группы, смотрим что идет после них
            for group in matches:
                group_key = f"2.{group.split('.')[-1]}" if '.' in group else f"2.{group}"
                # Ищем предмет после указания группы
                after_group = line.split(group)[-1] if group in line else line
                subject_match = re.search(subject_pattern, after_group)
                if subject_match:
                    subject, lesson_type, teacher, room = subject_match.groups()
                    groups_info[group_key] = {
                        'subject': subject.strip(),
                        'type': lesson_type,
                        'teacher': teacher.strip(),
                        'room': room.strip()
                    }
                else:
                    # Проверяем, нет ли объединенной пары
                    if any(x in line.lower() for x in ['лекция', 'семинар', 'практика']):
                        groups_info[group_key] = {'subject': 'ОБЪЕДИНЕННАЯ ПАРА', 'raw_text': line}
                    else:
                        groups_info[group_key] = None
    
    # Если не нашли группы по шаблонам, проверяем всю строку на наличие информации о парах
    if not groups_info:
        subject_match = re.search(subject_pattern, line)
        if subject_match:
            # Если есть предмет, но не указана группа, возможно это общая пара
            return {'common': {
                'subject': subject_match.group(1).strip(),
                'type': subject_match.group(2),
                'teacher': subject_match.group(3).strip(),
                'room': subject_match.group(4).strip()
            }}
    
    return groups_info


def reconstruct_schedule_table(text, page_num):
    """Восстановление структуры таблицы расписания с правильной разметкой"""
    
    lines = text.split('\n')
    schedule_data = []
    current_day = None
    current_time = None
    time_block_lines = []
    
    # Определяем возможные названия дней недели
    days_of_week = {
        'ПОНЕДЕЛЬНИК': 'понедельник',
        'ВТОРНИК': 'вторник',
        'СРЕДА': 'среда',
        'ЧЕТВЕРГ': 'четверг',
        'ПЯТНИЦА': 'пятница',
        'СУББОТА': 'суббота'
    }
    
    # Определяем группы (из структуры таблицы)
    all_groups = ['2.1', '2.2', '2.3', '2.4', '2.5', '2.6']
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Проверяем, является ли строка указанием дня недели
        day_found = False
        for ru_day, en_day in days_of_week.items():
            if ru_day in line.upper() or en_day in line.lower():
                # Если был предыдущий день с парами, сохраняем его
                if current_day and time_block_lines:
                    schedule_data.append({
                        'day': current_day,
                        'time': 'НЕИЗВЕСТНО',
                        'groups': process_time_block(time_block_lines, all_groups)
                    })
                    time_block_lines = []
                
                current_day = ru_day
                day_found = True
                break
        
        if day_found:
            continue
        
        # Проверяем, содержит ли строка временной интервал
        time_interval = parse_time_interval(line)
        if time_interval:
            # Если был предыдущий временной блок, сохраняем его
            if current_time and time_block_lines:
                schedule_data.append({
                    'day': current_day or 'НЕИЗВЕСТНО',
                    'time': current_time,
                    'groups': process_time_block(time_block_lines, all_groups)
                })
                time_block_lines = []
            
            current_time = time_interval
            # Добавляем саму строку с временем как часть блока
            time_block_lines.append(line)
            continue
        
        # Если строка не день и не время, но у нас есть текущее время, добавляем в блок
        if current_time is not None:
            time_block_lines.append(line)
    
    # Обрабатываем последний блок
    if current_time and time_block_lines:
        schedule_data.append({
            'day': current_day or 'НЕИЗВЕСТНО',
            'time': current_time,
            'groups': process_time_block(time_block_lines, all_groups)
        })
    
    return schedule_data


def process_time_block(lines, all_groups):
    """Обработка блока строк для одного временного интервала"""
    groups_info = {group: None for group in all_groups}
    
    # Объединяем все строки блока
    block_text = ' '.join(lines)
    
    # Удаляем временную информацию из блока для чистого анализа
    time_pattern = r'\d{1,2}[.:]?\d{2}\s*[-–]\s*\d{1,2}[.:]?\d{2}'
    clean_text = re.sub(time_pattern, '', block_text)
    
    # Разделяем по возможным разделителям групп
    # Ищем явные указания групп
    explicit_groups = re.findall(r'Группа\s*(\d\.\d)', clean_text)
    
    if explicit_groups:
        # Если есть явные указания групп, парсим для каждой
        for group in explicit_groups:
            group_key = f"2.{group.split('.')[-1]}" if '.' in group else f"2.{group}"
            # Извлекаем информацию после указания группы
            pattern = rf'Группа\s*{group}(.*?)(?:Группа\s*\d\.\d|$)'
            match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            if match:
                group_text = match.group(1).strip()
                groups_info[group_key] = extract_subject_info(group_text)
    else:
        # Если нет явных указаний групп, проверяем на объединенные пары
        # Ищем общие пары (лекции для всех групп)
        common_patterns = [
            r'([А-ЯЁа-яё\s\-]+)\s*\((л)\)\s*([А-ЯЁа-яё\.\s\-]+)\s*ауд\.?\s*(\d+)',
            r'Лекция\s*по\s*([А-ЯЁа-яё\s\-]+)',
        ]
        
        for pattern in common_patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                # Это общая пара для всех групп
                for group in all_groups:
                    groups_info[group] = {
                        'subject': match.group(1).strip() if len(match.groups()) >= 1 else 'ЛЕКЦИЯ',
                        'type': 'л',
                        'teacher': match.group(3).strip() if len(match.groups()) >= 3 else 'НЕИЗВЕСТНО',
                        'room': match.group(4).strip() if len(match.groups()) >= 4 else 'НЕИЗВЕСТНО',
                        'is_common': True
                    }
                break
        else:
            # Если не нашли общую пару, разбираем по семинарам
            # Ищем семинары с указанием групп
            seminar_pattern = r'([А-ЯЁа-яё\s\-]+)\s*\((с)\)\s*([А-ЯЁа-яё\.\s\-]+)\s*ауд\.?\s*(\d+)'
            seminar_matches = list(re.finditer(seminar_pattern, clean_text, re.IGNORECASE))
            
            if seminar_matches:
                # Распределяем семинары по группам (предполагаем порядок как в all_groups)
                for idx, match in enumerate(seminar_matches[:len(all_groups)]):
                    if idx < len(all_groups):
                        groups_info[all_groups[idx]] = {
                            'subject': match.group(1).strip(),
                            'type': 'с',
                            'teacher': match.group(3).strip(),
                            'room': match.group(4).strip()
                        }
    
    return groups_info


def extract_subject_info(text):
    """Извлечение информации о предмете из текста"""
    if not text or text.strip() == '':
        return None
    
    patterns = [
        r'([А-ЯЁа-яё\s\-]+?)\s*\((л|с)\)\s*([А-ЯЁа-яё\.\s\-]+?)\s*(?:ауд\.?)?\s*(\d+[А-Яа-я]*)',
        r'([А-ЯЁа-яё\s\-]+?)\s*\((л|с)\)\s*([А-ЯЁа-яё\.\s\-]+)',
        r'([А-ЯЁа-яё\s\-]+?)\s*\((л|с)\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            result = {
                'subject': groups[0].strip(),
                'type': groups[1]
            }
            if len(groups) > 2 and groups[2]:
                result['teacher'] = groups[2].strip()
            if len(groups) > 3 and groups[3]:
                result['room'] = groups[3].strip()
            return result
    
    # Если не нашли по шаблону, возвращаем сырой текст
    return {'raw_text': text.strip()}


def format_schedule_for_output(schedule_data):
    """Форматирование расписания для вывода"""
    output_lines = []
    
    for entry in schedule_data:
        day = entry['day']
        time = entry['time']
        groups = entry['groups']
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"📅 {day} | ⏰ {time}")
        output_lines.append(f"{'-'*60}")
        
        for group, info in sorted(groups.items()):
            if info is None:
                output_lines.append(f"  👥 Группа {group}: [НЕТ ПАРЫ]")
            elif isinstance(info, dict):
                if info.get('is_common'):
                    output_lines.append(f"  👥 ВСЕ ГРУППЫ: {info['subject']} ({info['type']}) | 👨‍🏫 {info.get('teacher', '')} | 🏛 {info.get('room', '')}")
                else:
                    subject = info.get('subject', '')
                    lesson_type = info.get('type', '')
                    teacher = info.get('teacher', '')
                    room = info.get('room', '')
                    
                    type_emoji = '📚' if lesson_type == 'л' else '💬'
                    output_lines.append(f"  👥 Группа {group}: {type_emoji} {subject} ({lesson_type}) | 👨‍🏫 {teacher} | 🏛 {room}")
    
    return '\n'.join(output_lines)


def ocr_pdf_to_text(pdf_path: str) -> str:
    """Выполнение OCR на PDF файле"""
    print("🚀 Запуск OCR для расписания...")
    
    # Конвертируем PDF в изображения
    pages = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH,
        dpi=300,
        grayscale=True,
    )
    
    all_schedule_data = []
    page_texts = []
    
    for i, page in enumerate(pages, start=1):
        print(f"📄 Обработка страницы {i}/{len(pages)}...")
        
        # Выполняем OCR
        raw_text = ocr_pil_image(page)
        page_texts.append(f"\n{'='*80}\nСТРАНИЦА {i}\n{'='*80}\n{raw_text}")
        
        # Восстанавливаем структуру расписания
        schedule_data = reconstruct_schedule_table(raw_text, i)
        
        if schedule_data:
            all_schedule_data.extend(schedule_data)
            
            # Форматируем и выводим результат для текущей страницы
            formatted = format_schedule_for_output(schedule_data)
            print(f"\n📋 РАСПИСАНИЕ СО СТРАНИЦЫ {i}:")
            print(formatted)
    
    # Сохраняем полный текст OCR
    full_text = '\n'.join(page_texts)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"\n✅ Полный текст OCR сохранён в: {CACHE_PATH}")
    
    # Форматируем все данные расписания
    final_output = format_schedule_for_output(all_schedule_data)
    
    # Сохраняем структурированное расписание в отдельный файл
    schedule_output_path = "structured_schedule.txt"
    with open(schedule_output_path, "w", encoding="utf-8") as f:
        f.write("🎓 РАСПИСАНИЕ ЗАНЯТИЙ\n")
        f.write("="*60 + "\n\n")
        f.write(final_output)
    
    print(f"✅ Структурированное расписание сохранено в: {schedule_output_path}")
    
    return full_text, final_output


def analyze_schedule_coverage(schedule_text):
    """Анализ покрытия расписания по группам"""
    print("\n📊 АНАЛИЗ РАСПИСАНИЯ:")
    print("-"*40)
    
    # Подсчет пар по группам
    group_stats = defaultdict(lambda: {'lectures': 0, 'seminars': 0, 'empty': 0})
    
    # Ищем записи о группах в отформатированном выводе
    lines = schedule_text.split('\n')
    for line in lines:
        if 'Группа' in line:
            # Извлекаем номер группы
            group_match = re.search(r'Группа\s*(\d\.\d)', line)
            if group_match:
                group = group_match.group(1)
                if '[НЕТ ПАРЫ]' in line:
                    group_stats[group]['empty'] += 1
                elif '(л)' in line:
                    group_stats[group]['lectures'] += 1
                elif '(с)' in line:
                    group_stats[group]['seminars'] += 1
    
    # Выводим статистику
    for group, stats in sorted(group_stats.items()):
        total = stats['lectures'] + stats['seminars'] + stats['empty']
        if total > 0:
            print(f"👥 Группа {group}:")
            print(f"   📚 Лекций: {stats['lectures']}")
            print(f"   💬 Семинаров: {stats['seminars']}")
            print(f"   ⏳ Окон: {stats['empty']}")
            print(f"   📈 Всего пар: {stats['lectures'] + stats['seminars']}")
            print()


if __name__ == "__main__":
    try:
        print("🎓 ОБРАБОТКА РАСПИСАНИЯ УНИВЕРСИТЕТА")
        print("="*60)
        
        # Выполняем OCR и получаем структурированное расписание
        raw_text, structured_schedule = ocr_pdf_to_text(PDF_PATH)
        
        # Анализируем покрытие расписания
        analyze_schedule_coverage(structured_schedule)
        
        print("\n" + "="*60)
        print("✅ Обработка завершена успешно!")
        print(f"📄 Исходный текст OCR: {CACHE_PATH}")
        print(f"📋 Структурированное расписание: structured_schedule.txt")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке: {str(e)}")
        import traceback
        traceback.print_exc()





# промпт для ллм с встроенным окр
'''
Ты - агент-аналитик данных и первоклассный Prompt-инженер, специализирующийся на извлечении структурированной информации и обработке естественного языка (NLP).

Твоя главная задача - принимать на вход пдф файл `vladick.pdf`. 

Ты должен анализировать, очищать текст в строгий, валидный JSON-объект с использованием заданной структуры схемы.

Твоя работа требует высокой точности, умения восстанавливать контекст и исправлять ошибки, опираясь на логику документа и предоставленную схему.

**Входные данные**: Текст будет предоставлен тебе напрямую из файла `vladick.pdf`. Не запрашивай путь к файлу и не придумывай его содержимое.

**Правила:**

    {{
    "schedule": {{
        "Модуль X (даты модуля)": {{
        "День недели": {{
            "Группа X.X": [
            {{
                "time": "чч:мм–чч:мм",
                "subject": "Название дисциплины",
                "type": "л/с (возможна приписка Онлайн)",
                "instructor": "ФИО преподавателя",
                "room": "номер кабинета"
            }}
            ]
        }}
        }}
    }}
    }}

    **ПРАВИЛА**:
    1. Читай таблицу слева направо, сверху вниз
    2. Объединённые пары = дублируй для каждой группы
    3. **Полнота по группам**: Для каждой группы в каждом дне недели создавай запись, даже если пар нет — используй объект с null-параметрами. Пустые ячейки = {"time": null, "subject": null, "type": null, "instructor": null, "room": null}
    4. **Нормализация времени**: 
   - Приводи к формату `чч:мм–чч:мм`
   - Длительность пары = 95 минут (например, `8:00-8:45 8:50-9:35` → `8:00-9:35`)
   - Убирай символы `</sup>`, `^`
   - Соблюдай последовательность: если вторая пара заканчивается в 12:00, третья не может начинаться раньше
    5. ТОЛЬКО ВАЛИДНЫЙ JSON БЕЗ ТЕКСТА!
    6. **Структура пары**: Обязательные поля — time, subject, type (л/с), instructor, room.
    7.**Строгая ориентация на OCR**: Не придумывай группы, предметы, временные периоды или преподавателей, которых нет в тексте.

    **Примеры:**

    Вход: "Понедельник, группа 2.1: 8:00-9:35 Теория управления (л) Ласкова Т.С., ауд. 118"
    Выход:
    {{
    "schedule": {{
        "Модуль 1": {{
        "Понедельник": {{
            "Группа 2.1": [
            {{
                "time": "8:00–9:35",
                "subject": "Теория управления",
                "type": "л",
                "instructor": "Ласкова Т.С.",
                "room": "118"
            }}
            ]
        }}
        }}
    }}
    }}

    Вход: "Среда, группа 2.4: 8:00-8:45 Теория управления (л) Ласкова Т.С., ауд. 423"
    Выход:
    {{
    "schedule": {{
        "Модуль 1": {{
        "Среда": {{
            "Группа 2.4": [
            {{
                "time": "8:00–9:35",
                "subject": "Теория управления",
                "type": "л",
                "instructor": "Ласкова Т.С.",
                "room": "423"
            }}
            ]
        }}
        }}
    }}
    }}

    Страница расписания:'''





# pdf_vision_ocr.py
import ollama
from pdf2image import convert_from_path
import json
import os
import re
from pathlib import Path

# === НАСТРОЙКИ ===
PDF_PATH = r"vladick.pdf"  # Путь к твоему PDF
POPPLER_PATH = r"C:\Users\sanya\AppData\Roaming\Python\Python313\Scripts\poppler-25.11.0\Library\bin"
TEMP_DIR = "temp_pages"
MODEL = "llava-phi3:3.8b" # ← Правильное имя
  # или "llama3.2-vision:1b" для слабого ПК

def pdf_to_temp_images(pdf_path: str):
    """Конвертирует PDF во временные PNG файлы."""
    print("🔄 Конвертирую PDF в изображения...")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=100)
    temp_paths = []
    
    for i, page in enumerate(pages, 1):
        temp_path = os.path.join(TEMP_DIR, f"page_{i:03d}.png")
        page.save(temp_path, "PNG")
        temp_paths.append(temp_path)
        print(f"  📄 Создана страница {i}/{len(pages)}")
    
    return temp_paths

def extract_schedule(page_path: str, page_num: int) -> dict:
    """Извлекает расписание с одной страницы с улучшенным промптом и парсингом."""
    response = ollama.chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': f'''ИЗВЛЕКИ РАСПИСАНИЕ ИЗ ТАБЛИЦЫ на изображении в JSON.

ПРАВИЛА:
1. Читай таблицу слева направо, сверху вниз
2. Группы: 2.1, 2.2, 2.3, 2.4
3. Дни: Понедельник, Вторник, Среда, Четверг, Пятница
4. Время: "8:00-9:35" (95 мин пары)
5. Формат пары: {{"time": "8:00-9:35", "subject": "дисциплина", "type": "л/с", "instructor": "ФИО", "room": "ауд."}}
6. Пустые ячейки: {{"time": null, "subject": null, "type": null, "instructor": null, "room": null}}

ТОЛЬКО JSON! Без текста, без ```, без объяснений!

ОЖИДАЕМЫЙ ФОРМАТ:
{{
  "schedule": {{
    "Модуль 1 (даты)": {{
      "Понедельник": {{
        "2.1": [{{"time": "8:00-9:35", "subject": "Матанализ", "type": "л", "instructor": "Иванов И.И.", "room": "118"}}]
      }},
      "Вторник": {{...}}
    }}
  }}
}}

Страница {page_num}:''',
            'images': [page_path]
        }
    ])
    
    # Очистка ответа
    content = response['message']['content'].strip()
    content = re.sub(r'```(?:json|css)?\s*', '', content)  # Убираем markdown
    content = content.strip()
    
    try:
        # Находим самый большой JSON блок
        json_match = re.search(r'\{[^{}]*"schedule"[^{}]*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed
        
        # Fallback: первый валидный JSON
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            parsed = json.loads(json_str)
            return parsed
            
        raise ValueError("JSON не найден")
        
    except json.JSONDecodeError as e:
        return {
            "page": page_num, 
            "error": f"JSON ошибка: {str(e)[:50]}", 
            "raw": content[:300]
        }
    except Exception as e:
        return {
            "page": page_num, 
            "error": "Парсинг", 
            "raw": content[:300]
        }

def main():
    # Проверка модели
    try:
        ollama.show(MODEL)
        print(f"✅ Модель {MODEL} готова")
    except:
        print(f"❌ Модель {MODEL} не найдена!")
        print(f"Запусти: ollama pull {MODEL}")
        return
    
    # Конвертация PDF
    page_paths = pdf_to_temp_images(PDF_PATH)
    
    # Обработка страниц
    full_schedule = {"pages": {}, "total_pages": len(page_paths)}
    
    for i, page_path in enumerate(page_paths, 1):
        print(f"\n📄 Обрабатываю страницу {i}/{len(page_paths)}...")
        schedule_page = extract_schedule(page_path, i)
        full_schedule["pages"][f"page_{i}"] = schedule_page
    
    # Сохранение
    output_file = "schedule_vision.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_schedule, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ ✅ РАСПИСАНИЕ СОХРАНЕНО: {output_file}")
    print(f"📊 Всего страниц: {len(page_paths)}")
    
    # Показываем превью
    print("\n📋 ПРЕВЬЮ ПЕРВОЙ СТРАНИЦЫ:")
    if "page_1" in full_schedule["pages"]:
        print(json.dumps(full_schedule["pages"]["page_1"], ensure_ascii=False, indent=2)[:1000])
    
    # Очистка временных файлов
    import shutil
    shutil.rmtree(TEMP_DIR)
    print("🧹 Временные файлы удалены")

if __name__ == "__main__":
    main()





# ocr_lighton new version working
import torch
from PIL import Image, ImageOps
from pdf2image import convert_from_path
from transformers import AutoProcessor, LightOnOCRForConditionalGeneration


# ================= НАСТРОЙКИ =================
PDF_PATH = "filevlad.pdf"
POPPLER_PATH = r"C:\Users\sanya\AppData\Roaming\Python\Python313\Scripts\poppler-25.11.0\Library\bin"

OCR_MODEL = "lightonai/LightOnOCR-1B-1025"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    return img.convert("RGB")


# ================= OCR PAGE =================
def ocr_page(page: Image.Image) -> str:
    page = normalize_page(page)

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
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    all_pages = []
    for i, page in enumerate(pages, start=1):
        print(f"📄 OCR страница {i}/{len(pages)}")
        text = ocr_page(page)
        all_pages.append(f"=== СТРАНИЦА {i} ===\n{text}")

    return "\n\n".join(all_pages)


# ================= MAIN =================
if __name__ == "__main__":
    text = ocr_pdf(PDF_PATH)

    with open("ocr_result.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("✅ OCR завершён")





# ocr_lighton.py from gpt 5.2
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
