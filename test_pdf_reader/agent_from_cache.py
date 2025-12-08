# agent_from_cache.py

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from prompts import SCHEDULE_AGENT_PROMPT

OCR_CACHE_FILE = "vladick_ocr_output.txt"

os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free", 
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=15000,
)


@tool
def get_schedule_from_cache() -> str:
    """Инструмент: вернуть текст расписания из кеш-файла OCR."""
    with open(OCR_CACHE_FILE, "r", encoding="utf-8") as f:
        return f.read()


# Вызываем инструмент
ocr_text = get_schedule_from_cache.invoke({})
print(f"📄 Длина текста OCR: {len(ocr_text)} символов\n")

# Формируем сообщения
messages = [
    SystemMessage(content=SCHEDULE_AGENT_PROMPT),
    HumanMessage(content=f"Вот текст расписания:\n\n{ocr_text}\n\nВерни расписание как JSON.")
]

# Вызов LLM
print("🤖 Отправляю запрос к LLM...\n")
result = llm.invoke(messages)
output_text = result.content

print("📥 Ответ от LLM:")
print(output_text)
print(f"\n📏 Длина ответа: {len(output_text)} символов\n")

# Сохранение
if output_text.strip():
    with open("schedule_output.json", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("✅ Результат сохранён в schedule_output.json")
else:
    print("❌ Ответ пустой, файл не записан")
