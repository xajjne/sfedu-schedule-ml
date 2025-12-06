import os

from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

from ocr_lighton import ocr_pdf_to_text, PDF_PATH  
from prompts import SCHEDULE_AGENT_PROMPT

llm = ChatOllama(
    model="llama3.2:3b",          
    base_url="http://localhost:11434",
    temperature=0.2,
)

def get_schedule_from_pdf(pdf_path: str) -> str:
    return ocr_pdf_to_text(PDF_PATH)


agent = create_react_agent(
    model=llm,
    tools=[get_schedule_from_pdf],
    prompt=SCHEDULE_AGENT_PROMPT,
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Прочитай файл с расписанием, используя инструмент "
                        "get_schedule_from_pdf, и верни расписание как JSON."
                    ),
                }
            ]
        },
    )
    print(result["messages"][-1]["content"])
