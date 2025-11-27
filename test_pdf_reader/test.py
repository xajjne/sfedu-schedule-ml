import os
import camelot           
import pdfplumber        

PDF_PATH = "vladic.pdf"

# --- Вариант 1: Camelot ---
# print("=== CAMEL0T (lattice) ===")
# if os.path.exists(PDF_PATH):
#     tables = camelot.read_pdf(PDF_PATH, pages="all", flavor="lattice")  # если не поймает линии, попробуй "stream"
#     for i, t in enumerate(tables, start=1):
#         print(f"\n--- Таблица {i} ---")
#         # t.df — pandas.DataFrame; to_string просто красиво печатает её
#         print(t.df.to_string(index=False))
# else:
#     print(f"Файл {PDF_PATH} не найден")


# --- Вариант 2: pdfplumber с настройками ---
print("\n\n=== PDFPLUMBER (tuned) ===")
table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
}

with pdfplumber.open(PDF_PATH) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table(table_settings)
        print(f"\n--- Страница {page_num} ---")
        if not table:
            print("Таблица не найдена")
            continue

        for row in table:
            # чистим ячейки: убираем лишние переводы строк и пробелы
            cleaned = [" ".join((cell or "").split()) for cell in row]
            print(" | ".join(cleaned))
