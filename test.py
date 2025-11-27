import re
import os
import requests
import shutil
from bs4 import BeautifulSoup
from urllib.parse import urljoin,urlparse
from os.path import splitext


url = "https://sfedu.ru/www/stat_pages22.show?p=STD/rasp/D"
resp = requests.get(url)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

select = soup.find("select", attrs={"name": "p_es_id"})

values = [opt["value"] for opt in select.find_all("option") if opt.get("value")]

dir = "schedules"

if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir, exist_ok=True)

for value in values:
    url = 'https://sfedu.ru/www/stat_pages22.show?p=STD/rasp/D&params=(p_es_id=%3E'+value+',p_tf_id=%3E1)'
    responce = requests.get(url)
    responce.raise_for_status()

    soup = BeautifulSoup(responce.text, 'html.parser')

    raw_name = soup.find("option", attrs={"value": value}).text
    dir_name = re.sub(r'[<>:"/\\|?*]', '_', raw_name)

    folder_path = os.path.join(dir, dir_name)
    os.makedirs(folder_path, exist_ok=True)

    for a in soup.find_all('a',string = "скачать"):
        file_url = urljoin(url, a.get('href'))

        r = requests.get(file_url, stream = True)
        r.raise_for_status()

        cd = r.headers.get("Content-Disposition", "")
        filename = None
        if "filename=" in cd:
            filename = cd.split("filename=")[1].strip('"; ')
            try:
                filename = filename.encode('latin1').decode('cp1251')
            except UnicodeDecodeError:
                filename = filename

        full_path = os.path.join(folder_path, filename)
        with open(full_path, "wb") as file:
            for chunk in r.iter_content(chunk_size = 8192):
                if chunk:
                    file.write(chunk)
        
        print(f"Скачан файл {filename} в папку {dir_name}")
        



