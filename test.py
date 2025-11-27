import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin,urlparse
from os.path import splitext
import shutil


# url = "https://sfedu.ru/www/stat_pages22.show?p=STD/rasp/D"

# responce = requests.get(url)
# responce.raise_for_status()

# soup = BeautifulSoup(responce.text, 'html.parser')

# select = soup.find("select", attrs= {"name": "p_tf_id"})
# values = [opt.get("value") for opt in select.find_all("option") if opt.get("value")]
# print(values)


url = "https://sfedu.ru/www/stat_pages22.show?p=STD/rasp/D"
resp = requests.get(url)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

select = soup.find("select", attrs={"name": "p_es_id"})

values = [opt["value"] for opt in select.find_all("option")]
print(values)
















# url = 'https://sfedu.ru/www/stat_pages22.show?p=STD/rasp/D&params=(p_es_id=%3E10000000000000,p_tf_id=%3E1)'

# responce = requests.get(url)
# responce.raise_for_status()

# soup = BeautifulSoup(responce.text, 'html.parser')

# dir = "schedules"

# if os.path.exists(dir):
#     shutil.rmtree(dir)
# os.makedirs(dir, exist_ok=True)


# for a in soup.find_all('a',string = "скачать"):
#     file_url = urljoin(url, a.get('href'))
#     # filename = urlparse(file_url).path.split('/')[-1]
#     # path = os.path.join(dir, filename)

#     r = requests.get(file_url, stream = True)
#     r.raise_for_status()

#     # path = urlparse(r.url).path
#     # name = os.path.basename(path)          
#     # root, ext = splitext(name)
#     # # if not ext:                        
#     # #     ext = ".pdf" 
#     # filename = (root or "schedule") + ext

#     cd = r.headers.get("Content-Disposition", "")
#     filename = None
#     if "filename=" in cd:
#         filename = cd.split("filename=")[1].strip('"; ')
#         try:
#             filename = filename.encode('latin1').decode('cp1251')
#         except UnicodeDecodeError:
#             filename = filename

#     # if not filename:
#     #     path = urlparse(r.url).path
#     #     name = os.path.basename(path)
#     #     root, ext = splitext(name)
#     #     filename = (root or "schedule") + (ext or "")

#     full_path = os.path.join(dir, filename)
#     with open(full_path, "wb") as file:
#         for chunk in r.iter_content(chunk_size = 8192):
#             if chunk:
#                 file.write(chunk)
    
#     print(f"Скачан файл: {filename}")