import os
import requests

path = r'D:\Enterprise RAG\Think Like an Entrepreneur, Act Like a CEO ( PDFDrive ).pdf'
print('exists', os.path.exists(path))
with open(path, 'rb') as f:
    r = requests.post(
        'http://localhost:8000/v1/ingest',
        files={'file': (os.path.basename(path), f, 'application/pdf')},
        timeout=600,
    )
print('status', r.status_code)
print(r.text)
