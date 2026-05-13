import io
from pathlib import Path
from urllib.parse import urlparse

import requests


class OpenOrDownload:
    def __init__(self, path_or_url: str):
        self.path_or_url = path_or_url
        self.file = None
        self.is_url = urlparse(path_or_url).scheme in ['http', 'https']

    def __enter__(self):
        if self.is_url:
            # URL: Download the file
            response = requests.get(self.path_or_url, stream=True)
            response.raise_for_status()
            self.file = io.BytesIO(response.content)
        else:
            # Local Path: Open the file
            file_path = Path(self.path_or_url)
            if file_path.exists() and file_path.is_file():
                self.file = open(file_path)
            else:
                raise FileNotFoundError(f"File not found: {self.path_or_url}")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
