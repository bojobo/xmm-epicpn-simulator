from pathlib import Path

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def download_file(url: str, out_path: Path) -> Path:
    retry_strategy = Retry(total=10, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = Session()
    prefix = "https://" if url.startswith("https://") else "http://"
    session.mount(prefix, adapter)

    with session.get(url, stream=True) as response, open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=int(1e6)):
            f.write(chunk)

    return out_path
