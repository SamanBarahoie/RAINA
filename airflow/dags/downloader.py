import os
import re
import json
import mimetypes
import requests
from urllib.parse import urljoin, unquote
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

class USBDownloaderMultipleFilters:
    def __init__(
        self,
        download_dir="data",
        track_file="data.json",
        max_retries=2,
        max_workers=10,
        pagesize=50,
        category_filters=None
    ):
        self.download_dir = os.path.abspath(download_dir)
        self.track_file = track_file
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.pagesize = pagesize
        self.category_filters = category_filters or ["آیین نامه ها", "فرایندها"]

        os.makedirs(self.download_dir, exist_ok=True)

        if os.path.exists(self.track_file):
            with open(self.track_file, "r", encoding="utf-8") as f:
                self.attachments_data = json.load(f)
        else:
            self.attachments_data = []

        self.downloaded_titles = {d["title"] for d in self.attachments_data}
        self.failed_downloads = []

        self.api_base = "https://www.usb.ac.ir/DesktopModules/DnnSharp/ActionGrid/Api.ashx?method=GetData"

    @staticmethod
    def sanitize_filename(name):
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    def download_file(self, title, file_url):
        parsed = requests.utils.urlparse(file_url)
        ext = os.path.splitext(unquote(parsed.path))[1].lower() or ".bin"

        ext_folder = os.path.join(self.download_dir, ext.lstrip("."))
        os.makedirs(ext_folder, exist_ok=True)

        file_path = os.path.join(ext_folder, self.sanitize_filename(title) + ext)
        if os.path.exists(file_path):
            if title not in self.downloaded_titles:
                self.attachments_data.append({"title": title, "url": file_url})
                self.downloaded_titles.add(title)
            return True

        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(file_url, timeout=30)
                r.raise_for_status()
                if not r.content:
                    raise ValueError("Empty content")
                with open(file_path, "wb") as f:
                    f.write(r.content)
                self.attachments_data.append({"title": title, "url": file_url})
                self.downloaded_titles.add(title)
                print(f"Downloaded: {file_path}")
                return True
            except Exception as e:
                print(f"Download failed ({attempt}/{self.max_retries}) for {title}: {e}")

        print(f"Failed to download: {title}")
        self.failed_downloads.append({"title": title, "url": file_url})
        return False

    def extract_detail_url(self, formatted_value):
        soup = BeautifulSoup(formatted_value, "html.parser")
        a = soup.find("a")
        return urljoin("https://www.usb.ac.ir", a["href"]) if a else None

    def fetch_page(self, page, category_filter):
        params = {
            "page14072": page,
            "size14072": self.pagesize,
            "TabId": 7634,
            "language": "fa-IR",
            "_alias": "www.usb.ac.ir",
            "_mid": 14072,
            "_tabid": 7634,
            "_url": f"https://www.usb.ac.ir/academics/process?page14072={page}&size14072={self.pagesize}&filter14072-FarayandNo1Name={category_filter}",
            "referrer": "",
            "timezone": 210,
            "ViewMode": "Legacy",
            "filter14072-FarayandNo1Name": category_filter,
            "page": page,
            "pagesize": self.pagesize,
            "search": "",
            "sort": "",
            "sortAsc": "true"
        }
        try:
            r = requests.get(self.api_base, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            return data.get("results", []), data.get("totalPages", 0)
        except Exception as e:
            print(f"Failed to fetch page {page} for {category_filter}: {e}")
            return [], 0

    def process_entry(self, entry, category_filter):
        fields = {f["Title"]: f["Value"] for f in entry.get("fields", [])}
        title = fields.get("عنوان")
        category = fields.get("نوع سند")

        if not title or title in self.downloaded_titles:
            return
        if category != category_filter:
            return

        detail_html = entry["fields"][-1]["FormattedValue"]
        detail_url = self.extract_detail_url(detail_html)
        if not detail_url:
            return

        try:
            r = requests.get(detail_url, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            attachments = [urljoin("https://www.usb.ac.ir", a["href"])
                           for a in soup.select("a.Normal")]
            if attachments:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for url in attachments:
                        executor.submit(self.download_file, title, url)
        except Exception as e:
            print(f"Failed to fetch detail page {detail_url}: {e}")

    def run_category(self, category_filter):
        page = 1
        total_pages = None
        while True:
            results, total_pages_api = self.fetch_page(page, category_filter)
            if total_pages is None:
                total_pages = total_pages_api
                print(f"Total pages to fetch for '{category_filter}': {total_pages}")

            if not results or page > total_pages:
                break

            print(f"Processing page {page}/{total_pages} for '{category_filter}'")
            for entry in results:
                self.process_entry(entry, category_filter)

            page += 1

    def run(self):
        for category in self.category_filters:
            self.run_category(category)

        # Save metadata
        with open(self.track_file, "w", encoding="utf-8") as f:
            json.dump(self.attachments_data, f, ensure_ascii=False, indent=2)

        # Retry failed downloads
        if self.failed_downloads:
            print("Retrying failed downloads...")
            for item in self.failed_downloads:
                self.download_file(item["title"], item["url"])
            with open(self.track_file, "w", encoding="utf-8") as f:
                json.dump(self.attachments_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    downloader = USBDownloaderMultipleFilters(
        category_filters=["فرآیندها"]
    )
    downloader.run()
