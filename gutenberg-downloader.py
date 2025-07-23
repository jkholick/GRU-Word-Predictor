######################################################
# This Program is to slowly download gutenberg books #
######################################################
import os
import requests
import time

# Directory to save books
save_dir = "gutenberg_books"
os.makedirs(save_dir, exist_ok=True)

def sanitize_filename(name):
    # Remove/replace characters that can't be in filenames
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in name).strip()

def download_text_file(url, filepath):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(r.content)
        print(f"Saved: {filepath}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def fetch_and_download_all_english_books():
    base_url = "https://gutendex.com/books"
    page = 1
    total_downloaded = 0
    
    while True:
        print(f"Fetching page {page} ...")
        try:
            resp = requests.get(base_url, params={"languages": "en", "page": page}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
        
        books = data.get("results", [])
        if not books:
            print("No more books found, ending.")
            break
        
        for book in books:
            title = book.get("title", "unknown_title")
            sanitized_title = sanitize_filename(title)
            gutenberg_id = book.get("id")
            
            # Find plain text utf-8 format URL
            formats = book.get("formats", {})
            text_url = None
            
            # Common keys for plain text utf-8:
            candidates = [
                "text/plain; charset=utf-8",
                "text/plain; charset=us-ascii",
                "text/plain",
            ]
            
            for key in candidates:
                if key in formats:
                    text_url = formats[key]
                    break
            
            if not text_url:
                print(f"No plain text file found for book {title} (ID: {gutenberg_id})")
                continue
            
            # Create filename: <ID>_<title>.txt
            filename = f"{gutenberg_id}_{sanitized_title}.txt"
            filepath = os.path.join(save_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Already downloaded: {filename}")
                continue
            
            print(f"Downloading book: {title} (ID: {gutenberg_id})")
            success = download_text_file(text_url, filepath)
            if success:
                total_downloaded += 1
            
            # Be polite: wait a bit between downloads
            time.sleep(1)
        
        # Next page
        if data.get("next") is None:
            print("No next page. Finished downloading all books.")
            break
        page += 1
    
    print(f"Total books downloaded: {total_downloaded}")

if __name__ == "__main__":
    fetch_and_download_all_english_books()

