import requests
from bs4 import BeautifulSoup
import re

def scrape_webpage():
    url = "https://en.wikipedia.org/wiki/Punk_rock"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to fetch the webpage. Status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", class_="mw-parser-output")

        if not content_div:
            print("❌ Could not find the main content container.")
            return None

        paragraphs = content_div.find_all("p")
        if not paragraphs:
            print("⚠️ No <p> tags found inside the content container.")
            return None

        cleaned_paragraphs = []

        # Smart citation cleaner: removes [1], [nb 3], [6, 2], [ ... ], [clarification needed], etc.
        citation_pattern = re.compile(
            r'\s*\[\s*(?:(?:\d+|\d+(?:\s*,\s*\d+)+)|(?:nb|note)?\s*\w+|[. …,]+|citation needed|clarification needed)\s*\]\s*',
            flags=re.IGNORECASE
        )

        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if text:
                cleaned = citation_pattern.sub(' ', text)
                cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # collapse multiple spaces
                cleaned_paragraphs.append(cleaned.strip())

        article_text = "\n\n".join(cleaned_paragraphs)

        with open("Selected_Document.txt", "w", encoding="utf-8") as file:
            file.write(article_text)

        print("✅ Successfully fetched, cleaned, and formatted article.")
        return article_text

    except Exception as e:
        print(f"⚠️ An error occurred: {e}")
        return None

def main():
    scrape_webpage()

if __name__ == '__main__':
    main()
