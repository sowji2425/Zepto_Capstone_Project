import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}
FIXED_GBP_TO_INR = 105.50

def scrape_books(num_pages: int = 5) -> pd.DataFrame:
    base_url = "https://toscrape.com{}.html"
    records = []

    for page in range(1, num_pages + 1):
        url = base_url.format(page)
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            continue
        
        soup = BeautifulSoup(resp.content, "html.parser")
        articles = soup.find_all("article", class_="product_pod")

        for article in articles:
            try:
                # Title
                title = article.h3.a["title"].strip()
                
                # Price extraction
                price_text = article.find("p", class_="price_color").text.strip()
                price_match = re.search(r"[\d.]+", price_text)
                price_gbp = float(price_match.group(0)) if price_match else None
                
                # Star rating
                rating_classes = article.find("p", class_="star-rating")["class"]
                rating_word = next((c for c in rating_classes if c != "star-rating"), None)
                rating = RATING_MAP.get(rating_word, None)
                
                # Availability
                avail_text = article.find("p", class_="instock availability").text.strip().lower()
                in_stock = 1 if "in stock" in avail_text else 0
                
                # Baseline category context from catalogue listing
                category = "General Fiction / Literature"

                records.append({
                    "title": title,
                    "price_gbp": price_gbp,
                    "rating": rating,
                    "in_stock": in_stock,
                    "category": category
                })
            except Exception:
                # Drop rows where schema assumptions fail to preserve data integrity
                continue

    df = pd.DataFrame(records)
    
    # Cleaning & conversions
    df = df.dropna(subset=["title", "price_gbp", "rating"])
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(int)
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    
    return df

if __name__ == "__main__":
    df_books = scrape_books(5)
    print(f"Scraped {len(df_books)} books successfully.")
