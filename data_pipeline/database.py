import sqlite3
import pandas as pd
from scraper import scrape_books

DB_NAME = "data_pipeline/zepto_catalog.db"

def init_and_load_db(df: pd.DataFrame):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Normalized schema
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
    """)

    # Populate categories
    categories = df["category"].drop_duplicates().tolist()
    for cat in categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()

    # Map category names to IDs
    category_map = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
    cat_to_id = dict(zip(category_map["category_name"], category_map["category_id"]))
    df["category_id"] = df["category"].map(cat_to_id)

    # Insert books
    books_data = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].to_dict(orient="records")
    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (:title, :price_gbp, :price_inr, :rating, :in_stock, :category_id)
    """, books_data)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    df = scrape_books(5)
    init_and_load_db(df)
    print("Database initialized and populated.")
