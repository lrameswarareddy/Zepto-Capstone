import re
import sqlite3
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

CONVERSION_RATE = 105.50


def fetch_page(url: str):
    """Fetch a page with browser-like headers to avoid 403 blocks from the target website."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://books.toscrape.com/",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


def fetch_category_urls():
    """Return category URLs from the site's navigation."""
    response = fetch_page("https://books.toscrape.com/")
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for a in soup.select("ul.nav-list li a"):
        href = a.get("href")
        if href and "category" in href:
            links.append(urljoin(response.url, href))
    return list(dict.fromkeys(links))


def parse_price(value):
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    return float(match.group())


def parse_rating(star_class):
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    if not star_class:
        return None
    return mapping.get(str(star_class).strip(), None)


def parse_availability(value):
    if not value:
        return None
    text = str(value).strip().lower()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None


def scrape_books():
    rows = []
    seen_titles = set()
    categories = fetch_category_urls()[1:4]
    for category_url in categories:
        page_url = category_url
        visited_pages = set()
        while page_url and page_url not in visited_pages:
            visited_pages.add(page_url)
            response = fetch_page(page_url)
            soup = BeautifulSoup(response.text, "html.parser")
            category_name = soup.select_one("h1").get_text(strip=True) if soup.select_one("h1") else "Unknown"
            for item in soup.select("article.product_pod"):
                title = item.select_one("h3 a").get("title", "").strip() if item.select_one("h3 a") else ""
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                price_tag = item.select_one("p.price_color")
                rating_tag = item.select_one("p.star-rating")
                stock_tag = item.select_one("p.instock.availability")
                star_classes = rating_tag.get("class", []) if rating_tag else []
                rating_value = parse_rating(star_classes[-1] if star_classes else "")
                rows.append(
                    {
                        "title": title,
                        "price_gbp": parse_price(price_tag.get_text(strip=True) if price_tag else None),
                        "rating": rating_value,
                        "in_stock": parse_availability(stock_tag.get_text(strip=True) if stock_tag else None),
                        "category": category_name,
                    }
                )
            next_link = soup.select_one("li.next a")
            page_url = urljoin(response.url, next_link.get("href")) if next_link else None
    if len(rows) < 60:
        raise ValueError(f"Expected at least 60 books, but scraped {len(rows)}.")
    df = pd.DataFrame(rows)
    return df


def clean_books(df):
    df = df.copy()
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["in_stock"] = df["in_stock"].astype("boolean")

    median_price = df["price_gbp"].median()
    median_rating = df["rating"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)
    df["rating"] = df["rating"].fillna(median_rating)
    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].fillna(bool(df["in_stock"].mode().iloc[0])).astype(bool)
    df["price_inr"] = (df["price_gbp"] * CONVERSION_RATE).round(2)
    df["category"] = df["category"].fillna("Unknown")
    return df


def create_database(db_path: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")
    cursor.execute("CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)")
    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        )
        """
    )

    cleaned_df = clean_books(scrape_books())
    unique_cats = sorted(cleaned_df["category"].dropna().unique().tolist())
    mapping = {}
    for idx, category_name in enumerate(unique_cats, start=1):
        cursor.execute("INSERT INTO categories (category_id, category_name) VALUES (?, ?)", (idx, category_name))
        mapping[category_name] = idx

    for _, row in cleaned_df.iterrows():
        cursor.execute(
            "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?, ?, ?, ?, ?, ?)",
            (row["title"], float(row["price_gbp"]), float(row["price_inr"]), int(row["rating"]), int(bool(row["in_stock"])), mapping[row["category"]]),
        )

    conn.commit()
    conn.close()
    return cleaned_df


def show_queries(db_path: Path):
    conn = sqlite3.connect(db_path)
    queries = [
        ("SELECT * FROM books LIMIT 5", "SELECT query"),
        ("SELECT title, price_gbp FROM books WHERE in_stock = 1 ORDER BY price_gbp DESC LIMIT 5", "WHERE + ORDER BY + LIMIT"),
        ("SELECT DISTINCT rating FROM books ORDER BY rating ASC", "DISTINCT"),
        ("SELECT title, price_inr FROM books WHERE price_inr BETWEEN 2000 AND 5000 ORDER BY price_inr LIMIT 10", "BETWEEN"),
        ("SELECT title, category_id FROM books WHERE category_id IN (1, 2) ORDER BY title LIMIT 10", "IN"),
        ("SELECT c.category_name, COUNT(*) AS book_count FROM categories c JOIN books b ON b.category_id = c.category_id GROUP BY c.category_name ORDER BY book_count DESC", "JOIN + GROUP BY"),
        ("SELECT c.category_name, AVG(b.price_inr) AS average_price_inr FROM categories c JOIN books b ON b.category_id = c.category_id GROUP BY c.category_name ORDER BY average_price_inr DESC", "JOIN + AVG"),
        (
            "SELECT b.title, c.category_name, b.rating FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC, b.title ASC LIMIT 10",
            "JOIN + ORDER BY",
        ),
    ]

    print("\nSQL query outputs:\n")
    for query, label in queries:
        print(f"--- {label} ---")
        print(query)
        print(pd.read_sql(query, conn))
        print()

    join_query = queries[-1][0]
    sql_join = pd.read_sql(join_query, conn)
    books_df = pd.read_sql("SELECT book_id, title, price_gbp, price_inr, rating, in_stock, category_id FROM books", conn)
    category_df = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
    merge_df = books_df.merge(category_df, on="category_id", how="left")
    merged_join = merge_df[["title", "category_name", "rating"]].sort_values(["rating", "title"], ascending=[False, True]).head(10).reset_index(drop=True)
    sql_join = sql_join.reset_index(drop=True)
    print("\n--- pandas.merge verification ---")
    print("SQL join result:\n", sql_join)
    print("\nPandas merge result:\n", merged_join)
    print("\nEquivalent output:", sql_join.equals(merged_join.rename(columns={"category_name": "category_name", "rating": "rating"}).reset_index(drop=True)))
    conn.close()


def main():
    db_path = Path(__file__).resolve().parent / "books.db"
    cleaned = create_database(db_path)
    print(f"Scraped and stored {len(cleaned)} cleaned rows in {db_path}")
    show_queries(db_path)


if __name__ == "__main__":
    main()
