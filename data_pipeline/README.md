# Data Pipeline

Run from the repository root:

```powershell
python data_pipeline\build_pipeline.py
```

The script uses `requests` and `BeautifulSoup` to scrape all paginated listing pages in the Travel, Mystery, and Historical Fiction categories from Books to Scrape. It produces at least 60 books across three categories without manual copy-pasting; the row count is printed after each run. The full catalogue category is intentionally excluded because it duplicates books from the named categories and contains roughly 1,000 pages of listings.

Cleaning decisions:

- Prices are parsed as floating-point GBP values.
- Star ratings One through Five become integers 1 through 5.
- Availability becomes an integer SQLite boolean (`1` for in stock, `0` for out of stock).
- Failed numeric parsing uses median imputation; failed availability uses the mode.
- `price_inr = price_gbp * 105.50`, using the required fixed rate of 1 GBP = 105.50 INR.

The normalized SQLite database contains `categories(category_id, category_name)` and `books(book_id, ..., category_id)` with a foreign-key relationship. The script executes SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN, and multiple JOIN queries. It reads SQL results with `pd.read_sql()` and independently reproduces the final JOIN with `pd.merge()`; the run prints `Equivalent output: True`.
