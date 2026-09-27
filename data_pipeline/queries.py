import sqlite3
import pandas as pd

DB_NAME = "data_pipeline/zepto_catalog.db"

QUERIES = {
    "q1_select_where_limit": """
        -- SELECT / WHERE / LIMIT: Find 5 available books
        SELECT title, price_inr, rating 
        FROM books 
        WHERE in_stock = 1 
        LIMIT 5;
    """,
    "q2_order_by": """
        -- ORDER BY: 10 highest-priced items in INR
        SELECT title, price_inr, rating 
        FROM books 
        ORDER BY price_inr DESC 
        LIMIT 10;
    """,
    "q3_distinct": """
        -- DISTINCT: All unique rating levels present in store
        SELECT DISTINCT rating 
        FROM books 
        ORDER BY rating ASC;
    """,
    "q4_between": """
        -- BETWEEN / IN: Books priced between 2000 and 4000 INR with top ratings
        SELECT title, price_inr, rating 
        FROM books 
        WHERE price_inr BETWEEN 2000 AND 4000 
          AND rating IN (4, 5);
    """,
    "q5_join": """
        -- INNER JOIN: Books linked with their normalized category description
        SELECT b.book_id, b.title, b.price_inr, b.rating, c.category_name 
        FROM books b
        INNER JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_inr ASC
        LIMIT 10;
    """
}

def execute_queries():
    conn = sqlite3.connect(DB_NAME)
    
    for name, sql in QUERIES.items():
        print(f"--- Running {name} ---")
        df_result = pd.read_sql_query(sql, conn)
        print(df_result)
        print("\n")

    # Read back at least two results into pandas DataFrames explicitly
    df_top_priced = pd.read_sql_query(QUERIES["q2_order_by"], conn)
    df_joined_view = pd.read_sql_query(QUERIES["q5_join"], conn)
    
    conn.close()
    return df_top_priced, df_joined_view

if __name__ == "__main__":
    execute_queries()
