import polars as pl
import pandas as pd
from sqlalchemy import Engine, create_engine
import json

def create_dataframe_pl(engine: Engine, query: str):
    try:
        with engine.connect() as conn:
            df = pl.read_database(query=query, connection=conn)
            return json.dumps(df.write_json())
    except Exception as e:
        return json.dumps({"error": f"Error processing query results: {str(e)}", "data": []})

# db_engine = create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment')

# print(create_dataframe_pl(db_engine, "SELECT ar.name AS artist_name, COUNT(DISTINCT al.album_id) AS album_count FROM artist ar JOIN album al ON ar.artist_id = al.artist_id JOIN track t ON al.album_id = t.album_id JOIN genre g ON t.genre_id = g.genre_id WHERE g.name = 'Metal' GROUP BY ar.name;;"))

def create_dataframe_pd_json(engine: Engine, query: str):
    try:
        # The engine.connect() establishes a connection from the pool
        with engine.connect() as conn:
            # pd.read_sql executes the query against the connection
            # and returns the results as a Pandas DataFrame.
            df = pd.read_sql(sql=query, con=conn)
            return df.to_json()
    except Exception as e:
        # It's good practice to handle potential errors
        # and return a consistent error format.
        return json.dumps({"error": f"Error processing query results: {str(e)}", "data": []})

def create_dataframe_pd(engine: Engine, query: str):
    try:
        # The engine.connect() establishes a connection from the pool
        with engine.connect() as conn:
            # pd.read_sql executes the query against the connection
            # and returns the results as a Pandas DataFrame.
            df = pd.read_sql(sql=query, con=conn)
            return df
    except Exception as e:
        # It's good practice to handle potential errors
        # and return a consistent error format.
        return json.dumps({"error": f"Error processing query results: {str(e)}", "data": []})
    
# print(create_dataframe_pd(db_engine, "SELECT ar.name AS artist_name, COUNT(DISTINCT al.album_id) AS album_count FROM artist ar JOIN album al ON ar.artist_id = al.artist_id JOIN track t ON al.album_id = t.album_id JOIN genre g ON t.genre_id = g.genre_id WHERE g.name = 'Metal' GROUP BY ar.name;;"))