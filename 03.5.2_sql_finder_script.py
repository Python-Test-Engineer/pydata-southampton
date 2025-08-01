# This gets an SQL query from a CSV file and allows you to find similar SQL queries based on natural language questions.

# We can then ask LLM to modify the SQL query to suit our needs.

# Afterwards I realised I used an LLM to create the SQL queries in the CSV file.


import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


def load_csv_to_chroma(csv_path):
    """
    Load questions and SQL queries from CSV into ChromaDB
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Initialize ChromaDB client
    client = chromadb.Client()

    # Create a sentence transformer embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create or get collection
    collection = client.create_collection(
        name="SQL_FINDER",
        embedding_function=embedding_function,
        metadata={"description": "Natural language questions paired with SQL queries"},
    )

    # Add documents to collection
    questions = df["Question"].tolist()
    sql_queries = df["SQL"].tolist()
    ids = [f"query_{i}" for i in range(len(questions))]

    collection.add(
        documents=questions, metadatas=[{"sql": sql} for sql in sql_queries], ids=ids
    )

    return collection


def find_similar_query(user_input, collection, n_results=1):
    """
    Find the most semantically similar question and return its SQL query
    """
    results = collection.query(query_texts=[user_input], n_results=n_results)

    matches = []
    for i in range(min(n_results, len(results["documents"][0]))):
        matches.append(
            {
                "question": results["documents"][0][i],
                "sql": results["metadatas"][0][i]["sql"],
                "distance": (
                    results["distances"][0][i] if "distances" in results else None
                ),
            }
        )

    return matches


if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "03.5_chinook_sql.csv"
    print("Starting SQL Finder Script...will take time as DB created from CSV.")
    print(f"CSV file path: {csv_path}")

    # Load CSV into ChromaDB
    print(f"Loading queries from {csv_path}...")
    collection = load_csv_to_chroma(csv_path)
    print(f"Loaded {collection.count()} queries into ChromaDB.")

    # Test with a specific question
    test_question = "Who made purchases in 2010?"
    test_question = (
        "What are the total sales by month of 2010 order by month desc?"  # 2
    )
    print(f"\nTesting with question: '{test_question}'")
    matches = find_similar_query(test_question, collection)
    if matches:
        match = matches[0]
        print("\nMost similar question:")
        print(f"Q: {match['question']}\n")
        print(f"SQL: {match['sql']}")
        if match["distance"] is not None:
            print(f"\nSimilarity score: {1 - match['distance']:.4f}\n")
    else:
        print("No matching queries found.")

    # # Interactive query mode
    print("\nEnter a question to find similar SQL queries (or 'exit' to quit):")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "exit":
            break

        matches = find_similar_query(user_input, collection)
        if matches:
            match = matches[0]
            print("\nMost similar question:")
            print(f"Q: {match['question']}\n")
            print(f"SQL: {match['sql']}")
            if match["distance"] is not None:
                print(f"Similarity score: {1 - match['distance']:.4f}\n")
        else:
            print("No matching queries found.")


"""
See 03.6 as well

~ Using a retreived SQL I asked it to modify it to give me a different result.

given this SQL for the data already used, modify it to give the SQL for an artist called 'David Bowie'.

Here is a useful SQL

SELECT Track.Name AS TrackName, Album.Title AS AlbumTitle, Artist.Name AS ArtistName FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId JOIN Artist ON Album.ArtistId = Artist.ArtistId

----------------
~ Claude's response
I'll modify the SQL query to filter for tracks by the artist "David Bowie".
Looking at the base query you provided, it retrieves track names along with their album titles and artist names. To filter this specifically for David Bowie, we need to add a WHERE clause that specifies the artist name.
Here's the modified SQL query:


SELECT Track.Name AS TrackName, Album.Title AS AlbumTitle, Artist.Name AS ArtistName 
FROM Track 
JOIN Album ON Track.AlbumId = Album.AlbumId 
JOIN Artist ON Album.ArtistId = Artist.ArtistId
WHERE Artist.Name = 'David Bowie'
This query will:

Join the Track, Album, and Artist tables
Filter the results to only include tracks where the artist name is 'David Bowie'
Return the track names, album titles, and artist name for all David Bowie tracks in the database

If you want to also sort the results, you could add an ORDER BY clause like this:

SELECT Track.Name AS TrackName, Album.Title AS AlbumTitle, Artist.Name AS ArtistName 
FROM Track 
JOIN Album ON Track.AlbumId = Album.AlbumId 
JOIN Artist ON Album.ArtistId = Artist.ArtistId
WHERE Artist.Name = 'David Bowie'
ORDER BY Album.Title, Track.Name

This would order the results first by album title and then by track name within each album.RetryClaude does not have the ability to run the code it generates yet.Claude can make mistakes. Please double-check responses. 3.7 Sonnet
"""
