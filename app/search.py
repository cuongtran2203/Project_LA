import argparse
import json
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Import any additional libraries or modules you may need
import pandas as pd
import pickle
import numpy as np
import paths
vectorizer = TfidfVectorizer(stop_words="english")
def search_tv_shows(input_file, output_json_file, encoding='UTF-8'):
    try:
        # Do something with your index, whereever you put it.

        # Read the search query from the input file
        
        # Implement your search logic here to find matching TV shows
        # read from the file you saved in index.py
        # matched_shows = search_tv_shows(search_query)
        description=[]
        with open(input_file) as f:
            data_list=f.readlines()
        
        #load vector database
        with open("vectors.pkl",'rb') as file:
            vectors=pickle.load(file)
        new_description_vector = vectorizer.transform(["about description"])
        similarity_scores = cosine_similarity(new_description_vector, vectors)
        # Sắp xếp kết quả
        top_indices = similarity_scores[0].argsort()[-3:][::-1]
        print(top_indices)
        # Example matched shows (replace with your actual search results)
        matched_shows = [
            {"tvmaze_id": 1, "showname": "Show 1"},
            {"tvmaze_id": 2, "showname": "Show 2"},
            {"tvmaze_id": 3, "showname": "Show 3"}
        ]

        # Write the matched shows to the output JSON file
        with open(output_json_file, 'w', encoding='UTF-8') as json_file:
            json.dump(matched_shows, json_file, ensure_ascii=False)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")

    args = parser.parse_args()

    search_tv_shows(args.input_file, args.output_json_file, args.encoding)
