import argparse
import json
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Import any additional libraries or modules you may need
import pandas as pd
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
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
        with open("inputs.json",encoding="utf8") as f:
            data_json=json.load(f)
        for key in data_json.keys():
            for k in data_json[key].keys():
                description.append(data_json[key][k])
        tokenized_corpus=[doc.split(" ") for doc in description]
        bm25_ranking=BM25Okapi(tokenized_corpus)
        top3=None
        with open(input_file,"r",encoding="utf8") as f:
            data_list=f.readlines()
        for data in data_list:
            data=data.replace("\n","")
            data=data.split(" ")
            score=bm25_ranking.get_scores(data)
            top_results = sorted(range(len(score)), key=lambda i: -score[i])
            top3=bm25_ranking.get_top_n(query=data,documents=description,n=3)
            # print(len(top3))
    
        #load vector database
        # tfidf=vectorizer.fit(["human resilience, and the quest for knowledge in the face of the unknown. It explores themes of time, space, and the boundaries of human understanding as the characters navigate this surreal and captivating world within the cosmic chronometer."])

        # Example matched shows (replace with your actual search results)
        matched_shows = []
        dict_=dict()
        for key in data_json.keys():
            for k in data_json[key].keys():
                # print(k)
                for t in top3:
                    # print(t)
                    check_str=data_json[key][k]
                    # print(check_str)
                    if check_str == t:
                       
                        dict_["tvmaze_id"]=key
                        dict_["showname"]=k
                        matched_shows.append(dict_)

        # Write the matched shows to the output JSON file
        with open(output_json_file, 'w', encoding='UTF-8') as json_file:
            json.dump(matched_shows, json_file, ensure_ascii=False,indent=4)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")

    args = parser.parse_args()

    search_tv_shows(args.input_file, args.output_json_file, args.encoding)
