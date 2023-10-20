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
# vectorizer = TfidfVectorizer(stop_words="english")
def remove_tags(text):
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    text = text.replace('</b>', '')
    text = text.replace('<b>', '')
    text = text.replace('<i>', '')
    text = text.replace('</i>', '')
    text = text.replace(r'\W', '')
    text=text.lower()
    return text
def search_tv_shows(input_file, output_json_file, encoding='UTF-8'):
    try:
        description=[]
        with open("inputs.json",encoding="utf8") as f:
            data_json=json.load(f)
        for key in data_json.keys():
            for k in data_json[key].keys():
                description.append(data_json[key][k])
        # X=vectorizer.fit_transform(description)
        tokenized_corpus=[doc.split(" ") for doc in description]
        bm25_ranking=BM25Okapi(tokenized_corpus)
        top3=None
        with open(input_file,"r",encoding="utf8") as f:
            data_list=f.readlines()
        for data in data_list:
            data=data.replace("\n","")
            data=remove_tags(data)
            data=data.split(" ")
            score=bm25_ranking.get_scores(data)
            top3= np.argsort(score)[::-1][:3]
            # print(top3)
        matched_shows = []
        for top in top3:
            dict_=dict()  
            dict_["tvmaze_id"]=list(data_json.keys())[top]
            dict_["showname"]=list(list(data_json.values())[top].keys())[0]
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
