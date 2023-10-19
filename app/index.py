import argparse
import sqlite3
import json
import pickle
# Import any additional libraries or modules you may need

import paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer(stop_words="english")
def index_data(sqlite_file,query="SELECT * FROM tvmaze"):
    try:
        # Connect to the SQLite database
        connection = sqlite3.connect(sqlite_file)
        cursor = connection.cursor()

        # update this query to reflect the tvmaze schema
        query = query
        cursor.execute(query)

        data_to_index = {}
        vector_list=[]
        for row in cursor.fetchall():
            # Perform any preprocessing tasks here (e.g., calculate embeddings, stem words)
            # This is a placeholder; replace with your actual preprocessing code
            showname=""
            description=""
            new_description_vector=0
            
            id,showname,description = preprocess_data(row)
            print(id,showname)
            if len(showname)>5 and description is not None and description!="None" and showname!="None":
    
                X= vectorizer.fit_transform([description])
                dict_={str(id):{showname:description}}
                data_to_index.update(dict_)
                vector_list.append(X)

        print(X)
 
        
        with open(paths.location_of_index, 'a+',encoding="utf-8") as index_file:
            json.dump(data_to_index,index_file,indent = 4,ensure_ascii=False)
        # Save the indexed data to a separate file (e.g., JSON? Flat lines?)

        connection.close()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
def remove_tags(text):
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    text = text.replace('</b>', '')
    text = text.replace('<b>', '')
    text = text.replace(r'\W', '')
    text=text.lower()
    return text
def preprocess_data(data):
    # Implement your preprocessing tasks here
    # This is a placeholder; replace with your actual preprocessing code
    showname=""
    description=""
    if data[-1] is not None :
        id=data[0]
        showname=data[2]
        description=data[-1]
        description=remove_tags(description)
        return id,showname,description
    return 999999999999,"None","None"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index data from an SQLite database")
    parser.add_argument("--raw-data", required=True,default="tvmaze.sqlite", help="Path to the SQLite database file")
    parser.add_argument("--query",default="SELECT * FROM tvmaze", help="Path to the SQLite database file")

    args = parser.parse_args()

    index_data(args.raw_data,args.query)
