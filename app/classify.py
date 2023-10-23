import argparse
import json
import os
import tensorflow
from tensorflow.keras.models import load_model
# Import any additional libraries or modules you may need
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import paths
def preprocess(X:str):
    tokenizer = Tokenizer(num_words=300)
    tokenizer.fit_on_texts([X])
    X = tokenizer.texts_to_sequences([X])
    X = pad_sequences(X, maxlen=200)
    return X
#Loại bỏ những kí tự đặc biệt
def remove_tags(text):
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    text = text.replace('</b>', '')
    text = text.replace('<b>', '')
    text = text.replace(r'\W', '')
    text = text.replace('</i>', '')
    text = text.replace('<i>', '')
    text=text.lower()
    return text
def classify_tv_show(input_file, output_json_file, encoding='UTF-8', explanation_output_dir=None):
    try:
        # Read the description from the input file

        # Load your model, perhaps from paths.location_of_model
        
        # Implement your classification logic here to identify TV show genres
        # load your model from somewhere in the /app directory
        model=load_model(paths.location_of_model)
        # This is a placeholder; you should replace it with your actual code
        # genres = classify_tv_show(description)
        threshold=0.35
        count=0
        # Example genres (replace with your actual genre prediction)
        genres = ['Drama', 'Anime', 'Mystery', 'Comedy', 'Crime', 'Romance', 'Legal',
       'Children', 'War', 'Action', 'Adventure', 'Science-Fiction',
       'Family', 'Supernatural', 'History', 'Thriller', 'Fantasy',
       'Medical', 'Nature', 'Travel', 'Sports', 'DIY', 'Adult', 'Music',
       'Horror', 'Food', 'Espionage', 'Western']
        with open(input_file,"r") as f:
            data_list=f.readlines()
        for data in data_list:
            # input=preprocess(data.replace("\n","").lower())
            data=remove_tags(data)
            
            output=model.predict([data])
            print(output)
            output=np.where(output>=threshold,1,0)
            idx=np.where(np.asarray(output)>=threshold)
            print(idx)
            result=[]
            for i in idx:
                # # Write the identified genres to the output JSON file
                if i.size>0:
                    for ix in i:
                        if genres[ix] not in result:
                            result.append(genres[ix])
                            print(genres[ix])
            print(result)
            with open(output_json_file, 'w', encoding='UTF-8') as json_file:
                json.dump({count:result}, json_file, ensure_ascii=False,indent=4)
            count+=1

        # Optionally, write an explanation to the explanation output directory
        if explanation_output_dir:
            explanation_filename = os.path.join(explanation_output_dir, "explanation.txt")
            explanation = "This is an example explanation."
            with open(explanation_filename, 'w', encoding='UTF-8') as exp_file:
                exp_file.write(explanation)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify TV show genres based on description")
    parser.add_argument("--input-file", required=True, help="Path to the input file with TV show description")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for genres")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")
    parser.add_argument("--explanation-output-dir", help="Directory for explanation output")

    args = parser.parse_args()

    classify_tv_show(args.input_file, args.output_json_file, args.encoding, args.explanation_output_dir)
