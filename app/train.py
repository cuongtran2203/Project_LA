import argparse
import sqlite3
import pandas as pd
import tensorflow
import paths
import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import numpy as np
# Cân bằng dữ liệu
target_classes=['Drama', 'Anime', 'Mystery', 'Comedy', 'Crime', 'Romance', 'Legal',
    'Children', 'War', 'Action', 'Adventure', 'Science-Fiction',
    'Family', 'Supernatural', 'History', 'Thriller', 'Fantasy',
    'Medical', 'Nature', 'Travel', 'Sports', 'DIY', 'Adult', 'Music',
    'Horror', 'Food', 'Espionage', 'Western']
def resample_dataset(annotation_df):
    num_samples=2000

    target_classes.sort()
    df_resample_list=list()
    for target in target_classes:
        df_new=annotation_df[annotation_df["genre"] == target].copy()
        df_r=resample(df_new,n_samples=num_samples,random_state=42)
        df_resample_list.append(df_r)
    return pd.concat(df_resample_list).reset_index(drop=True)
#Load dữ liệu từ SQLite
def load_data(sqlite_file):
    con=sqlite3.connect(sqlite_file)
    df_info=pd.read_sql_query("SELECT * from tvmaze",con)
    df_genre=pd.read_sql_query("SELECT * from tvmaze_genre",con)
    con.close()
    return df_info,df_genre
#Loại bỏ những kí tự đặc biệt
def remove_tags(text):
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    text = text.replace('</b>', '')
    text = text.replace('<b>', '')
    text = text.replace(r'\W', '')
    text=text.lower()
    return text
def onehot(x):
    zero_list=[0]*len(target_classes)
    for d in x:
        zero_list[target_classes.index(d)]=1
    return zero_list
    
#Tiền xử lý dữ liệu
def prepocess(df_info,df_genre):
    df_info["description"].dropna(axis=0)
    df_info['description'].fillna('', inplace=True)
    df_info["description"].apply(lambda x: remove_tags(x))
    df_info["description"].str.lower()
    df_merged=pd.merge(df_info,df_genre,on="tvmaze_id",how="inner")
    df_resample=resample_dataset(df_merged)
    # grouped_df = df_resample.groupby(['tvmaze_id',"description"])['genre'].apply(set).reset_index()
    # grouped_df["genre"]=grouped_df["genre"].apply(lambda x:onehot(x))
    return  df_resample

    
    
    
    
def train_model(X_train,y_train,batch_size=512,epoch=2):

    # Build model
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(28, activation='softmax'))  # Adjust the output dimension based on your number of genres
    loss= tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    # Train model
    print(model.summary())

    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,  validation_data=(X_test, y_test))   
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

def save_model(model):
    # Save the trained model to paths.location_of_model
    model.save("best_model_v.h5")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model from SQLite data")
    parser.add_argument("--training-data",default="tvmaze.sqlite", required=True, help="Path to the SQLite database file")
    parser.add_argument("--batch_size",default=512, help="Batch size")
    parser.add_argument("--epochs",default=2, help="epochs")
    
    

    args = parser.parse_args()

    # Load data from the SQLite database
    df_info,df_genre = load_data(args.training_data)
    df_resample=prepocess(df_info,df_genre)
    print(df_resample)
    X = df_resample['description']
    y = df_resample['genre'].to_list()
    # Tokenization
    tokenizer = Tokenizer(num_words=300)
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=200)
    #Encoding labels  
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(y_train)
    y_train=np.array(y_train)
    
    y_test=np.array(y_test)
    # Train the model
    model = train_model(X_train,y_train,batch_size=args.batch_size,epoch=args.epochs)
    # Evaluate the model (assuming you want to)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model training complete. Accuracy: {accuracy:.2f}")
    # Do you want to retrain the model on the whole data set now?
    # Save the trained model to a file
    save_model(model)
