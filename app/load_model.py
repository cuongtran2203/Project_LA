import tensorflow as tf
from tensorflow.keras.models import load_model
import paths
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
model=load_model(paths.location_of_model)
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(["The friendship of two women is torn as one, supposedly innocent, is jailed accused of murdering the other one's husband, blaming her."])
X = tokenizer.texts_to_sequences(["The friendship of two women is torn as one, supposedly innocent, is jailed accused of murdering the other one's husband, blaming her."])
X = pad_sequences(X, maxlen=200)

string="is a show on The Weather Channel that airs 3 times every weekday."
df=pd.DataFrame({"description":[string.lower()]})
output=model.predict(X)
predicted_label_index = np.argmax(output, axis=1)
# print(predicted_label_index )
top3 = np.argsort(-output, axis=1)[:, :3] 
print(top3)
genres = ['Drama', 'Anime', 'Mystery', 'Comedy', 'Crime', 'Romance', 'Legal',
'Children', 'War', 'Action', 'Adventure', 'Science-Fiction',
'Family', 'Supernatural', 'History', 'Thriller', 'Fantasy',
'Medical', 'Nature', 'Travel', 'Sports', 'DIY', 'Adult', 'Music',
'Horror', 'Food', 'Espionage', 'Western']
