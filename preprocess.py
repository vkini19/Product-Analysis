import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('C:/Users/varnk/Downloads/Reviews.csv')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

    words = text.split()

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(preprocess_text)

df.to_csv('cleaned_reviews.csv', index=False)
print("Preprocessing complete! Saved to 'cleaned_reviews.csv'")