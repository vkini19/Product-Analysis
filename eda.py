import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('cleaned_reviews.csv')

df['rating'].hist()
plt.title("Distribution of Rating")
plt.savefig('rating_distribution.png')

df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
print(df['word_count'].describe())

all_text = ' '.join(df['cleaned_text'])

wordcloud = WordCloud(width=800, height=400).generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png')