# Product Review Sentiment Analysis

An end-to-end data science project that scrapes product reviews, analyzes sentiment using ML/DL models, and deploys insights via a web app.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/GitHub-Repo-brightgreen)
![License](https://img.shields.io/badge/License-MIT-orange)

## Features
- **Web Scraping**: Collect reviews from e-commerce sites (Amazon/Yelp)
- **Text Preprocessing**: NLP cleaning (stopword removal, lemmatization)
- **Sentiment Classification**: Logistic Regression, LSTM, or BERT
- **Deployment**: Flask API or Streamlit dashboard

## Project Structure
sentiment-analysis/  
├── data/  
│ ├── raw_reviews.csv # Scraped raw data  
│ └── cleaned_reviews.csv # Processed data  
├── notebooks/  
│ └── EDA.ipynb # Exploratory analysis  
├── src/  
│ ├── scraper.py # Review scraper  
│ ├── preprocess.py # Text cleaning  
│ ├── train.py # Model training  
│ └── app.py # Flask API  
├── models/  
│ └── sentiment_model.pkl # Trained model  
└── README.md  
