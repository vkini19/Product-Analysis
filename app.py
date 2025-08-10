from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    cleaned = preprocess(text)  # Reuse preprocessing function
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)