from fastapi import FastAPI
import joblib
from functions.preprocess import preprocess
from functions.tokenization import tokenize_texts

app = FastAPI()

# Load your model
model = joblib.load('model_filename.pkl')

@app.post('/predict')
def predict(data):
    # Preprocess input data
    processed_data = preprocess(data)
    processed_data = tokenize_texts(processed_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Postprocess and return prediction
    return prediction

@app.post('/hello')
def hello():
    return 'Hello World!'