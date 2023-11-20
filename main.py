from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from functions.preprocess import preprocess
from functions.tokenization import tokenize_texts
from functions.model import test
from pydantic import BaseModel

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PredictionInput(BaseModel):
    data: str

class ReviewList(BaseModel):
    reviews: list[PredictionInput]

# Load your model
#model = joblib.load('model_filename.pkl')

@app.post('/predict-a-lot')
def predict_a_lot(review_list: ReviewList):
    
    predictions = []

    for i in range(0, len(review_list.reviews)):
        processed_data = preprocess(review_list.reviews[i].data)
        prediction = test(processed_data)
        predictions.append(prediction)
    return predictions


@app.post('/predict')
def predict(input: PredictionInput):
    processed_data = preprocess(input.data)
    # Make prediction
    prediction = test(processed_data)

    return prediction

@app.get('/hello')
def hello():
    return 'Hello World!'
