# Load Saved Components and Define the Prediction Logic

# Import required libraries
import pickle
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import re


# Load the TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Load the Trained Neural Network Model
model = load_model('mbti_model.h5')

# Recompile the model to suppress warnings and ensure compatibility
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the Label Encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Function to clean and preprocess user input
def clean_text(text):
    text = text.replace('|||', ' ')
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to predict MBTI type
def predict_mbti(text, tfidf_vectorizer, model, label_encoder):
    # Clean the user input
    cleaned_text = clean_text(text)
    print(f"Cleaned Text: {cleaned_text}")
    
    # Transform the input using the trained TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([cleaned_text]).toarray().astype('float32')
    print(f"TF-IDF Input Shape: {input_tfidf.shape}, Type: {type(input_tfidf)}")
    
    # Ensure TF-IDF dimensions match the model's input dimensions
    if input_tfidf.shape[1] > model.input_shape[1]:
        input_tfidf = input_tfidf[:, :model.input_shape[1]]
    elif input_tfidf.shape[1] < model.input_shape[1]:
        raise ValueError(f"Model expects {model.input_shape[1]} features, but TF-IDF vectorizer produced {input_tfidf.shape[1]} features.")
    
    print(f"Adjusted TF-IDF Input Shape: {input_tfidf.shape}")
    
    # Predict the probabilities for each class
    predictions = model.predict(input_tfidf)
    print(f"Predictions: {predictions}")
    
    # Get the class index with the highest probability
    predicted_class_index = np.argmax(predictions)
    print(f"Predicted Class Index: {predicted_class_index}")
    
    # Decode the class index back to the MBTI type
    predicted_mbti = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_mbti

# Prompt the user for input
print("Welcome to the MBTI Predictor!")
user_input = input("Enter a short description or text: ")

# Predict and display the MBTI type
predicted_type = predict_mbti(user_input, tfidf, model, label_encoder)
print(f"The predicted MBTI type is: {predicted_type}")
