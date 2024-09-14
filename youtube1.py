import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# App title
st.title('YouTube Comment Spam Predictor')
st.write('An Insightful App to Predict Whether a YouTube Comment is Spam or Not')

# Upload the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv("Youtube-Spam-Dataset.csv")
    
    # Preview the dataset
    st.subheader('Dataset Preview')
    st.write(df.head())
    
    # Data Preprocessing
    st.subheader('Data Preprocessing')
    
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'\W', ' ', text)      # remove special characters
        text = text.lower()                  # convert to lowercase
        return text
    
    df['cleaned_text'] = df['CONTENT'].apply(clean_text)
    
    st.write('Data after Cleaning:')
    st.write(df[['CONTENT', 'cleaned_text']].head())
    
    # Split the dataset into features and target
    X = df['cleaned_text']
    y = df['CLASS']  # Assuming 'CLASS' is the label (0 for not spam, 1 for spam)
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Show model accuracy
    st.subheader('Model Performance')
    st.write(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    st.text('Classification Report:')
    st.text(classification_report(y_test, y_pred))
    
    # User Input for prediction
    st.subheader('Test with Your Own Comment')
    user_input = st.text_area('Enter a YouTube comment:')
    
    if st.button('Predict'):
        if user_input:
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)
            
            if prediction == 1:
                st.write('Prediction: Spam')
            else:
                st.write('Prediction: Not Spam')

