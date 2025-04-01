# IMDB Sentiment Analysis

## Overview
This project implements sentiment analysis on the IMDB movie reviews dataset using both traditional machine learning and deep learning approaches. It compares the performance of Logistic Regression, Naive Bayes, LSTM, and Bidirectional LSTM models for classifying movie reviews as positive or negative.

## Project Structure
```
IMDB-Sentiment-Analysis/
│
├── notebooks/
│   └── nlp_sentiment_analysis.ipynb  # Main analysis notebook
│
├── images/                 # Visualizations generated during analysis
│
├── models/                 # Saved trained models
│
└── data/
    └── README.md           # Info about the IMDB dataset from Kaggle
```

## Features
- Comprehensive text preprocessing with spaCy lemmatization
- Implementation of traditional ML models (Logistic Regression, Naive Bayes)
- Implementation of deep learning models (LSTM, Bidirectional LSTM)
- Detailed performance comparison and visualization
- Error analysis and model interpretation
- Saved models for future use

## Requirements
All required packages are listed in `requirements.txt`. Install them using:
```
pip install -r requirements.txt
```

## Dataset
This project uses the IMDB Movie Reviews dataset from Kaggle, which contains 50,000 movie reviews labeled as positive or negative. The dataset can be accessed at:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Running the Project
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/IMDB-Sentiment-Analysis.git
   cd IMDB-Sentiment-Analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the IMDB dataset from Kaggle and place it in the appropriate location.

4. Run the Jupyter notebook:
   ```
   jupyter notebook notebooks/nlp_sentiment_analysis.ipynb
   ```

5. The notebook contains all the steps:
   - Data exploration
   - Preprocessing
   - Feature engineering
   - Model implementation
   - Evaluation
   - Error analysis
   - Conclusions

## Results
The project achieves the following performance metrics on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|-------|----------|-----------|--------|----------|-------------------|
| Logistic Regression | 0.8642 | 0.8715 | 0.8542 | 0.8628 | 6.23 |
| Naive Bayes | 0.8327 | 0.8498 | 0.8089 | 0.8288 | 0.78 |
| LSTM | 0.8786 | 0.8823 | 0.8739 | 0.8781 | 145.47 |
| Bi-LSTM | 0.8895 | 0.8967 | 0.8808 | 0.8887 | 210.35 |

## Key Findings
- Bidirectional LSTM achieved the best performance with an F1-score of 0.8887
- Traditional ML models, particularly Logistic Regression, offered competitive performance with significantly lower computational requirements
- There's a clear trade-off between model performance and training time
- All models struggled with sarcasm, irony, and complex negation patterns

## Using the Models
The saved models can be loaded and used for sentiment prediction:

```python
# Example for loading traditional ML models
import pickle

# Load models
with open('models/traditional_models_package.pkl', 'rb') as f:
    models = pickle.load(f)

# Get logistic regression model and vectorizer
lr_model = models['logistic_regression']['model']
tfidf_vectorizer = models['logistic_regression']['vectorizer']

# Predict sentiment
def predict_sentiment(text):
    # Preprocess text (simplified version shown)
    processed_text = text.lower()
    
    # Vectorize
    features = tfidf_vectorizer.transform([processed_text])
    
    # Predict
    prediction = lr_model.predict(features)[0]
    return "Positive" if prediction == 1 else "Negative"
```

## Future Work
- Implement transformer-based models like BERT or RoBERTa
- Explore ensemble methods combining multiple model predictions
- Investigate advanced techniques for handling sarcasm and irony
- Apply transfer learning with pre-trained language models

## Author
Khadhar Hameed Khan Pathan -  Hameed1117

## Acknowledgements
- The IMDB dataset creators
- Kaggle for hosting the dataset
- The scikit-learn, TensorFlow, and spaCy development teams

