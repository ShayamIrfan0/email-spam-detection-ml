# Email Spam Detection Using Machine Learning

## Overview
A machine learning project that classifies text messages
as spam or legitimate using Python.

## Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Streamlit

## Dataset
This project uses the UCI SMS Spam Collection dataset.

The dataset contains labelled SMS messages rather than
full email messages, so this version is a text spam
classification prototype.

Dataset:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Machine Learning
The project uses TF-IDF to convert text into numerical
features and Multinomial Naive Bayes for classification.

## Evaluation
The model is evaluated using accuracy, precision,
recall, F1-score, and a confusion matrix.

## Run Locally

1. Install Python.
2. Install dependencies:

   pip install -r requirements.txt

3. Prepare the dataset using prepare_data.py.
4. Train the model:

   python train_model.py

5. Launch the website:

   python -m streamlit run app.py

## Limitations
This project is for educational purposes.
Its predictions are not guaranteed to be correct
and should not be treated as proof that a message is safe.