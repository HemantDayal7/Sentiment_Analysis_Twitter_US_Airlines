# Sentiment Analysis of Twitter Data for US Airlines

This project implements sentiment analysis on tweets related to US airlines using a fine-tuned BERT model. The aim is to classify tweets into positive, neutral, or negative sentiments, leveraging the power of transformer-based architectures for accurate and context-aware predictions.

## Project Overview
- **Dataset**: Twitter US Airline Sentiment Dataset
- **Model**: Fine-tuned BERT (Bidirectional Encoder Representations from Transformers)
- **Performance**: Achieved an accuracy of 79.5% and an F1-score of 79.1%

## Folder Structure
```
Sentiment_Analysis_Twitter_US_Airlines/
├── Final_AI_Bert_Model_Code.ipynb  # Jupyter Notebook for the project
├── tweets.csv                      # Dataset used for training and evaluation
```

## Key Features
1. **Preprocessing Steps**:
   - Removal of URLs, mentions, and special characters.
   - Tokenization and lemmatization to standardize text.
2. **Model Training**:
   - Fine-tuned the `bert-base-uncased` model on the dataset.
   - Employed cross-entropy loss and the AdamW optimizer.
3. **Evaluation Metrics**:
   - Accuracy, precision, recall, and F1-score were used to assess performance.

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/HemantDayal7/Sentiment_Analysis_Twitter_US_Airlines.git
   cd Sentiment_Analysis_Twitter_US_Airlines
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Final_AI_Bert_Model_Code.ipynb
   ```

4. Run the notebook to preprocess the data, train the model, and evaluate its performance.

## Results
- **Accuracy**: 79.5%
- **F1-Score**: 79.1%

The model outperformed baseline approaches (Logistic Regression and Majority Class Classifier) in capturing nuanced sentiments in tweets.

## Future Work
- Incorporate ensemble methods to improve classification of neutral tweets.
- Experiment with other transformer-based models (e.g., RoBERTa, XLNet).

## References
- Crowdflower. (2015). Twitter US Airline Sentiment Dataset. [Dataset Link](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

---

Feel free to use and modify this project for educational or research purposes!
