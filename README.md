# Financial Sentiment Analysis with BERT Models

## Introduction
This project implements financial sentiment analysis using two hybrid models: **BERT + LSTM** and **BERT + BiLSTM**. It covers the complete pipeline from web scraping to model training, evaluation, and application on real-world crawled datasets.

---

## Project Structure
The project is divided into three main components:

### 1. **Web Scraping**
- **Purpose**: Collect financial news data from CNN and Factiva for training and testing.
- **Key Scripts**:
  - `test_cnn.js`: Scrapes financial articles from CNN.
  - `test_dow_jones.js`: Scrapes data from Factiva.
- The scraped data is saved locally for further preprocessing.

### 2. **Model Training**
- **Purpose**: Train and evaluate two models, **BERT + LSTM** and **BERT + BiLSTM**, using curated datasets or the crawled data.
- **Key Scripts**:
  - `model.py`: Defines the architecture for BERT + LSTM model.
  - `model_bilstm.py`: Defines the architecture for BERT + BiLSTM model.
  - `train.py`: Handles training and evaluation of the models. Results are saved to the `logs` folder.
  - `plotf1.py`: Visualizes training and validation metrics (loss, accuracy, F1 score) by reading results from the `logs` folder.
  - `parameters.py`: Saves and loads the best parameters from training, ensuring reproducibility and easy deployment.


### 3. **Model Application**
- **Purpose**: Apply the trained models on the crawled dataset to analyze sentiment in real-world data.
- **Key Scripts**:
  - `data_preprocessing.py`: Prepares the scraped data for model input by cleaning and formatting it.
  - `model.py`: Same architecture as in the training phase.
  - `train.py`: Reused for fine-tuning or evaluation on the crawled data.
  - `plot.py`: Visualizes performance metrics.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/crazyDingding/financial-sentiment-analysis.git
   cd financial-sentiment-analysis
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt