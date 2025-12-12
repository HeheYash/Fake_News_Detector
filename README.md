# 📰 Fake News Detector — Machine Learning + Deep Learning Hybrid Model
A complete Fake News Classification system that combines **NLP preprocessing**, **TF-IDF vectorization**, **PCA dimensionality reduction**, and a **Neural Network classifier** to detect FAKE or REAL news with improved stability using a trigger-word safety mechanism.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Dataset Details](#dataset-details)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Running the Detector](#running-the-detector)
- [Model Architecture](#model-architecture)
- [Project Files](#project-files)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🔍 Overview
This project builds a **Fake News Detection** pipeline that identifies whether a news headline or article is *REAL* or *FAKE*.  
It uses a combination of classical NLP techniques and a lightweight deep learning model to deliver high accuracy, fast inference, and a smooth interactive prediction experience.

The system includes:
- Text cleaning and stemming  
- TF-IDF vectorization  
- Dimensionality reduction via PCA  
- A dense neural network classifier  
- A safety mechanism that boosts the "fake" probability when common misinformation keywords appear  

---

## ⭐ Key Features

### ✔ 1. Automated Text Cleaning Pipeline
- Lowercasing  
- Regex filtering  
- Stopword removal  
- Porter stemming  

### ✔ 2. TF-IDF Vectorization
- Converts text into numerical feature vectors  
- 3000 max features for speed and efficiency  

### ✔ 3. PCA Dimensionality Reduction
- Reduces dimensionality from 3000 → 100 components  
- Faster training and reduced memory usage  

### ✔ 4. Deep Learning Classifier
A simple yet effective neural network:
- Dense layers  
- ReLU activation  
- Dropout regularization  
- Sigmoid output for binary classification  

### ✔ 5. Trigger-Word Safety Net
Certain words like *“leaked”*, *“conspiracy”*, *“banned”*, etc. increase the model’s confidence toward FAKE when present.

### ✔ 6. Interactive Prediction Mode
You can type in any headline and instantly get:
- Internal model score  
- Adjusted score  
- Prediction: **FAKE**, **REAL**, or **UNSURE**

---

## 📊 Dataset Details

### 🟦 True News Dataset (`True.csv`)
Contains verified, factual news articles including:
- `title`
- `text`
- `subject`
- `date`

The script automatically labels this dataset as **0 (REAL)**.

### 🟥 Fake News Dataset (`Fake.csv`)
Contains fabricated or manipulated news stories with identical fields:
- `title`
- `text`
- `subject`
- `date`

The script labels this dataset as **1 (FAKE)**.

### Data Handling Highlights
- Column names are forced to lowercase to avoid mismatches  
- Rows missing `title` or `text` are removed  
- TRUE and FAKE datasets are merged and shuffled  
- Dataset trimmed to **15,000 rows** for performance (can be increased)  

---

## 🧱 Project Architecture
Fake News Detector/
│
├── Fake.csv # Fake news dataset
├── True.csv # Real news dataset
├── Fake News Detector.py # Main script (training + demo)
│
└── README.md # Project documentation

---

## ⚙ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Install NLTK resources
```
import nltk
nltk.download("stopwords")
```

## 🧠 How It Works
### PHASE 1 — File Loading

Loads Fake.csv and True.csv, normalizes columns, and performs safety checks.

### HASE 2 — Labeling

Assigns:

REAL → 0

FAKE → 1

Datasets are merged and shuffled.

### PHASE 3 — Preprocessing + Vectorization + PCA

Clean text (regex, stopwords, stemming)

Convert to TF-IDF vectors

Reduce dimensionality (3000 → 100)

### PHASE 4 — Neural Network Training

A lightweight NN trains using an 80/20 split.
