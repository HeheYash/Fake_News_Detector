📰 Fake News Detector — Machine Learning + Deep Learning Hybrid Model
=====================================================================

A robust Fake News Classification system that combines **NLP preprocessing**, **TF-IDF vectorization**, **PCA dimensionality reduction**, and a **Neural Network classifier** to distinguish between **REAL** and **FAKE** news articles.The system includes a **safety-net trigger-word mechanism** for improving predictions on ambiguous headlines.

📌 Table of Contents
--------------------

*   Overview
    
*   Key Features
    
*   Project Architecture
    
*   Dataset Details
    
*   Installation
    
*   How It Works
    
*   Running the Detector
    
*   Model Architecture
    
*   Project Files
    
*   Future Improvements
    
*   License
    

🔍 Overview
-----------

This project implements a **Fake News Detection** pipeline using classical NLP techniques and a lightweight neural network. It processes two datasets:

*   **True News Dataset**
    
*   **Fake News Dataset**
    

The system cleans, merges, preprocesses, vectorizes, and reduces textual data before training a neural classifier. It includes a **live interactive prediction mode**, allowing users to input news headlines and instantly receive a "Real", "Fake", or "Unsure" verdict.

⭐ Key Features
--------------

### ✔️ **1\. Automated Text Cleaning & Preprocessing**

*   Regex-based cleaning
    
*   Lowercasing
    
*   Stopword removal
    
*   Porter stemming
    
*   Merged "title" + "text" cleaned inputs
    

### ✔️ **2\. TF-IDF Vectorization**

*   Converts cleaned text into numerical vectors
    
*   Max features capped at 3000 for speed & memory efficiency
    

### ✔️ **3\. PCA Dimensionality Reduction**

*   Reduces TF-IDF vectors from 3000 → 100 dimensions
    
*   Improves model speed and reduces overfitting
    

### ✔️ **4\. Neural Network Classifier**

*   3-layer dense network with dropout
    
*   Predicts probability of a headline being Fake
    

### ✔️ **5\. Safety-Net Trigger Word System**

If certain high-risk words appear (e.g., _"leaked"_, _"bombshell"_, _"conspiracy"_), the model adjusts the score upward toward FAKE.

### ✔️ **6\. Interactive Demo Mode**

*   Enter a headline
    
*   Model outputs internal score + adjusted score
    
*   Classifies as: **REAL**, **FAKE**, or **UNSURE**
    

📊 Dataset Details
------------------

Your project uses two CSV datasets:

### **🟦 True News Dataset (True.csv)**

Contains legitimate, factual news articles. Each row includes:

*   **title** — headline
    
*   **text** — article body
    
*   **subject** — topic category
    
*   **date** — publication dateA label **0** is assigned automatically by the script.
    

### **🟥 Fake News Dataset (Fake.csv)**

Contains fabricated or manipulated news. Same structure as the True dataset.A label **1** is assigned automatically.

### 🔧 Dataset Handling in Code

Based on the script :

*   Column names are **converted to lowercase** to prevent KeyErrors.
    
*   Missing values in _title_ or _text_ are dropped.
    
*   Data is **merged** and **shuffled**.
    
*   For performance reasons, it is limited to **15,000 rows**, but this can be increased.
    

🧱 Project Architecture
-----------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Fake News Detector/  │  ├── Fake.csv                 # Fake news dataset  ├── True.csv                 # Real news dataset  ├── Fake News Detector.py    # Main training + demo script  │  └── README.md                # (YOU ARE HERE)   `

⚙️ Installation
---------------

### 1\. Clone the Repository

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone https://github.com/your-username/fake-news-detector.git  cd fake-news-detector   `

### 2\. Install Dependencies

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

### 3\. Ensure NLTK Stopwords Are Installed

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import nltk  nltk.download('stopwords')   `

🧠 How It Works
---------------

The entire workflow follows these sequential phases:

### **PHASE 1 — Load & Prepare Files**

Reads the uploaded Fake.csv and True.csv, normalizes column names, and applies labels.

### **PHASE 2 — Labeling & Merging**

Assigns:

DatasetLabel**True.csv**0**Fake.csv**1

Then merges into a single dataframe.

### **PHASE 3 — Preprocessing → TF-IDF → PCA**

*   Cleans title+text
    
*   Converts text to numeric vectors
    
*   Reduces vector dimensions from 3000 → 100
    

### **PHASE 4 — Neural Network Training**

A dense NN is trained using 80/20 split.

🧪 Running the Detector
-----------------------

### 1\. Run the main script

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python "Fake News Detector.py"   `

### 2\. Enter Headlines Interactively

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   📝 Enter Headline: Leaked FBI documents reveal secret alien contact   `

### 3\. Example Output

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [Internal Score: 0.73 | Final: 1.00]  🚨 RESULT: FAKE NEWS DETECTED   `

🧬 Model Architecture
---------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Input Layer (100 PCA components)         ↓  Dense (64 neurons, ReLU)  Dropout (30%)         ↓  Dense (32 neurons, ReLU)         ↓  Dense (1 neuron, Sigmoid)   `

### Loss Function

*   Binary Crossentropy
    

### Optimizer

*   Adam
    

### Output

*   Probability between **0 (REAL)** and **1 (FAKE)**
    

📁 Project Files
----------------

FileDescription**Fake News Detector.py**Full training + prediction pipeline, interactive model, and text preprocessing **Fake.csv**Fake news dataset (labeled as 1)**True.csv**Real news dataset (labeled as 0)

🚀 Future Improvements
----------------------

Here are optional next steps to evolve the project:

*   Replace PCA + TF-IDF with **BERT embeddings**
    
*   Add FastAPI or Flask **API endpoints**
    
*   Provide accuracy, precision, recall, confusion matrix
    
*   Add full web interface with frontend UI
    
*   Support article body instead of headline-only demo
    
*   Use LSTM or Transformer classifier
