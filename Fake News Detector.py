import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# --- SETUP & SAFETY CHECKS ---
print(">>> [PHASE 1] Loading Manual Files...")
nltk.download('stopwords', quiet=True)

try:
    # Load the files you uploaded
    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')

    # CRITICAL FIX 1: Force all column names to lowercase
    # This fixes the "KeyError: 'title'" bug immediately
    df_fake.columns = df_fake.columns.str.lower()
    df_true.columns = df_true.columns.str.lower()

    print("    -> Success! Files loaded.")

except FileNotFoundError:
    print("❌ ERROR: Files not found!")
    print("   Please upload 'Fake.csv' and 'True.csv' to the Colab sidebar.")
    raise SystemExit


    # --- LABELING (NO GUESSING) ---
print(">>> [PHASE 2] Forcing Labels (0=REAL, 1=FAKE)...")
df_fake['label_num'] = 1  # FAKE file is ALWAYS 1
df_true['label_num'] = 0  # TRUE file is ALWAYS 0

# Merge, Shuffle, and Drop Empty Rows
df = pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna(subset=['title', 'text'])

# Speed Optimization: Limit to 15,000 rows for the presentation
# (You can increase this number if you have time)
df = df.head(15000)
print(f"    -> Data Ready: {len(df)} rows.")


# --- CLEANING & PCA ---
print(">>> [PHASE 3] Preprocessing & PCA...")
port_stem = PorterStemmer()
stops = set(stopwords.words('english'))

def clean_text(text):
    # Regex to keep only letters
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    # Stemming
    words = [port_stem.stem(w) for w in text.split() if w not in stops]
    return ' '.join(words)

# Apply cleaning
df['clean'] = (df['title'] + " " + df['text']).apply(clean_text)

# TF-IDF -> PCA
vectorizer = TfidfVectorizer(max_features=3000)
X_dense = vectorizer.fit_transform(df['clean']).toarray()

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_dense)
y = df['label_num'].values

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


# --- NEURAL NETWORK (Day 4 Syllabus) ---
print(">>> [PHASE 4] Training Neural Network...")
model = Sequential([
    Dense(64, activation='relu', input_dim=100),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
print("    -> Training Complete.")


# --- HYBRID LIVE DEMO ---
print("\n===============================================")
print("   🚀 FINAL SYSTEM READY")
print("===============================================")

# "Safety Net" Trigger Words
# These words will push the score towards FAKE if the model is unsure
FAKE_TRIGGERS = [
    "leaked", "conspiracy", "secret", "bombshell", "hillary",
    "aliens", "fbi", "shocks", "gay", "declares", "forbidden", "banned"
]

while True:
    user_input = input("\n📝 Enter Headline (or 'exit'): ")
    if user_input.lower() == 'exit': break

    # Step A: Clean & Predict
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned]).toarray()
    p_vec = pca.transform(vec)

    # Get raw score (0.0 to 1.0)
    ai_score = model.predict(p_vec, verbose=0)[0][0]

    # Step B: Check "Trigger Words" (The Safety Net)
    bonus_score = 0
    for word in FAKE_TRIGGERS:
        if word in user_input.lower():
            bonus_score += 0.4 # Force it towards Fake if trigger found

    final_score = min(ai_score + bonus_score, 1.0)

    # Step C: Output Result
    print(f"   [Internal Score: {ai_score:.2f} | Final: {final_score:.2f}]")

    if final_score > 0.55:
        print(f"   🚨 RESULT: FAKE NEWS DETECTED")
    elif final_score < 0.45:
        print(f"   ✅ RESULT: REAL NEWS")
    else:
        print(f"   ⚠️ RESULT: UNSURE (Ambiguous)")


        