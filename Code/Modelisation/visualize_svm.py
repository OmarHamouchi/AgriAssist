import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os

# --- GESTION ARABE ---
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def make_arabic_readable(text):
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError:
    def make_arabic_readable(text): return text

# 1. RECHARGER LES DONNÉES ET LE MODÈLE
train_path = 'Data/processed/classical/train.csv'
if not os.path.exists(train_path):
    print("❌ Erreur : Données introuvables.")
    exit()

df = pd.read_csv(train_path).fillna("")
X = df['text_processed']
y = df['label']

# On recrée le pipeline (identique à train_svm.py)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LinearSVC(random_state=42, tol=1e-5))
])

print("⏳ Génération de la courbe d'apprentissage (cela peut prendre quelques secondes)...")

# 2. CALCUL DE LA LEARNING CURVE
# Cela va entraîner le modèle plusieurs fois sur des tailles différentes de données
train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, 
    cv=5, # Validation croisée à 5 plis (très robuste)
    n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5), # 10%, 30%, 50%, 75%, 100% des données
    scoring='accuracy'
)

# Calcul des moyennes et écarts-types
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 3. TRACÉ DU GRAPHIQUE
plt.figure(figsize=(10, 6))
plt.title(make_arabic_readable("Courbe d'Apprentissage (Learning Curve) - SVM"))
plt.xlabel("Nombre d'exemples d'entraînement")
plt.ylabel("Accuracy (Précision)")
plt.grid()

# Zone d'écart-type (pour faire "scientifique")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

# Courbes
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score Entraînement")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score Cross-Validation")

plt.legend(loc="best")
plt.show()