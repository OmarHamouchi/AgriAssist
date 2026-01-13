import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- CONFIGURATION ---
# Pour afficher l'arabe correctement dans les graphiques
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def make_arabic_readable(text):
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError:
    def make_arabic_readable(text): return text

# =============================================================================
# 1. CHARGEMENT DES DONNÉES PRÉTRAITÉES (CLASSIQUE)
# =============================================================================
print("--- Chargement des données (SVM) ---")
train_path = 'Data/processed/classical/train.csv'
test_path = 'Data/processed/classical/test.csv'

if not os.path.exists(train_path):
    print("❌ Erreur : Données introuvables. Lancez preprocessing_split.py avant.")
    exit()

# Important : On remplit les valeurs NaN par vide (au cas où le nettoyage a tout effacé)
train_df = pd.read_csv(train_path).fillna("")
test_df = pd.read_csv(test_path).fillna("")

X_train = train_df['text_processed']
y_train = train_df['label']
X_test = test_df['text_processed']
y_test = test_df['label']

print(f"Train : {len(X_train)} | Test : {len(X_test)}")

# =============================================================================
# 2. CRÉATION DU PIPELINE (TF-IDF + SVM)
# =============================================================================
print("\n--- Entraînement du modèle SVM ---")

# Pipeline : Les données brutes entrent -> TF-IDF -> SVM -> Prédiction
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), # On regarde les mots un par un ET par paires (bi-grams)
    ('clf', LinearSVC(random_state=42, tol=1e-5))   # Modèle SVM rapide et robuste
])

# Entraînement
pipeline.fit(X_train, y_train)
print("✅ Modèle entraîné avec succès.")

# =============================================================================
# 3. ÉVALUATION ET RAPPORT
# =============================================================================
print("\n--- Évaluation sur le Test Set ---")
y_pred = pipeline.predict(X_test)

# A. Accuracy Globale
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Précision Globale (Accuracy) : {accuracy:.2%}")

# B. Rapport détaillé par classe
print("\n📊 Rapport de Classification :")
print(classification_report(y_test, y_pred))

# C. Matrice de Confusion (Visuelle)
plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_test, y_pred)
# Récupération des noms de classes uniques
class_names = pipeline.classes_
readable_classes = [make_arabic_readable(c) for c in class_names]

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=readable_classes, yticklabels=readable_classes)
plt.title('Matrice de Confusion - SVM')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
plt.show()

# =============================================================================
# 4. SAUVEGARDE DU MODÈLE
# =============================================================================
output_dir = 'models/classical'
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'svm_model.pkl')
joblib.dump(pipeline, model_path)
print(f"\n✅ Modèle sauvegardé sous : {model_path}")

# =============================================================================
# 5. TEST RAPIDE EN DIRECT
# =============================================================================
print("\n--- Test Rapide ---")
# Une phrase piège (en dialecte ou avec bruit)
test_phrase = "اوراق الطماطم صفراء وفيها بقع سوداء" # (Les feuilles de tomates sont jaunes avec taches noires)
# Attention : Il faut appliquer le preprocess_classical sur l'input utilisateur !
# Pour ce script de test simple, le modèle va essayer de se débrouiller, 
# mais dans l'app finale, on réimportera la fonction de nettoyage.

prediction = pipeline.predict([test_phrase])[0]
print(f"Question : {test_phrase}")
print(f"Prédiction : {prediction}")