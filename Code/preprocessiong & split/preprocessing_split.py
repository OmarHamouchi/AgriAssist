import pandas as pd
import re
import os
import string
import nltk
from sklearn.model_selection import train_test_split
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords

# =============================================================================
# 0. CONFIGURATION & TÉLÉCHARGEMENT RESSOURCES
# =============================================================================
print("--- Initialisation des ressources NLTK ---")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Téléchargement des stop-words arabes...")
    nltk.download('stopwords')

# Initialisation des outils
stemmer = ISRIStemmer()
arabic_stopwords = set(stopwords.words('arabic'))

# =============================================================================
# 1. FONCTIONS DE PREPROCESSING
# =============================================================================

def normalize_arabic(text):
    """
    Normalisation de base (Commune aux deux méthodes).
    Unifie les caractères pour réduire le bruit (Alef, Ya, Ta-Marbuta).
    """
    text = str(text)
    # Unification des Alef
    text = re.sub(r"[إأآا]", "ا", text)
    # Unification des Ya / Alef Maqsura
    text = re.sub(r"ى", "ي", text)
    # Unification Ta Marbuta
    text = re.sub(r"ة", "ه", text)
    # Suppression du Tashkeel (Voyelles courtes)
    text = re.sub(r"[\u064B-\u065F]", "", text)
    return text

def preprocess_classical(text):
    """
    Preprocessing LOURD pour SVM :
    1. Normalisation
    2. Suppression Ponctuation (STRICTE)
    3. Suppression Stop-words
    4. Stemming
    """
    # 1. Normalisation
    text = normalize_arabic(text)
    
    # --- CORRECTION ICI : SUPPRESSION STRICTE DE LA PONCTUATION ---
    # Liste des ponctuations arabes et anglaises
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation # Inclut !, ?, ., etc.
    all_punctuations = set(arabic_punctuations + english_punctuations)
    
    # On supprime tout caractère qui est dans cette liste
    text = ''.join([char for char in text if char not in all_punctuations])
    # -------------------------------------------------------------
    
    # 2. Nettoyage final (Garder uniquement lettres arabes et espaces)
    # On garde \u0621-\u064A (Lettres pures) au lieu de tout le bloc
    text = re.sub(r'[^\u0621-\u064A\s]', ' ', text)
    
    # 3. Tokenization
    words = text.split()
    
    # 4. Filtre Stop-words & Stemming
    meaningful_words = []
    for w in words:
        if w not in arabic_stopwords and len(w) > 1:
            root = stemmer.stem(w)
            meaningful_words.append(root)
    
    return " ".join(meaningful_words)

def preprocess_advanced(text):
    """
    Pipeline B : Méthode AVANCÉE (pour MarBERT / Deep Learning).
    Objectif : Contextual Embedding.
    Traitements :
      1. Normalisation légère
      2. Conservation de TOUT le reste (Stop-words, structure, ponctuation)
    """
    # Seule la normalisation est nécessaire.
    # Le Tokenizer de BERT gérera le reste lors de l'entraînement.
    return normalize_arabic(text)

# =============================================================================
# 2. CHARGEMENT ET SPLIT (TRAIN / TEST)
# =============================================================================
input_path = 'Data/data_cleaned/Arabic_Question_Cleaned.csv'

if not os.path.exists(input_path):
    print(f"❌ Erreur : Fichier introuvable à {input_path}")
    print("Veuillez lancer l'étape précédente (Nettoyage & EDA) d'abord.")
    exit()

print(f"\nChargement des données depuis : {input_path}")
df = pd.read_csv(input_path)
print(f"Total lignes : {len(df)}")

# SPLIT STRATIFIÉ (80% Train - 20% Test)
# Important : On split AVANT le preprocessing spécifique pour garantir
# que les mêmes questions sont dans le Train/Test pour les deux méthodes.
print("\nDivision des données (Split Stratifié)...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['text_raw'], 
    df['intent'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['intent'] # Garantit l'équilibre des classes
)

print(f"Train Set : {len(X_train_raw)} questions")
print(f"Test Set  : {len(X_test_raw)} questions")

# =============================================================================
# 3. APPLICATION DES DEUX PIPELINES
# =============================================================================
print("\n--- Application des Traitements ---")

# --- BRANCHE 1 : CLASSIQUE (SVM) ---
print("Traitement 'Classique' (SVM)...")
# Création des DataFrames
train_svm = pd.DataFrame({'id': X_train_raw.index, 'text': X_train_raw, 'label': y_train})
test_svm = pd.DataFrame({'id': X_test_raw.index, 'text': X_test_raw, 'label': y_test})

# Application de la fonction
train_svm['text_processed'] = train_svm['text'].apply(preprocess_classical)
test_svm['text_processed'] = test_svm['text'].apply(preprocess_classical)

# --- BRANCHE 2 : AVANCÉE (BERT) ---
print("Traitement 'Avancé' (BERT)...")
# Création des DataFrames
train_bert = pd.DataFrame({'id': X_train_raw.index, 'text': X_train_raw, 'label': y_train})
test_bert = pd.DataFrame({'id': X_test_raw.index, 'text': X_test_raw, 'label': y_test})

# Application de la fonction
train_bert['text_processed'] = train_bert['text'].apply(preprocess_advanced)
test_bert['text_processed'] = test_bert['text'].apply(preprocess_advanced)

# =============================================================================
# 4. DÉMONSTRATION (POUR VOTRE RAPPORT)
# =============================================================================
print("\n" + "="*50)
print("   COMPARAISON AVANT/APRÈS (Exemple)")
print("="*50)
sample_idx = X_train_raw.index[0] # Premier exemple du set d'entrainement
original = X_train_raw.loc[sample_idx]
svm_res = train_svm.loc[sample_idx, 'text_processed']
bert_res = train_bert.loc[sample_idx, 'text_processed']

print(f"📍 Original : {original}")
print(f"🔧 SVM      : {svm_res}")
print(f"🧠 BERT     : {bert_res}")
print("="*50)

# =============================================================================
# 5. SAUVEGARDE ORGANISÉE
# =============================================================================
# Création des dossiers
path_classical = 'Data/processed/classical'
path_advanced = 'Data/processed/advanced'
os.makedirs(path_classical, exist_ok=True)
os.makedirs(path_advanced, exist_ok=True)

# Sauvegarde CSV
train_svm.to_csv(f'{path_classical}/train.csv', index=False)
test_svm.to_csv(f'{path_classical}/test.csv', index=False)

train_bert.to_csv(f'{path_advanced}/train.csv', index=False)
test_bert.to_csv(f'{path_advanced}/test.csv', index=False)

print(f"\n✅ TRAITEMENT TERMINÉ AVEC SUCCÈS !")
print(f"📂 Données SVM  -> {path_classical}")
print(f"📂 Données BERT -> {path_advanced}")