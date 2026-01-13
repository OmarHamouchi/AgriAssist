import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier  # <--- NOUVEAU
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
import os

# --- GESTION ARABE ---
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def make_arabic_readable(text):
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError:
    def make_arabic_readable(text): return text

# =============================================================================
# 1. CHARGEMENT DES DONNÉES (CLASSIQUE)
# =============================================================================
train_path = 'Data/processed/classical/train.csv'
test_path = 'Data/processed/classical/test.csv'

if not os.path.exists(train_path):
    print("❌ Données introuvables. Vérifiez le dossier Data/processed/classical/")
    exit()

print("Chargement des données...")
train_df = pd.read_csv(train_path).fillna("")
test_df = pd.read_csv(test_path).fillna("")

X_train, y_train = train_df['text_processed'], train_df['label']
X_test, y_test = test_df['text_processed'], test_df['label']

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# =============================================================================
# 2. DÉFINITION DES 3 MODÈLES
# =============================================================================
models = {
    'SVM (LinearSVC)': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LinearSVC(random_state=42, tol=1e-5))
    ]),
    'Naive Bayes (MultinomialNB)': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ]),
    'Random Forest': Pipeline([ # <--- NOUVEAU
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        # n_estimators=100 : On crée 100 arbres
        # n_jobs=-1 : Utilise tous les cœurs du processeur pour aller plus vite
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}

results = []

# =============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT & COMPARAISON
# =============================================================================
print("\n--- DÉBUT DU BENCHMARK (3 MODÈLES) ---")

for name, pipeline in models.items():
    print(f"\n🚀 Entraînement de : {name} ...")
    start_time = time.time()
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Prédiction
    y_pred = pipeline.predict(X_test)
    
    # Mesures
    acc = accuracy_score(y_test, y_pred)
    duration = time.time() - start_time
    
    print(f"   ✅ Accuracy : {acc:.2%}")
    print(f"   ⏱️ Temps : {duration:.4f} sec")
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Time': duration,
        'Pipeline': pipeline
    })

# =============================================================================
# 4. VISUALISATION COMPARATIVE (BAR CHART)
# =============================================================================
results_df = pd.DataFrame(results)

# Graphique de Précision
plt.figure(figsize=(12, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='magma')
plt.title('Comparaison de Précision (Accuracy)')
plt.xlim(0.8, 1.0) # Zoom entre 80% et 100%

# Afficher les pourcentages sur les barres
for index, row in results_df.iterrows():
    plt.text(row.Accuracy, index, f' {row.Accuracy:.2%}', color='black', ha="left", va="center", fontweight='bold')

plt.tight_layout()
plt.show()

# Graphique de Temps (Optionnel mais intéressant pour Random Forest qui est souvent lent)
plt.figure(figsize=(12, 4))
sns.barplot(x='Time', y='Model', data=results_df, palette='coolwarm')
plt.title("Comparaison du Temps d'Entraînement (Secondes)")
plt.xlabel("Secondes")
plt.tight_layout()
plt.show()

# =============================================================================
# 5. COURBES D'APPRENTISSAGE (LEARNING CURVES) - 3 PLOTS
# =============================================================================
print("\n--- Génération des Learning Curves (Cela peut prendre un peu de temps...) ---")

# On crée 3 sous-graphiques côte à côte
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for i, model_data in enumerate(results):
    name = model_data['Model']
    pipeline = model_data['Pipeline']
    ax = axes[i]
    
    print(f"Calcul courbe pour {name}...")
    
    # Calcul Learning Curve
    # cv=3 pour aller plus vite, n_jobs=-1 pour paralléliser
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X_train, y_train, 
        cv=3, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        scoring='accuracy'
    )
    
    # Moyennes et Ecarts-types
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Tracé
    ax.plot(train_sizes, train_mean, 'o-', color="r", label="Entraînement")
    ax.plot(train_sizes, test_mean, 'o-', color="g", label="Validation (Cross-Val)")
    
    # Zone d'ombre (écart-type)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    ax.set_title(make_arabic_readable(name), fontsize=14)
    ax.set_xlabel("Nombre d'échantillons")
    if i == 0: ax.set_ylabel("Accuracy")
    ax.legend(loc="best")
    ax.grid(True)

plt.tight_layout()
plt.show()

print("\n✅ Benchmark complet terminé !")