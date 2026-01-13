import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION POUR L'AFFICHAGE DE L'ARABE SUR LES GRAPHIQUES ---
# Cette partie s'assure que les lettres ne sont pas inversées dans les figures
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def make_arabic_readable(text):
        reshaped_text = arabic_reshaper.reshape(str(text))
        return get_display(reshaped_text)
except ImportError:
    print("⚠️ Attention : Installez 'arabic-reshaper' et 'python-bidi' pour un affichage parfait.")
    def make_arabic_readable(text): return text 

# Configuration du style des graphiques
sns.set(style="whitegrid")

# =============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# =============================================================================
file_path = 'Data/data_staging/Arabic_Question.csv'

if not os.path.exists(file_path):
    print(f"❌ Erreur : Fichier introuvable à {file_path}")
    exit()

df = pd.read_csv(file_path)
print("✅ Dataset chargé avec succès !")
print(f"Taille initiale : {df.shape[0]} lignes")

# =============================================================================
# ÉTAPE 2 : NETTOYAGE SANITAIRE
# =============================================================================
print("\n--- DÉBUT DU NETTOYAGE SANITAIRE ---")

# 1. Suppression des doublons exacts
initial_count = len(df)
df = df.drop_duplicates(subset=['text_raw'], keep='first')
print(f"Doublons supprimés : {initial_count - len(df)}")

# 2. Suppression des lignes vides
df = df.dropna(subset=['text_raw', 'intent'])

# 3. Suppression des questions trop courtes (<= 1 mot)
df['word_count'] = df['text_raw'].apply(lambda x: len(str(x).split()))
short_questions = df[df['word_count'] <= 1]
df = df[df['word_count'] > 1] 
print(f"Questions trop courtes (<= 1 mot) supprimées : {len(short_questions)}")

print(f"Taille finale après nettoyage : {df.shape[0]} lignes")

# =============================================================================
# ÉTAPE 3 : ANALYSE EXPLORATOIRE (EDA) & VISUALISATION
# =============================================================================
print("\n--- DÉBUT DE LA VISUALISATION (EDA) ---")

# Préparation des labels arabes lisibles
df['intent_display'] = df['intent'].apply(make_arabic_readable)

# --- GRAPHIQUE 1 : DIAGRAMME CIRCULAIRE (PIE CHART) ---
plt.figure(figsize=(10, 8))

# Calcul des comptes par classe
counts = df['intent'].value_counts()
# Labels lisibles pour le camembert
labels = [make_arabic_readable(l) for l in counts.index]

# Fonction personnalisée pour afficher le % et le nombre exact (ex: 25% (1200))
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
    return my_autopct

plt.pie(counts, 
        labels=labels, 
        autopct=make_autopct(counts), 
        startangle=140, 
        colors=sns.color_palette('pastel'),
        textprops={'fontsize': 12}) # Taille du texte

plt.title(make_arabic_readable("Répartition des Données par Classe (Intent)"))
plt.axis('equal') # Pour que le camembert soit bien rond
plt.show()

# --- GRAPHIQUE 2 : DISTRIBUTION DES LONGUEURS (HISTOGRAMME) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], bins=30, kde=True, color='green')
plt.title('Distribution de la longueur des questions (Nombre de mots)')
plt.xlabel('Nombre de mots')
plt.ylabel('Fréquence')
plt.show()

# --- STATISTIQUES TEXTUELLES ---
print("\nStatistiques sur la longueur des questions (mots) :")
print(df['word_count'].describe())

# =============================================================================
# ÉTAPE 4 : SAUVEGARDE
# =============================================================================
output_dir = 'Data/data_cleaned'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'Arabic_Question_Cleaned.csv')
df.to_csv(output_path, index=False)
print(f"\n✅ Fichier nettoyé sauvegardé sous : {output_path}")