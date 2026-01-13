import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PARTIE SPÉCIALE POUR L'AFFICHAGE DE L'ARABE ---
# Si vous n'avez pas ces librairies, les graphiques s'afficheront mais le texte arabe sera peut-être inversé.
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def make_arabic_readable(text):
        reshaped_text = arabic_reshaper.reshape(str(text))
        return get_display(reshaped_text)
except ImportError:
    print("⚠️ Note : Installez 'arabic-reshaper' et 'python-bidi' pour un affichage parfait de l'arabe sur les graphes.")
    def make_arabic_readable(text):
        return text 
# ----------------------------------------------------

# Configuration du style
sns.set(style="whitegrid")

# 1. CHARGEMENT DES DONNÉES
file_path = 'Data/data_staging/Arabic_Question.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("✅ Dataset chargé avec succès !")
    print(f"Taille initiale : {df.shape[0]} lignes")
    print("\nAperçu des colonnes :", df.columns.tolist())
else:
    print(f"❌ Erreur : Le fichier n'a pas été trouvé au chemin : {file_path}")
    # Arrêt du script ici si pas de fichier
    exit()

# ---------------------------------------------------------
# 2. NETTOYAGE SANITAIRE (AVANT LE SPLIT)
# ---------------------------------------------------------
print("\n--- DÉBUT DU NETTOYAGE SANITAIRE ---")

# A. Suppression des doublons exacts sur la colonne 'text_raw'
initial_count = len(df)
df = df.drop_duplicates(subset=['text_raw'], keep='first')
print(f"Doublons supprimés : {initial_count - len(df)}")

# B. Suppression des valeurs nulles
df = df.dropna(subset=['text_raw', 'intent'])

# C. Suppression des questions trop courtes (< 2 mots)
# En arabe, split() fonctionne bien pour compter les "mots graphiques" (séparés par espace)
df['word_count'] = df['text_raw'].apply(lambda x: len(str(x).split()))

# Filtre : On garde si > 1 mot (car 1 mot est rarement une question complète)
short_questions = df[df['word_count'] <= 1]
df = df[df['word_count'] > 1] 

print(f"Questions trop courtes (<= 1 mot) supprimées : {len(short_questions)}")
print(f"Taille finale après nettoyage : {df.shape[0]} lignes")

# ---------------------------------------------------------
# 3. ANALYSE EXPLORATOIRE (EDA)
# ---------------------------------------------------------
print("\n--- DÉBUT DE L'ANALYSE (EDA) ---")

# Préparation des labels arabes pour le graphique
df['intent_display'] = df['intent'].apply(make_arabic_readable)

# Figure 1 : Distribution des Classes (Intent)
plt.figure(figsize=(12, 6))
# On utilise la colonne intent_display pour que l'arabe soit lisible si les intents sont en arabe
ax = sns.countplot(x='intent', data=df, palette='viridis', order=df['intent'].value_counts().index)

plt.title('Distribution des questions par classe (Intent)')
plt.xlabel('Classes (Intents)')
plt.ylabel('Nombre de questions')
plt.xticks(rotation=45)

# Afficher les nombres sur les barres
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout() # Pour éviter que les labels soient coupés
plt.show()

# Figure 2 : Distribution de la longueur des questions
plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], bins=30, kde=True, color='green')
plt.title('Distribution de la longueur des questions (Nombre de mots)')
plt.xlabel('Nombre de mots')
plt.ylabel('Fréquence')
plt.show()

# Stats textuelles
print("\nStatistiques sur la longueur des questions (mots) :")
print(df['word_count'].describe())

# ---------------------------------------------------------
# 4. SAUVEGARDE
# ---------------------------------------------------------
# On sauvegarde dans un nouveau fichier pour ne pas écraser l'original
output_path = 'Data/data_cleaned/Arabic_Question_Cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Fichier nettoyé sauvegardé sous : {output_path}")