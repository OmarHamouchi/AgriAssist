import os
import glob
from pypdf import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

# --- CONFIGURATION ---
BASE_PATH = "./RAG/data_collection"  # Le dossier contenant tes 4 sous-dossiers
CLASSES = ["Maladies", "Irrigation", "Pesticides", "Recolte"]

# Liste pour stocker les données
dataset = []

print("📂 Démarrage de l'analyse des PDF...")

# 1. LECTURE DES FICHIERS
for category in CLASSES:
    folder_path = os.path.join(BASE_PATH, category)
    files = glob.glob(f"{folder_path}/*.pdf")
    
    print(f"   🔹 Classe '{category}' : {len(files)} fichiers trouvés.")
    
    for file_path in files:
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            
            # On stocke les infos
            dataset.append({
                "category": category,
                "filename": os.path.basename(file_path),
                "char_count": len(text),
                "word_count": len(text.split()),
                "content": text # On garde le texte pour le WordCloud
            })
        except Exception as e:
            print(f"   ⚠️ Erreur lecture {file_path}: {e}")

# Création d'un DataFrame (Tableau)
df = pd.read_csv(pd.compat.StringIO("")) if not dataset else pd.DataFrame(dataset)

if df.empty:
    print("❌ Aucun donnée extraite. Vérifie tes dossiers !")
    exit()

print(f"\n✅ Extraction terminée. Total : {len(df)} documents traités.")
print("-" * 50)

# 2. VISUALISATION 1 : Quantité de texte par Classe
# Cela permet de voir si une classe est "pauvre" en information
plt.figure(figsize=(10, 6))
sns.barplot(x="category", y="word_count", data=df, estimator="sum", errorbar=None, palette="viridis")
plt.title("Volume total de mots par classe (Richesse du corpus)")
plt.ylabel("Nombre total de mots")
plt.xlabel("Classe")
plt.show()

# 3. VISUALISATION 2 : Distribution de la taille des documents
# Pour voir si tu as des documents très petits ou très gros
plt.figure(figsize=(10, 6))
sns.boxplot(x="category", y="word_count", data=df, palette="Set2")
plt.title("Distribution de la taille des documents (Mots par PDF)")
plt.ylabel("Nombre de mots")
plt.show()

# 4. VISUALISATION 3 : WordCloud (Nuage de mots en Arabe)
# Fonction pour corriger l'affichage arabe
def make_arabic_wordcloud(text_list, title):
    # Concaténer tout le texte
    full_text = " ".join(text_list)
    
    # Reshape pour l'arabe (lettres attachées + sens RTL)
    reshaped_text = arabic_reshaper.reshape(full_text)
    bidi_text = get_display(reshaped_text)
    
    # Création du WordCloud
    # Note: 'arial' est souvent dispo, sinon mettre le chemin vers une police arabe .ttf
    wc = WordCloud(
        font_path='arial', # ⚠️ Si ça plante, télécharge une police arabe (ex: Amir.ttf) et mets le chemin ici
        width=800, height=400, 
        background_color='white',
        stopwords={} # Tu pourras ajouter des stopwords arabes ici plus tard
    ).generate(bidi_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Mots fréquents : {title}")
    plt.show()

# Générer un WordCloud pour la classe 'maladies' (exemple)
print("☁️  Génération du WordCloud pour 'maladies'...")
maladies_text = df[df['category'] == 'maladies']['content'].tolist()
if maladies_text:
    try:
        make_arabic_wordcloud(maladies_text, "Classe Maladies")
    except Exception as e:
        print(f"⚠️ Impossible d'afficher le WordCloud (problème de police probable) : {e}")