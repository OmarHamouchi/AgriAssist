import sys
import os
import subprocess

def check_and_install_dependencies():
    """
    Checks for required packages and tries to install them if missing.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        import pypdf
        import sentence_transformers
        print("✅ All dependencies are installed.")
        return
    except ImportError:
        print("⚠️ Missing one or more required packages. Attempting installation...")

    # Find requirements.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, '..', 'requirements.txt')
    if not os.path.exists(requirements_path):
        # Fallback for different CWD, e.g. running from project root
        requirements_path = os.path.join('RAG', 'requirements.txt')

    if not os.path.exists(requirements_path):
        print("❌ CRITICAL: Could not find 'requirements.txt'.")
        print("   Please make sure the file is present at 'NewCode/RAG/requirements.txt'.")
        sys.exit(1)

    print(f"   Installing packages from: {requirements_path}")
    try:
        # Use the same python executable that is running this script
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("\n✅ Dependencies installed successfully.")
        print("   Please run the script again to start the indexing process.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Automatic installation failed: {e}")
        print("   Please install the dependencies manually by running this command in your terminal:")
        print(f'      "{sys.executable}" -m pip install -r "{requirements_path}"')

    sys.exit(1) # Exit after attempting installation

check_and_install_dependencies()

# --- Main script starts here ---
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

# --- CONFIGURATION ---
DATA_PATH = "./RAG/data_collection"   # Ton dossier avec les 4 sous-dossiers
DB_PATH = "./chroma_db"           # Là où la base de données sera créée
CLASSES = ["Maladies", "Irrigation", "Pesticides", "Recolte"]

# Modèle d'embedding (traducteur texte -> vecteur)
# On utilise un modèle léger et performant en multilingue (Arabe inclus)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def process_documents():
    # 1. Initialiser le modèle d'embedding
    print(f"🔌 Chargement du modèle d'embedding '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Si la DB existe déjà, on la supprime pour repartir à zéro (mode clean)
    if os.path.exists(DB_PATH):
        print("🗑️  Suppression de l'ancienne base de données pour reconstruction...")
        shutil.rmtree(DB_PATH)

    # 2. Boucle sur chaque classe
    for category in CLASSES:
        print(f"\n🚀 Traitement de la classe : {category.upper()}")
        
        # Chemin vers les fichiers de cette classe
        folder_path = os.path.join(DATA_PATH, category)
        pdf_files = glob.glob(f"{folder_path}/*.pdf")
        
        if not pdf_files:
            print(f"⚠️  Aucun fichier trouvé dans {category}")
            continue

        documents = []
        
        # A. Chargement
        for file_path in pdf_files:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # On ajoute des métadonnées pour tracer la source
                for doc in docs:
                    doc.metadata["source_class"] = category
                    doc.metadata["filename"] = os.path.basename(file_path)
                documents.extend(docs)
                print(f"   📄 Chargé : {os.path.basename(file_path)}")
            except Exception as e:
                print(f"   ❌ Erreur lecture {os.path.basename(file_path)}: {e}")

        # B. Découpage (Chunking)
        # 1000 caractères par morceau, avec 200 de chevauchement pour le contexte
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   ✂️  Découpé en {len(chunks)} morceaux (chunks).")

        # C. Stockage dans ChromaDB (Une collection par classe !)
        if chunks:
            print(f"   💾 Indexation dans ChromaDB (Collection: {category})...")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_PATH,
                collection_name=category  # <--- CRUCIAL : On crée une collection séparée
            )
            print(f"   ✅ Terminé pour {category}.")

    print(f"\n🏆 Indexation terminée ! La base de données est sauvegardée dans '{DB_PATH}'")

if __name__ == "__main__":
    process_documents()