import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================================================
# 🔻 CONFIGURATION (MODE HORS LIGNE) 🔻
# =========================================================
MODEL_PATH = "./model"   # Ton dossier MarBERT
DB_PATH = "./chroma_db"  # Ta base de données
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALPHA = 0.15             # Seuil pour le soft routing

# Mapping (MarBERT -> Dossiers ChromaDB)
LABEL_MAPPING = {
    "maladies_et_ravageurs": "Maladies",
    "maladies": "Maladies",
    "irrigation_et_eau": "Irrigation",
    "irrigation": "Irrigation",
    "pesticides_et_securite": "Pesticides",
    "pesticides": "Pesticides",
    "recolte_et_post_recolte": "Recolte",
    "recolte": "Recolte"
}
# =========================================================

class AgriSearchEngine:
    def __init__(self):
        print("🤖 Initialisation du Moteur de Recherche (MODE NO-LLM)...")
        
        # 1. Chargement MarBERT (Le Cerveau de tri)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   🔹 Chargement MarBERT sur {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(self.device)
        except Exception as e:
            print(f"   ❌ Erreur chargement modèle : {e}")
            exit()
        
        # 2. Connexion ChromaDB (La Mémoire)
        print("   🔹 Connexion à la base vectorielle...")
        self.embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(DB_PATH):
            print(f"   ❌ ERREUR : Le dossier '{DB_PATH}' n'existe pas.")
            exit()

    def predict_intent(self, text):
        """Phase 1 : Identification de l'intention (Router)"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 2)
        
        score1 = top_probs[0][0].item()
        score2 = top_probs[0][1].item()
        
        raw_label1 = self.classifier.config.id2label[top_indices[0][0].item()]
        raw_label2 = self.classifier.config.id2label[top_indices[0][1].item()]

        class1 = LABEL_MAPPING.get(raw_label1, raw_label1)
        class2 = LABEL_MAPPING.get(raw_label2, raw_label2)

        diff = score1 - score2
        target_classes = [class1]
        mode = "Direct"
        
        # Si le score est serré, on cherche dans les deux dossiers
        if diff < ALPHA:
            target_classes.append(class2)
            mode = "Hybride"

        return target_classes, mode, score1, raw_label1

    def retrieve_context(self, query, classes):
        """Phase 2 : Extraction des passages pertinents"""
        results_found = []
        for category in classes:
            try:
                # Connexion à la collection spécifique
                vector_store = Chroma(
                    persist_directory=DB_PATH, 
                    embedding_function=self.embedding_fn,
                    collection_name=category
                )
                # On récupère les 2 meilleurs passages par classe
                docs = vector_store.similarity_search(query, k=2)
                
                for doc in docs:
                    results_found.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("filename", "Inconnu"),
                        "category": category
                    })
            except Exception as e:
                print(f"   ⚠️ Collection vide ou erreur : {category}")
                
        return results_found

    def display_results(self, results):
        """Phase 3 : Affichage propre (Sans LLM)"""
        if not results:
            return "❌ Aucun document pertinent trouvé dans la base de connaissances."

        output = "\n🔎 RÉSULTATS TROUVÉS DANS LES DOCUMENTS :\n"
        output += "=" * 60 + "\n"
        
        for i, item in enumerate(results, 1):
            output += f"📄 DOCUMENT {i} (Source: {item['source']} | Dossier: {item['category']})\n"
            output += "-" * 60 + "\n"
            # On nettoie un peu le texte (retours à la ligne excessifs)
            clean_text = item['content'].replace("\n", " ").strip()
            output += f"{clean_text[:500]}..." # On affiche les 500 premiers caractères
            output += "\n" + "=" * 60 + "\n"
            
        return output

    def run(self):
        print("\n🌾 AgriAssist (Mode RECHERCHE SEULE - Pas d'IA Générative) 🌾")
        print("Tape 'q' pour quitter.")
        
        while True:
            query = input("\n❓ Recherche : ")
            if query.lower() in ['q', 'quit']: break
            
            # 1. Routing
            targets, mode, conf, raw = self.predict_intent(query)
            print(f"   🔍 Analyse: '{raw}' -> Recherche dans: {targets} ({conf:.1%})")
            
            # 2. Retrieval
            results = self.retrieve_context(query, targets)
            print(f"   📄 {len(results)} extraits trouvés.")
            
            # 3. Affichage Brut
            print(self.display_results(results))

if __name__ == "__main__":
    engine = AgriSearchEngine()
    engine.run()