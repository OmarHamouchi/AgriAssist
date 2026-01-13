import os
import requests
import json
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================================================
# 🔻 CONFIGURATION 🔻
# =========================================================
load_dotenv() # Chargement du .env

MODEL_PATH = "./model"  # ⚠️ Vérifie que ton dossier s'appelle bien "model" ou modifie ici
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALPHA = 0.15 # Seuil pour le soft routing

# Mapping exact (MarBERT -> Dossiers ChromaDB)
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

class SmartAgriAgent:
    def __init__(self):
        print("🤖 Initialisation de l'Agent AgriAssist (Version Llama/Requests)...")
        
        # 1. Configuration API (Depuis .env)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "AgriAssist")

        if not self.api_key:
            print("⚠️ ATTENTION : Pas de clé API trouvée dans .env !")
            self.has_llm = False
        else:
            self.has_llm = True
            print(f"   🔹 API Configurée avec modèle : {self.model_name}")

        # 2. Chargement MarBERT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   🔹 Chargement MarBERT sur {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(self.device)
        
        # 3. Connexion ChromaDB
        print("   🔹 Connexion à ChromaDB...")
        self.embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(DB_PATH):
            print(f"   ❌ ERREUR CRITIQUE : Le dossier '{DB_PATH}' n'existe pas. Lance l'indexation d'abord.")

    def predict_intent(self, text):
        """Phase 1 : Classification de la question"""
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
        
        # Logique de Soft Routing (Fallback)
        if diff < ALPHA:
            target_classes.append(class2)
            mode = "Hybride"

        return target_classes, mode, score1, raw_label1

    def retrieve_context(self, query, classes):
        """Phase 2 : Recherche dans la base vectorielle"""
        combined_context = []
        for category in classes:
            try:
                # On cible la collection spécifique
                vector_store = Chroma(
                    persist_directory=DB_PATH, 
                    embedding_function=self.embedding_fn,
                    collection_name=category
                )
                results = vector_store.similarity_search(query, k=3)
                for doc in results:
                    combined_context.append(doc.page_content)     
            except Exception as e:
                print(f"   ⚠️ Erreur lecture collection '{category}': {e}")
        return combined_context

    def generate_response(self, query, context_list):
        """Phase 3 : Génération via OpenRouter (Requests)"""
        if not self.has_llm:
            return "Pas de clé API configurée. Documents trouvés :\n" + "\n".join(context_list[:2])

        if not context_list:
            return "Je n'ai trouvé aucune information pertinente dans mes documents."

        # Construction du Prompt
        system_msg = "Tu es un expert agricole nommé AgriAssist. Tu dois répondre en Arabe. Base ta réponse UNIQUEMENT sur le contexte technique fourni ci-dessous."
        user_msg = f"CONTEXTE TECHNIQUE :\n{' '.join(context_list)}\n\nQUESTION : {query}"

        # Configuration de la requête HTTP (Exactement comme ton exemple)
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.3 # Plus bas = plus factuel
        }

        try:
            # Envoi de la requête POST
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            
            if r.status_code == 200:
                response_json = r.json()
                return response_json['choices'][0]['message']['content']
            else:
                return f"Erreur OpenRouter ({r.status_code}) : {r.text}"
                
        except Exception as e:
            return f"Erreur Technique : {str(e)}"

    def run(self):
        print("\n🌾 AgriAssist Prêt (Tape 'q' pour quitter) 🌾")
        while True:
            query = input("\n❓ Question : ")
            if query.lower() in ['q', 'quit']: break
            
            # 1. Routing
            targets, mode, conf, raw = self.predict_intent(query)
            print(f"   🔍 Intent: '{raw}' -> Dossier: {targets} ({conf:.1%})")
            
            # 2. Retrieval
            context = self.retrieve_context(query, targets)
            print(f"   📄 Docs trouvés : {len(context)}")
            
            # 3. Generation
            print("   🤖 Réponse :")
            print(self.generate_response(query, context))
            print("-" * 50)

if __name__ == "__main__":
    agent = SmartAgriAgent()
    agent.run()