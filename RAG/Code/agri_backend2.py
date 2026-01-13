# agri_backend.py
import os
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
load_dotenv()
MODEL_PATH = "./model" 
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALPHA = 0.15

LABEL_MAPPING = {
    "maladies_et_ravageurs": "Maladies", "maladies": "Maladies",
    "irrigation_et_eau": "Irrigation", "irrigation": "Irrigation",
    "pesticides_et_securite": "Pesticides", "pesticides": "Pesticides",
    "recolte_et_post_recolte": "Recolte", "recolte": "Recolte"
}
# ---------------------

class SmartAgriAgent:
    def __init__(self):
        self.init_logs = []
        self.log_init("🤖 Initialisation du backend AgriAssist...")
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "AgriAssist")

        if self.api_key:
            self.has_llm = True
            self.log_init(f"🔹 API LLM Configurée : {self.model_name}")
        else:
            self.has_llm = False
            self.log_init("⚠️ Pas de clé API OpenRouter.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_init(f"🔹 Chargement MarBERT sur {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(self.device)
        except Exception as e:
            self.log_init(f"❌ Erreur fatale MarBERT : {e}")
            raise e

        self.log_init("🔹 Connexion à ChromaDB...")
        self.embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(DB_PATH):
             self.log_init(f"❌ ERREUR : Dossier DB introuvable.")

        self.log_init("✅ Backend prêt !")

    def log_init(self, message):
        print(message)
        self.init_logs.append(message)

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 2)
        score1, score2 = top_probs[0][0].item(), top_probs[0][1].item()
        raw1 = self.classifier.config.id2label[top_indices[0][0].item()]
        raw2 = self.classifier.config.id2label[top_indices[0][1].item()]
        
        class1 = LABEL_MAPPING.get(raw1, raw1)
        class2 = LABEL_MAPPING.get(raw2, raw2)
        
        diff = score1 - score2
        targets = [class1]
        mode = "Direct"
        if diff < ALPHA:
            targets.append(class2)
            mode = "Hybride (Ambiguïté)"

        return targets, mode, score1, raw1, score2, raw2

    def retrieve_context(self, query, classes):
        combined_context = []
        sources = []
        for category in classes:
            try:
                vector_store = Chroma(persist_directory=DB_PATH, embedding_function=self.embedding_fn, collection_name=category)
                results = vector_store.similarity_search(query, k=2)
                for doc in results:
                    combined_context.append(doc.page_content)
                    sources.append(doc.metadata.get('filename', 'Inconnu'))
            except Exception:
                pass
        return combined_context, list(set(sources))

    def generate_response(self, query, context_list):
        if not self.has_llm: return "عذراً، خدمة الذكاء الاصطناعي غير متصلة. (Pas de LLM)"
        
        # PROMPT STRICT ARABE
        system_msg = """
        أنت مساعد زراعي ذكي يسمى AgriAssist.
        مهمتك هي مساعدة المزارعين والإجابة على أسئلتهم بدقة.
        القواعد:
        1. يجب أن تكون الإجابة باللغة العربية فقط.
        2. اعتمد في إجابتك فقط على المعلومات التقنية المقدمة أدناه.
        3. كن لطيفاً ومفيداً.
        """
        user_msg = f"المعلومات التقنية:\n{' '.join(context_list)}\n\nالسؤال: {query}"
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "HTTP-Referer": self.site_url, "X-Title": self.app_name}
        payload = {"model": self.model_name, "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}], "temperature": 0.3}

        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=45)
            if r.status_code == 200: return r.json()['choices'][0]['message']['content']
            return f"Error API: {r.text}"
        except Exception as e: return f"Connection Error: {e}"

    def process_query(self, query):
        logs = []
        logs.append(f"▶️ INPUT: '{query}'")

        targets, mode, conf1, raw1, conf2, raw2 = self.predict_intent(query)
        logs.append(f"🔍 CLASSIFICATION:")
        logs.append(f"  ├─ Intent 1: '{raw1}' ({conf1:.1%})")
        if mode.startswith("Hybride"):
             logs.append(f"  ├─ Intent 2: '{raw2}' ({conf2:.1%})")
        logs.append(f"  └─ Routing: {mode} -> {targets}")

        context, sources = self.retrieve_context(query, targets)
        logs.append(f"📂 RAG SYSTEM:")
        logs.append(f"  ├─ Docs Retrieved: {len(context)}")
        logs.append(f"  └─ Sources: {sources}")

        logs.append("🧠 LLM GENERATION...")
        answer = self.generate_response(query, context)
        logs.append("✅ COMPLETE.")

        return answer, logs