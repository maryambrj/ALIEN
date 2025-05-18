import os
import json
from sklearn.metrics import accuracy_score, f1_score
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()  # Load from .env file
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") 

RELATION_LABELS = ['no_relation': 0, 'headquartered_in': 1, 'formed_in': 2, 'title': 3, 'shares_of': 4, 'loss_of': 5, 'acquired_on': 6, 'agreement_with': 7, 'operations_in': 8, 'subsidiary_of': 9,'employee_of': 10, 'attended': 11, 'cost_of': 12, 'acquired_by': 13, 'member_of': 14, 'profit_of': 15, 'revenue_of': 16, 'founder_of': 17, 'formed_on': 18]

# === Custom DeepSeek Wrapper ===
class DeepSeekChat(BaseChatModel):
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", temperature: float = 0.0):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    def _call(self, messages, **kwargs) -> str:
        prompt = messages[-1].content
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def __call__(self, messages, **kwargs):
        content = self._call(messages)
        return content

# === Load corpus and embed ===
loader = TextLoader("corpus.txt")  # Replace with your actual path
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = DeepSeekChat(api_key=DEEPSEEK_API_KEY, model_name="deepseek-chat")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# === Load annotated dataset ===
with open("dataset.json", "r") as f:  # Replace with your actual path
    data = json.load(f)

true_labels = []
predicted_labels = []

for item in data:
    sentence = item["sentence"]
    entities = item["entities"]  # Assuming this contains entity information
    true_label = item["label"]

    label_list_text = "\\n".join(RELATION_LABELS)

    prompt = f"""
    You are an expert in financial relation extraction.
    
    Given the following sentence:
    "{sentence}"
    
    Your task is to identify the correct relationship between **Entity 1** and **Entity 2**, where:
    - Entity 1 is the first tagged entity in the sentence.
    - Entity 2 is the second tagged entity.
    - The relationship should be interpreted as:  
      **"Entity 1 is <RELATION> of Entity 2"**,  
      where <RELATION> is selected from the predefined label list below.
    
    Label set:
    {label_list_text}
    
    If none of the relationships apply, respond with: no_relation.
    
    Respond with exactly one label from the list.
    """

    result = qa_chain.run(prompt)
    predicted_label = result.strip()
    if predicted_label not in RELATION_LABELS:
        predicted_label = "no_relation"


    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

acc = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1 Score: {f1:.4f}")
