import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import pymupdf  # pymupdf
import pandas as pd
import os
import json

FILEPATH_TXT = "/Users/gui/Documents/UFABChatBot/Dados/txt/"
FILEPATH_XLSX = "/Users/gui/Documents/UFABChatBot/Dados/xlsx/"
FILEPATH_ODS = "/Users/gui/Documents/UFABChatBot/Dados/ods/"
FILEPATH_PDF = "/Users/gui/Documents/UFABChatBot/Dados/pdf/"

class Rag():
    def __init__(self):
        self.collection = None
        self.embedder = None
        self.question = ""
        self.client = None
            
    def extract_from_pdf(self, filepath):
        doc = pymupdf.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def extract_from_xlsx(self, filepath):
        df = pd.read_excel(filepath)
        json_data = df.to_json(orient='records', force_ascii=False)
        data = json.loads(json_data)
        return data

    def extract_from_ods(self, filepath):
        df = pd.read_excel(filepath, engine="odf")
        rows = df.astype(str).apply(lambda row: " | ".join(row.values), axis=1)
        return "\n".join(rows)

    def extract_from_txt(self, filepath):
        with open (filepath, "r", encoding="utf-8") as f:
            return f.read()

    def split_into_chunks(self, text, chunk_size=400, overlap=50):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap  # overlap keeps context between chunks
        return chunks

    def json_chunks (self, records, chunk_size = 1):
        chunks = []
        for i in range (0, len(records), chunk_size):
            batch = records[i:i+chunk_size]
            chunks.append(json.dumps(batch, ensure_ascii=False))

        return chunks

    def jsonToSentence (self, json):
        sentences = []
        for j in json:
            new_sentence =  ", ".join([f"{key} é {value}" for key, value in j.items()]) + "."
            sentences.append(new_sentence)

        return sentences


    def read_txt_files(self, folderpath):
        allText = ""
        fileList = os.listdir(folderpath)
        for f in fileList:
            text = self.extract_from_txt(folderpath + f)
            allText += text 

        chunks = self.split_into_chunks(allText)
        return chunks

    def read_xlsx_files(self, folderpath):
        allExcel = []
        fileList = [f for f in os.listdir(folderpath) if f.endswith("xlsx")]
        for f in fileList:
            xlsx = self.extract_from_xlsx(folderpath + f)
            allExcel.extend(xlsx)

        #chunks = self.json_chunks(allExcel)
        chunks = self.jsonToSentence(allExcel)
        return chunks

    def read_ods_files(self, folderpath):
        allOds = ""
        fileList = os.listdir(folderpath)
        for f in folderpath:
            ods = self.extract_from_ods(folderpath + f)
            allOds += ods

        chunks = self.split_into_chunks(allOds)
        return chunks

    def chunking(self):
        chunks = self.read_txt_files(FILEPATH_TXT)
        chunksExcel = self.read_xlsx_files(FILEPATH_XLSX)

        chunks += chunksExcel
        print(f"Total chunks: {len(chunks)}")
        
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-base")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("college_docs")

        for i, chunk in enumerate(chunks):
            embedding = self.embedder.encode(chunk).tolist()
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"chunk_{i}"]
            )

        print("Documents stored in Chroma.")

    def relevantChunks(self, question):
        # embed the question and find the most relevant chunks
        question_embedding = self.embedder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=10  # fetch top 5 most relevant chunks
        )

        print("=== RETRIEVED CHUNKS ===")
        for i, doc in enumerate(results["documents"][0]):
            print(f"--- chunk {i} ---")
            print(doc)  # first 200 chars, just to eyeball relevance
        print("========================")

        retrieved_chunks = "\n\n".join(results["documents"][0])

        return retrieved_chunks

    def ask(self, question):
        retrieved_chunks = self.relevantChunks(question)
        messages = [
            {
                "role": "system",
                "content": "Você é um assistente virtual para estudantes da Universidade Federal do ABC. Responda sempre em português. Use apenas as informações fornecidas no contexto para responder. Se não souber a resposta, diga que não sabe."
            },
            {
                "role": "user",
                "content": f"Use a informação a seguir para responder a pergunta.\n\nContexto:\n{retrieved_chunks}\n\nPergunta: {question}"
            }
        ]

        response = ollama.chat(model="llama3.1:8b", messages=messages)
        return response["message"]["content"]


def main():
    rag = Rag()
    rag.chunking()

if __name__ == "__main__":
    main()