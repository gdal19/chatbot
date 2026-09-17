import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # pymupdf
import pandas as pd
import os
from fastapi import FastAPI
import json

FILEPATH_TXT = "/Users/gui/Documents/UFABChatBot/Dados/txt/"
FILEPATH_XLSX = "/Users/gui/Documents/UFABChatBot/Dados/xlsx/"
FILEPATH_ODS = "/Users/gui/Documents/UFABChatBot/Dados/ods/"
FILEPATH_PDF = "/Users/gui/Documents/UFABChatBot/Dados/pdf/"

# --- STEP 1: Extract text from your file ---

def extract_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_from_xlsx(filepath):
    df = pd.read_excel(filepath)
    # converts each row into a readable string
    # rows = df.fillna("").astype(str).apply(lambda row: " | ".join(row.values), axis=1)
    json_data = df.to_json(orient='records', force_ascii=False)
    data = json.loads(json_data)
    return data

def extract_from_ods(filepath):
    df = pd.read_excel(filepath, engine="odf")
    rows = df.astype(str).apply(lambda row: " | ".join(row.values), axis=1)
    return "\n".join(rows)

def extract_from_txt(filepath):
    with open (filepath, "r", encoding="utf-8") as f:
        return f.read()

def split_into_chunks(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap keeps context between chunks
    return chunks

def json_chunks (records, chunk_size = 23):
    chunks = []
    for i in range (0, len(records), chunk_size):
        batch = records[i:i+chunk_size]
        chunks.append(json.dumps(batch, ensure_ascii=False))

    return chunks


def read_txt_files(folderpath):
    allText = ""
    fileList = os.listdir(folderpath)
    for f in fileList:
        text = extract_from_txt(folderpath + f)
        allText += text 

    chunks = split_into_chunks(allText)
    return chunks

def read_xlsx_files(folderpath):
    allExcel = []
    fileList = [f for f in os.listdir(folderpath) if f.endswith("xlsx")]
    for f in fileList:
        xlsx = extract_from_xlsx(folderpath + f)
        allExcel.extend(xlsx)

    #records = json.loads(allExcel)
    chunks = json_chunks(allExcel)
    return chunks

def read_ods_files(folderpath):
    allOds = ""
    fileList = os.listdir(folderpath)
    for f in folderpath:
        ods = extract_from_ods(folderpath + f)
        allOds += ods

    chunks = split_into_chunks(allOds)
    return chunks

# text = extract_from_txt("/Users/gui/Documents/UFABChatBot/Dados/txt/calendario_academico_2026_rag.txt")
# chunks = [line.strip() for line in text.split("\n") if line.strip()]
# or: text = extract_from_xlsx("schedule.xlsx")
# or: text = extract_from_ods("schedule.ods")

#chunks = split_into_chunks(text)


# --- STEP 4: Query function ---

def ask(question):
    chunks = read_txt_files(FILEPATH_TXT)
    chunksExcel = read_xlsx_files(FILEPATH_XLSX)

    chunks += chunksExcel
    print(f"Total chunks: {len(chunks)}")
    
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection("college_docs")

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )

    print("Documents stored in Chroma.")
    # embed the question and find the most relevant chunks
    question_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10  # fetch top 3 most relevant chunks
    )

    retrieved_chunks = "\n\n".join(results["documents"][0])

    # build the prompt with the retrieved context injected
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


# --- STEP 5: Test it ---

app = FastAPI()

@app.get("/prompt")
async def sendPrompt(prompt : str):
    return ask(prompt)


def main ():
    question = "Qual a nota de corte da disciplina Categorização de materiais diurno?"
    answer = ask(question)
    print(f"Q: {question}")
    print(f"A: {answer}")

if __name__ == '__main__':
    main()