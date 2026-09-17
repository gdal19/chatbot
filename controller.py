import ollama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service import Rag
from DTO.promptResponse import PromptResponse
from DTO.promptRequest import PromptRequest

app = FastAPI()
rag = Rag()
rag.chunking()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prompt")
async def sendPrompt(request: PromptRequest):
    response = rag.ask(request.prompt)
    final_response = PromptResponse(text = response)
    return final_response

