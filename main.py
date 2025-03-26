from fastapi import FastAPI, Request
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
import redis

app = FastAPI()

# Redis setup
r = redis.Redis(host="redis", port=6379, decode_responses=True)

def get_chain(user_id: str):
    message_history = RedisChatMessageHistory(
        session_id=user_id,
        url="redis://redis:6379"
    )
    memory = ConversationBufferMemory(
        memory_key="history",
        chat_memory=message_history,
        return_messages=True
    )
    llm = Ollama(model="tinyllama", base_url="http://host.docker.internal:11434")
    return ConversationChain(llm=llm, memory=memory, verbose=True)

@app.post("/chat/")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    message = data.get("message", "")

    chain = get_chain(user_id)
    response = chain.predict(input=message)

    return {"response": response}
