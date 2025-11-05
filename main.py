from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Pinecone setup ===
pc = Pinecone(api_key="pcsk_6e2JzS_PZYtxdavrAwT6JVaBzFVB2gNs4j6DVeu8LQwNf8WfoT3bwmrbQSCWpHHBvWh91P")
index_name = "infosec-policies"
index = pc.Index(index_name)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === TinyLlama setup ===
print("Loading TinyLlama model... (first load may take a minute)")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype=torch.float16
)

# === Retrieve top policy from Pinecone ===
def retrieve_policy(query, top_k=1):
    vector = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = index.query(
        namespace="policies",
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )
    if not results["matches"]:
        return None
    return results["matches"][0]["metadata"]["text"]

# === Generate TinyLlama answer ===
def ask_tinyllama(query):
    context = retrieve_policy(query)
    if not context:
        return "No relevant policy found in the database."

    prompt = (
        f"You are a cybersecurity policy assistant.\n"
        f"Only answer using the text below.\n"
        f"Be concise and factual.\n\n"
        f"--- POLICY TEXT ---\n{context}\n"
        f"-------------------\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response

# === Chat loop ===
print("\n Cybersecurity Policy Chatbot (TinyLlama + Pinecone)")
print("Type your question (or 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    answer = ask_tinyllama(user_input)
    print(f"Bot: {answer}\n")

