# =========================================================
# Local RAG Chatbot using TinyLlama + FAISS (fully local)
# =========================================================

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

class ourlogic:
    # ----------------------------
    # Load the TinyLlama model
    # ----------------------------
    print("Loading TinyLlama model... (this may take a minute on first run)")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto"  # Automatically chooses CPU or GPU if available
    )

    # ----------------------------
    # Load company policies
    # ----------------------------
    with open("company_policies.json", "r", encoding="utf-8") as f:
        policies = json.load(f)

    policy_texts = [p["text"] for p in policies]

    # ----------------------------
    # Create embeddings
    # ----------------------------
    print("Generating embeddings for policy documents...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast model
    policy_embeddings = embedder.encode(policy_texts, convert_to_numpy=True)

    # ----------------------------
    # Store in FAISS index
    # ----------------------------
    dimension = policy_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(policy_embeddings)

    print(f"Stored {len(policies)} documents in FAISS vector index.")

    # ----------------------------
    # retrieval helper
    # ----------------------------
    def retrieve_context(query, k=2):
        """Find top-k similar policy texts for the user's question."""
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k)
        contexts = [policy_texts[i] for i in indices[0]]
        return "\n".join(contexts)

    # ----------------------------
    # Chat function
    # ----------------------------
    def ask_llama(query, similarity_threshold=0.50):
        """
        Generate a response using retrieved company policies.
        If no relevant policy matches the query, return a fallback message.
        """

        # Retrieve context from FAISS
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, 1)  # top 1 match
        best_distance = distances[0][0]
        best_index = indices[0][0]

        # Convert distance (smaller = closer) to similarity
        # Normalize to 0–1 range for readability
        similarity = 1 / (1 + best_distance)

        # If no sufficiently close match, return placeholder
        if similarity < similarity_threshold:
            return "No such record found in company policy."

        # Otherwise, retrieve the best policy text
        context = policy_texts[best_index]

        # Structured, minimal prompt
        prompt = (
            f"You are a company policy assistant.\n"
            f"Only answer using the information below.\n"
            f"Be brief and factual (one sentence max).\n"
            f"Do not restate the question.\n\n"
            f"--- Company Policy ---\n{context}\n"
            f"----------------------\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        # Tokenize + generate
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and clean
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[-1]
        response = response.strip()

        return response


        # Decode and clean
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip everything before or after the true answer portion
        if "Answer:" in response:
            response = response.split("Answer:")[-1]
        if "Question:" in response:
            response = response.split("Question:")[-1]

        # Final cleanup
        response = response.strip()
        return response


    # ----------------------------
    # interactive chat
    # ----------------------------
    #print("\n✅ RAG system ready! Type your question below.")
    #print("Type 'exit' to quit.\n")

    #while True:
    #    query = input("You: ")
    #    if query.lower() in ["exit", "quit"]:
    #        print("Goodbye!")
    #        break

    #    answer = ask_llama(query)
    #    print(f"Answer: {answer}\n")
