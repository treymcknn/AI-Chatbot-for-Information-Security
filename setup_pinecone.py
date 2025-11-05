from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# === Initialize Pinecone ===
pc = Pinecone(api_key="pcsk_6e2JzS_PZYtxdavrAwT6JVaBzFVB2gNs4j6DVeu8LQwNf8WfoT3bwmrbQSCWpHHBvWh91P")
index_name = "infosec-policies"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 output dimension
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

index = pc.Index(index_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Hardcoded cybersecurity policies ===
policies = [
    {
        "section": "Password Policy",
        "text": "Employees must reset passwords every 90 days and use at least 12 characters including uppercase, lowercase, numbers, and symbols."
    },
    {
        "section": "Device Policy",
        "text": "All personal devices must be approved by the IT department and must run updated antivirus software before accessing company networks."
    },
    {
        "section": "Remote Work Policy",
        "text": "Employees working remotely must connect through the corporate VPN and use company-managed devices to access internal systems."
    },
    {
        "section": "Email Security Policy",
        "text": "Employees must not open attachments or click on links from unknown senders. All suspicious emails should be reported to IT immediately."
    },
    {
        "section": "Incident Response Policy",
        "text": "Any suspected data breach or cybersecurity incident must be reported to the IT security team within one hour of discovery."
    }
]

# === Embed and upload to Pinecone ===
print("Embedding and uploading Infosec policies to Pinecone...")
embeddings = embedder.encode([p["text"] for p in policies], convert_to_numpy=True)

records = [
    {
        "id": f"policy_{i}",
        "values": embeddings[i].tolist(),
        "metadata": {"section": p["section"], "text": p["text"]}
    }
    for i, p in enumerate(policies)
]

index.upsert(vectors=records, namespace="policies")

print(f" Uploaded {len(policies)} hardcoded policies to Pinecone index '{index_name}'.")
