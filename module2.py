


# Import the Pinecone library
from pinecone import Pinecone

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_6e2JzS_PZYtxdavrAwT6JVaBzFVB2gNs4j6DVeu8LQwNf8WfoT3bwmrbQSCWpHHBvWh91P")

index_name = "developer-quickstart-py"

  # Target the index
dense_index = pc.Index(index_name)

# Define the query
query = "talk about cyber policies"

# Search the dense index
results = dense_index.search(
    namespace="testing-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    }
)

# Print the results
for hit in results['result']['hits']:
        print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")



