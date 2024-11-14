import numpy as np
import redis

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

### store embeddings and file paths

# Sample data: embeddings and file paths
data = {
    "response1": {
        "text": "hello",
        "embedding": np.array(
            [0.1, 0.2, 0.3]
        ),  # embeddings will be the person's talking
        "filepath": "/path/to/audio1.wav",
    },
    "response2": {
        "text": "hello",
        "embedding": np.array([0.4, 0.5, 0.6]),
        "filepath": "/path/to/audio2.wav",
    },
    "response3": {
        "text": "hello",
        "embedding": np.array([0.1, 0.2, 0.25]),
        "filepath": "/path/to/audio3.wav",
    },
}

# Store each embedding and filepath in Redis
for key, value in data.items():
    # Store embedding as a byte array
    r.hset(
        key,
        mapping={
            "text": value["text"],
            "embedding": value["embedding"].tobytes(),
            "filepath": value["filepath"],
        },
    )


### retrieve embeddings and file paths
def get_audio_data(key):
    """Retrieve embedding and filepath from Redis."""
    data = r.hgetall(key)
    if data:
        # Convert embedding back from bytes to numpy array
        embedding = np.frombuffer(data[b"embedding"], dtype=np.float64)
        filepath = data[b"filepath"].decode("utf-8")
        return embedding, filepath
    return None, None


# Retrieve data for a specific audio file
embedding, filepath = get_audio_data("audio1")
print(f"Embedding: {embedding}, Filepath: {filepath}")


### import faiss
# Prepare embeddings for FAISS
embedding_matrix = np.array(
    [np.frombuffer(r.hget(key, "embedding"), dtype=np.float64) for key in data.keys()]
).astype("float32")

# Create a FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance
index.add(embedding_matrix)  # Add embeddings to the index


###
def search_similar_embeddings(query_embedding, top_k=5):
    distances, indices = index.search(
        np.array([query_embedding], dtype="float32"), top_k
    )
    results = []
    for i in range(top_k):
        key = list(data.keys())[indices[0][i]]
        embedding, filepath = get_audio_data(key)
        results.append(
            (key, embedding, filepath, distances[0][i])
        )  # Include distance for reference
    return results


# Example query
query_embedding = np.array([0.1, 0.2, 0.3], dtype="float32")
similar_embeddings = search_similar_embeddings(query_embedding)
print(similar_embeddings)
