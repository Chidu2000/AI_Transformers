# Installing the necessary libraries
# !pip install datasets==2.14.0
# !pip install torch[cpu]
# !pip install sentence-transformers==2.2.2


from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import os

dataset = load_dataset("multi_news", split="test")

df = dataset.to_pandas().sample(2000, random_state=42)

model = SentenceTransformer("all-MiniLM-L6-v2")

passage_embeddings = list(model.encode(df['summary'].to_list(), show_progress_bar=True))
print(passage_embeddings[0].shape)

query = "Find me some articles about technology and artificial intelligence"

# Find relevant articles
query_embedding = model.encode(query)
similarities = util.cos_sim(query_embedding, passage_embeddings)

top_indices = torch.topk(similarities.flatten(), 3).indices
top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indices]
print(top_relevant_passages)

# utility function
def find_relevant_news(query):
    # Encode the query using the same model
    query_embedding = model.encode(query)

    # Calculate the cosine similarity between the query and passage embeddings
    similarities = util.cos_sim(query_embedding, passage_embeddings)

    # Get the indices of the top 3 most similar passages
    top_indices = torch.topk(similarities.flatten(), 3).indices

    # Retrieve the summaries of the top 3 passages and truncate them to 160 characters
    top_relevant_passages = [df.iloc[x.item()]["summary"][:160] + "..." for x in top_indices]

    return top_relevant_passages

# Example queries to explore
print(find_relevant_news("Natural disasters"))
print(find_relevant_news("Law enforcement and police"))
print(find_relevant_news("Politics, diplomacy and nationalism"))



