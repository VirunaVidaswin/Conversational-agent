import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Sentence-BERT set up for document retrieval
# Load the pre-trained Sentence-BERT model for creating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample knowledge base (documents)
documents = [
    "AI stands for Artificial Intelligence. It refers to the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI, which enables systems to learn from data without explicit programming.",
    "Natural Language Processing is a field of AI that focuses on the interaction between computers and human languages.",
    "Deep Learning is a subset of Machine Learning that uses neural networks to learn from large amounts of data.",
    "Reinforcement Learning is a type of Machine Learning where an agent learns by interacting with an environment."
]

# Convert documents to embeddings
document_embeddings = model.encode(documents)

# FAISS set up for document retrieval
# FAISS index for efficient similarity search
dimension = document_embeddings.shape[1]  # The size of the embeddings (vector dimension)
index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance
index.add(np.array(document_embeddings))  # Add document embeddings to the index

# Function to retrieve documents based on user query
def retrieve_documents(query, top_k=2):
    query_embedding = model.encode([query])  # Convert the query into an embedding
    D, I = index.search(np.array(query_embedding), top_k)  # Search for top-k similar documents
    return [documents[i] for i in I[0]]  # Return the top-k retrieved documents

# Load the pre-trained GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to generate a response using GPT-2
def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = gpt2_model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=gpt2_tokenizer.eos_token_id,
        eos_token_id=gpt2_tokenizer.eos_token_id,
    )
    full_output = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated answer
    if "Answer:" in full_output:
        answer = full_output.split("Answer:")[-1].strip().split("Question:")[0].strip()
    else:
        answer = full_output.strip()

    # Return both context and answer nicely
    return f"{answer}"



# Retrieval and Generation
# Function to create an AI agent that retrieves relevant documents and generates a response
def ai_agent(query):
    # Retrieve relevant documents using FAISS
    retrieved_docs = retrieve_documents(query)
    context_str = ' '.join(retrieved_docs)  # Combine retrieved documents into a context string

    # Limit the context to only the first 200 words (adjust as necessary)
    context_str = ' '.join(context_str.split()[:200])  # Limit context to the first 200 words

    # Generate a response using GPT-2 based on the query and retrieved documents
    response = generate_response(query, context_str)
    return response

# Streamlit UI
import streamlit as st

st.title("🧠 AI Chatbot for Document-Based Responses")
st.write("Ask me anything related to AI, Machine Learning, or related topics!")

# Input for user query
user_input = st.text_input("You:", "")

if user_input:
    with st.spinner("Thinking..."):
        response = ai_agent(user_input)
        st.text_area("Bot:", value=response, height=200)
