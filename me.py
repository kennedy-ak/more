import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

# Set API keys (use environment variables in production)
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medibot",
    embedding=embeddings,
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up the LLM model
llm = OpenAI(temperature=0.4, max_tokens=500)

# Define prompt
system_prompt = (
    "You are a medical assistant. Answer ONLY using retrieved context. "
    "If no relevant information is found, say 'I don’t know'. "
    "use 2 sentences each"
    "Provide a page number when applicable.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI
st.title("MediChat - AI Medical Assistant")
st.write("Ask me any medical question!")

# Initialize chat history if not already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your question...")
if user_input:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get AI response
    response = rag_chain.invoke({"input": user_input})
    ai_response = response["answer"]
    
    # Display AI response
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)