import streamlit as st

from src.llm import generate_no_rag_response, generate_rag_response
from vector_store import prepare_document_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def setup_page():
    st.set_page_config(layout="wide")
    st.title("Question Answering System")


# Initialize or load existing retriever
def initialize_retriever():
    if 'retriever' not in st.session_state:
        st.session_state.retriever = prepare_document_retriever()


def get_user_question():
    return st.text_input("Ask your question here:")


def handle_question(question):
    if question:
        no_rag_response = generate_no_rag_response(question)
        rag_response, retrieved_chunks = generate_rag_response(question, st.session_state.retriever)
        display_responses(rag_response, retrieved_chunks, no_rag_response)
    else:
        st.warning("Please enter a question.")


def display_responses(rag_response, retrieved_chunks, no_rag_response):
    col1, col2 = st.columns(2)

    with col1:
        st.header("RAG")
        st.subheader("Response")
        st.write(rag_response)
        display_retrieved_chunks(retrieved_chunks)

    with col2:
        st.header("No RAG")
        st.subheader("Response")
        st.write(no_rag_response)


def display_retrieved_chunks(retrieved_chunks):
    st.subheader("Retrieved Chunks")
    for index, chunk in enumerate(retrieved_chunks):
        with st.expander(f"Chunk {index + 1}"):
            st.write(chunk.page_content)


def main():
    setup_page()
    initialize_retriever()
    question = get_user_question()
    if st.button("Get Answer"):
        handle_question(question)


if __name__ == "__main__":
    main()
