from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_endpoint="azure_endpoint",
    azure_deployment="gpt-35-turbo",
    temperature=0,
)


def format_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)


def generate_rag_response(user_question, retriever):
    chunks = retriever.invoke(user_question)
    context = format_chunks(chunks)

    prompt = f'''Use the following pieces of context to answer the question at the end. Answer in the same language 
    of the given question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

       {context}

       Question: {user_question}

       Answer:'''

    response = llm.invoke(prompt)
    return response.content, chunks


def generate_no_rag_response(user_question):
    response = llm.invoke(user_question)
    return response.content
