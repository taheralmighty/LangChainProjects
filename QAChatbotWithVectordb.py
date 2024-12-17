import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile
import time

import streamlit as st

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# configuring streamlit page settings
st.set_page_config(page_title=" ChatBot", page_icon="💬", layout="centered")

# streamlit page title
st.title("🤖 Q&A ChatBot")

greeting_msg = "Hello👋! I'm your AI assistant. Feel free to ask me anything related to the pdf you uploaded I'm here to help! 😊"

# initializing chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": greeting_msg})

# Initializing memory to store conversation history
if "store" not in st.session_state:
    st.session_state.store = {}

# Flag to track if the PDF has been uploaded and processed
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

def chatbot(input_user_message, vectorstore):
    # creating a prompt template
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based on the provided context. If the answer isn't in the text, respond with Not enough information to answer this."),
            # ("AI", "{context}"),
            MessagesPlaceholder(variable_name="history_messages"),
            ("human", "{input_user_message}"),
        ]
    )

    # initializing OpenAI Chat model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    trimmer = trim_messages(
        max_tokens=100,
        strategy="last",
        token_counter=llm,
        # Usually, we want to keep the SystemMessage
        include_system=True,
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
    )

    def get_session_history(session_id):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        else:
            st.session_state.store[session_id].messages = trimmer.invoke(
                st.session_state.store[session_id].messages
            )
        return st.session_state.store[session_id]

    # Initializing the output parser
    output_parser = StrOutputParser()

    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(input_user_message)

    # Combine the context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Creating a Chain with the prompt and memory
    QA_chain = chat_prompt | llm | output_parser

    model_with_memory = RunnableWithMessageHistory(
        QA_chain,
        get_session_history,
        input_messages_key="input_user_message",
        history_messages_key="history_messages",
    )
    session_id = "1234"

    response = model_with_memory.stream(
        {
            "input_user_message": f"Context:\n{context}\n\nQuestion: {input_user_message}"
        },
        {"configurable": {"session_id": session_id}},
    )

    return response

# If the PDF is not uploaded yet show the pdf upload UI
if not st.session_state.pdf_uploaded:
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(
                uploaded_file.read()
            )  # Write the uploaded file content to the temp file
            temp_file_path = temp_file.name

        # Loading and preprocessing the PDF
        st.info("Processing PDF...")
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Splitting documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Creating embeddings and vector store
        st.info("Creating embeddings...")
        embedding_model = OpenAIEmbeddings()
        st.session_state.vectorstore = FAISS.from_documents(texts, embedding_model)

        # Set the flag indicating the PDF has been uploaded
        st.session_state.pdf_uploaded = True
        st.info("PDF processed! You can now ask questions related to the uploaded PDF.")
        # Re-render the page to show the chatbot interface
        time.sleep(2)
        st.rerun()

else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # input field for user message
    user_prompt = st.chat_input("Ask me a question about the pdf I am here to help....")

    if user_prompt:
        # adding user message to chat and display it
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        vectorstore = st.session_state.vectorstore
        # sending user message to GPT and get displaying response
        with st.chat_message("assistant"):
            assistant_response = st.write_stream(chatbot(user_prompt, vectorstore))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_response}
        )
