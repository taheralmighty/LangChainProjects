# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# import tempfile

# # Title of the app
# st.title("Q&A Chatbot with PDF and FAISS")

# # Upload PDF file
# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(
#             uploaded_file.read()
#         )  # Write the uploaded file content to the temp file
#         temp_file_path = temp_file.name
#     # Step 1: Load and preprocess the PDF
#     st.info("Processing PDF...")
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = text_splitter.split_documents(documents)

#     # Step 2: Create embeddings and vector store
#     st.info("Creating embeddings...")
#     embedding_model = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(texts, embedding_model)

#     # Step 3: Define LLM and prompt
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
#     prompt_template = """
#     Answer the user's question using the provided context.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     prompt = ChatPromptTemplate.from_template(prompt_template)

#     # Step 4: Accept user queries
#     st.info("Ask a question about the PDF!")
#     user_query = st.text_input("Enter your question")

#     if user_query:
#         # Retrieve relevant documents
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#         relevant_docs = retriever.get_relevant_documents(user_query)

#         # Combine the context
#         context = "\n".join([doc.page_content for doc in relevant_docs])

#         # Generate response
#         with st.spinner("Generating answer..."):
#             response = llm.invoke({"context": context, "question": user_query})

#         # Display the answer
#         st.success("Answer:")
#         st.write(response)

#         # Optional: Display retrieved context
#         with st.expander("Retrieved Context"):
#             st.write(context)

import streamlit as st

st.title("Hello Custom CSS Chatbot 🤖")

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

load_css()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! How can I help you today? 😊"}
    ]

# Function to render a single chat message
def render_chat_message(role, content):
    div = f"""
    <div class="chat-row 
    {'row-reverse' if role == 'user' else ''}">
    <img class="chat-icon" src="app/static/{'bot.png' if role == 'assistant' else 'user.png'}"
    width=32 height=32>
    <div class="chat-bubble {'ai-bubble' if role == 'assistant' else 'human-bubble'}">
    &#8203;{content}
    </div>
    </div>
    """
    st.markdown(div, unsafe_allow_html=True)

# Initial rendering of chat history
for message in st.session_state.chat_history:
    render_chat_message(message["role"], message["content"])

# Input field for user messages
user_message = st.chat_input("Type your message here...")
if user_message:
    # Add user message to chat history
    user_entry = {"role": "user", "content": user_message}
    st.session_state.chat_history.append(user_entry)
    render_chat_message(user_entry["role"], user_entry["content"])

    # Simulate assistant response (replace this with actual chatbot logic)
    assistant_response = "Thanks for your question! Let me find the answer."
    assistant_entry = {"role": "assistant", "content": assistant_response}
    st.session_state.chat_history.append(assistant_entry)
    render_chat_message(assistant_entry["role"], assistant_entry["content"])