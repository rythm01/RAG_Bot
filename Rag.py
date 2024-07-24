from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import re,json
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Use HuggingFaceInstructEmbeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs = {'device':'cpu'},
    encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for message in st.session_state.chat_history:
        if message.type == 'human':
            st.chat_message("human").write(message.content)
        elif message.type == 'ai':
            # Process the response content and extract the helpful answer
            json_response = convert_to_json(message.content)
            print(json_response)
            helpful_response = extract_helpful_response(json_response)
            st.chat_message("ai").write(helpful_response)

def convert_to_json(content):
    try:
        # Attempt to parse as JSON if possible
        return json.loads(content)
    except json.JSONDecodeError:
        # If not JSON, return a dict with the content as a string
        return {"response": content}

def extract_helpful_response(json_response):
    response_text = json_response.get("response", "")
    # Use regex to find the last helpful answer within the response text
    matches = re.findall(r"Helpful Answer:\s*(.*?)\s*(?:\n|$)", response_text, re.DOTALL)
    if matches:
        last_answer = matches[-1].strip()
        # Check if the last answer is empty
        if last_answer:
            return last_answer
    # Fallback if the pattern is not found or the last answer is empty
    return "No idea"
    
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":crystal_ball:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Custom Data Of PDFs :crystal_ball:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()
