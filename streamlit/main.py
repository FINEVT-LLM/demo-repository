import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS, Milvus, Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# API KEY 정보 불러오기
dotenv_path = "/home/seongeonkim/Seongeon/config/.env"
load_dotenv(dotenv_path=dotenv_path)

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.title("FIneVT RAG chatbot v0.0")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4o-turbo", "gpt-4o-mini"], index=0
    )
    selected_prompt = st.selectbox("프롬프트 선택", ["FineVT Advanced Prompt"])
    selected_vectordb = st.selectbox(
        "Vector DB 선택", ["FAISS", "Milvus(현재는 불가능)", "Chroma"]
    )
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])


# 이전 대화 출력
def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 입력
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 업로드한 파일 캐시 저장
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file, vectordb=selected_vectordb):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    class EmbeddingClient:
        def __init__(self, api_url: str):
            self.api_url = api_url

        def __call__(self, text):
            """
            객체 호출 시 text embedding
            """
            return self.embed_query(text)

        def embed_documents(self, texts, mode: str = "batch"):
            """
            Documnet 구조의 text 임베딩 및 반환
            """
            response = requests.post(
                f"{self.api_url}/embed",
                json={"texts": texts, "mode": mode},
            )
            if response.status_code == 200:
                return response.json()["embeddings"]
            else:
                raise RuntimeError(
                    f"API Error: {response.status_code}, {response.text}"
                )

        def embed_query(self, text):
            """
            단일 text 임베딩
            """
            return self.embed_documents([text])[0]

    client = EmbeddingClient("http://127.0.0.1:8111")

    # 벡터 데이터베이스 생성
    if vectordb == "FAISS":
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=client)
    elif vectordb == "Milvus(현재는 불가능)":
        vectorstore = Milvus.from_documents(documents=split_documents, embedding=client)
    elif vectordb == "Chroma":
        vectorstore = Chroma.from_documents(documents=split_documents, embedding=client)

    retriever = vectorstore.as_retriever()

    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o", prompt_name="FineVT Basic Prompt"):
    if prompt_name == "FineVT Advanced Prompt":
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    """You are an assistant for question-answering tasks. 
                            Use the following pieces of retrieved context to answer the question. 
                            If you don't know the answer, just say that you don't know. 
                            Please write your answer in a markdown table format with the main points.
                            Be sure to include your source and page numbers in your answer.
                            Answer in Korean.""",
                ),
                (
                    "user",
                    """#Example Format:
                        (brief summary of the answer)
                        (table)
                        (detailed answer to the question)
                        
                        **출처**
                        - (page source and page number)

                        #Question: 
                        {question}
                            
                        #Context: 
                        {context} 

                        #Answer:""",
                ),
            ]
        )

    llm = ChatOpenAI(model_name=model_name, temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성
    retriever = embed_file(uploaded_file, selected_vectordb)
    chain = create_chain(
        retriever, model_name=selected_model, prompt_name=selected_prompt
    )
    st.session_state["chain"] = chain

if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_message()

user_input = st.chat_input("저는 FineVT 챗봇입니다. 무엇을 도와드릴까요?")

# 경고 메시지를 띄우기 위한 빈 영역
warning_message = st.empty()

if user_input:
    # chain 생성
    chain = st.session_state["chain"]

    if chain is not None:
        st.chat_message("user").write(user_input)
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일 업로드 경고 메시지
        warning_message.error("파일을 업로드해주세요.")
