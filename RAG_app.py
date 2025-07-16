####################################################################
#                         import
####################################################################

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path

# --- LangChain providers ---------------------------------------------------
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.retrievers.document_compressors import CohereRerank

# LangChain core
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers import ContextualCompressionRetriever

# Streamlit
import streamlit as st

####################################################################
#              Config & constants
####################################################################
list_LLM_providers = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
    ":hugging_face: **HuggingFace**",
]

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)

# --- ensure directories exist ---------------------------------------------
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#            Streamlit page
####################################################################
st.set_page_config(page_title="Chat With Your Data")
st.title("🤖 RAG chatbot")

# Initialise session-state keys
for _k in [
    "openai_api_key",
    "google_api_key",
    "cohere_api_key",
    "hf_api_key",
    "error_message",
]:
    st.session_state.setdefault(_k, "")

####################################################################
#            Helper – Expander with model params
####################################################################
def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="API key",
    list_models=None,
):
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            text_input_API_key, type="password", placeholder="Insert your API key"
        )
        st.session_state.google_api_key = st.session_state.hf_api_key = ""
    elif LLM_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key, type="password", placeholder="Insert your API key"
        )
        st.session_state.openai_api_key = st.session_state.hf_api_key = ""
    else:  # HuggingFace
        st.session_state.hf_api_key = st.text_input(
            text_input_API_key, type="password", placeholder="Insert your API key"
        )
        st.session_state.openai_api_key = st.session_state.google_api_key = ""

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {LLM_provider} model", list_models or []
        )
        st.session_state.temperature = st.slider(
            "temperature", 0.0, 1.0, 0.5, 0.1, help="Higher = more creative"
        )
        st.session_state.top_p = st.slider(
            "top_p", 0.0, 1.0, 0.95, 0.05, help="Nucleus sampling parameter"
        )

####################################################################
#            Sidebar & document chooser
####################################################################
def sidebar_and_documentChooser():
    with st.sidebar:
        st.caption(
            "🚀 A retrieval-augmented chatbot powered by LangChain, Cohere, OpenAI, Gemini & 🤗"
        )

        llm_chooser = st.radio(
            "Select provider",
            list_LLM_providers,
            captions=[
                "[OpenAI pricing](https://openai.com/pricing)",
                "Gemini rate limit ~60 RPM",
                "Free tier via HF Inference API",
            ],
        )

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                "OpenAI",
                "OpenAI API Key",
                ["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
            )
        elif llm_chooser == list_LLM_providers[1]:
            expander_model_parameters(
                "Google",
                "Google API Key",
                [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                ],
            )
        else:
            expander_model_parameters(
                "HuggingFace",
                "HuggingFace API Key",
                ["mistralai/Mistral-7B-Instruct-v0.2"],
            )

        st.session_state.assistant_language = st.selectbox(
            "Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        retrievers = (
            list_retriever_types[:-1]
            if st.session_state.selected_model == "gpt-3.5-turbo"
            else list_retriever_types
        )
        st.session_state.retriever_type = st.selectbox("Retriever type", retrievers)

        if st.session_state.retriever_type == "Cohere reranker":
            st.session_state.cohere_api_key = st.text_input(
                "Cohere API Key", type="password", placeholder="Insert Cohere key"
            )

    # ---- Tabs ----------------------------------------------------------------
    tab_new, tab_open = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )
    with tab_new:
        st.session_state.uploaded_file_list = st.file_uploader(
            "Select documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv"],
        )
        st.session_state.vector_store_name = st.text_input(
            "Vectorstore name (folder will be created under data/vector_stores/)"
        )
        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        if st.session_state.error_message:
            st.warning(st.session_state.error_message)

        # =========== OPEN VECTORSTORE TAB ===========
    with tab_open:
        store_root = LOCAL_VECTOR_STORE_DIR
        all_stores = sorted([d.name for d in store_root.iterdir() if d.is_dir()])

        if not all_stores:
            st.info("No vector‑stores found. Create one in the first tab.")
        else:
            selected_vs = st.selectbox("Select a saved Vectorstore", all_stores)

            if st.button("Load selected Vectorstore") and selected_vs:
                # check API keys
                missing = []
                if not any([st.session_state.openai_api_key,
                            st.session_state.google_api_key,
                            st.session_state.hf_api_key]):
                    missing.append(f"{st.session_state.LLM_provider} API key")
                if (st.session_state.retriever_type == "Cohere reranker"
                        and not st.session_state.cohere_api_key):
                    missing.append("Cohere API key")
                if missing:
                    st.warning("Please insert " + ", and ".join(missing) + ".")
                    st.stop()

                selected_path = store_root / selected_vs
                with st.spinner("Loading vector‑store…"):
                    try:
                        embeddings = select_embeddings_model()
                        st.session_state.vector_store = Chroma(
                            embedding_function=embeddings,
                            persist_directory=selected_path.as_posix(),
                        )
                        st.session_state.retriever = create_retriever(
                            st.session_state.vector_store,
                            embeddings,
                            st.session_state.retriever_type,
                        )
                        (st.session_state.chain,
                         st.session_state.memory) = create_ConversationalRetrievalChain(
                            st.session_state.retriever,
                            language=st.session_state.assistant_language,
                        )
                        clear_chat_history()
                        st.success(f"Loaded **{selected_vs}** successfully.")
                    except Exception as e:
                        st.error(e)

            with st.spinner("Loading vectorstore…"):
                try:
                    embeddings = select_embeddings_model()
                    st.session_state.vector_store = Chroma(
                        embedding_function=embeddings,
                        persist_directory=selected_path,
                    )
                    st.session_state.retriever = create_retriever(
                        st.session_state.vector_store,
                        embeddings,
                        st.session_state.retriever_type,
                    )
                    (
                        st.session_state.chain,
                        st.session_state.memory,
                    ) = create_ConversationalRetrievalChain(
                        st.session_state.retriever,
                        language=st.session_state.assistant_language,
                    )
                    clear_chat_history()
                    st.info("Vectorstore loaded successfully.")
                except Exception as e:
                    st.error(e)

####################################################################
#        Load & split docs
####################################################################
def delte_temp_files():
    for f in glob.glob(TMP_DIR.as_posix() + "/*"):
        try:
            os.remove(f)
        except:
            pass


def langchain_document_loader():
    docs = []
    loaders = [
        ("**/*.txt", TextLoader),
        ("**/*.pdf", PyPDFLoader),
        ("**/*.csv", CSVLoader),
        ("**/*.docx", Docx2txtLoader),
    ]
    for glob_pat, cls in loaders:
        docs.extend(
            DirectoryLoader(
                TMP_DIR.as_posix(),
                glob=glob_pat,
                loader_cls=cls,
                show_progress=True,
                loader_kwargs={"encoding": "utf8"} if cls is CSVLoader else None,
            ).load()
        )
    return docs


def split_documents_to_chunks(documents):
    return RecursiveCharacterTextSplitter(
        chunk_size=1600, chunk_overlap=200
    ).split_documents(documents)

####################################################################
#        Embeddings & Retriever helpers
####################################################################
def select_embeddings_model():
    if st.session_state.LLM_provider == "OpenAI":
        return OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    if st.session_state.LLM_provider == "Google":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.session_state.google_api_key,
        )
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=st.session_state.hf_api_key, model_name="thenlper/gte-large"
    )


def Vectorstore_backed_retriever(vectorstore, k=16):
    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )


def create_compression_retriever(embeddings, base_retriever, k=16):
    pipeline = DocumentCompressorPipeline(
        transformers=[
            CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator=". "),
            EmbeddingsRedundantFilter(embeddings=embeddings),      # ← fixed
            EmbeddingsFilter(embeddings=embeddings, k=k),          # ← fixed
            LongContextReorder(),
        ]
    )
    return ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=base_retriever
    )


def CohereRerank_retriever(
    base_retriever, cohere_api_key, model_id="rerank-multilingual-v3.0", top_n=10
):
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model=model_id, top_n=top_n
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )


def create_retriever(
    vector_store, embeddings, retriever_type="Contextual compression"
):
    base = Vectorstore_backed_retriever(vector_store)
    if retriever_type == "Vectorstore backed retriever":
        return base
    if retriever_type == "Contextual compression":
        return create_compression_retriever(embeddings, base)
    # Cohere reranker
    model_id = "rerank-multilingual-v3.0"
    if (
        st.session_state.get("cohere_api_key")
        and st.session_state.get("cohere_api_key").strip()
    ):
        return CohereRerank_retriever(
            base, st.session_state.cohere_api_key, model_id=model_id
        )
    st.warning("Cohere key missing – falling back to contextual compression.")
    return create_compression_retriever(embeddings, base)

####################################################################
#        Build the full RAG chain
####################################################################
def answer_template(language="english"):
    return f"""Answer the question at the end, using only the following context (delimited by <context></context>).
<context>
{{chat_history}}

{{context}}
</context>

Question: {{question}}
Language: {language}.
"""


def create_memory(model_name):
    if model_name == "gpt-3.5-turbo":
        return ConversationSummaryBufferMemory(
            max_token_limit=1024,
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key,
                temperature=0.1,
            ),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    return ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question",
    )


def create_chat_llm():
    # map deprecated gemini-pro → gemini-2.5-pro
    if st.session_state.selected_model in {"gemini-pro", "models/gemini-pro"}:
        st.session_state.selected_model = "gemini-2.5-pro"

    provider = st.session_state.LLM_provider
    mdl = st.session_state.selected_model
    t, top_p = st.session_state.temperature, st.session_state.top_p

    if provider == "OpenAI":
        return ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=mdl,
            temperature=t,
            model_kwargs={"top_p": top_p},
        )
    if provider == "Google":
        return ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=mdl,
            temperature=t,
            top_p=top_p,
            convert_system_message_to_human=True,
        )
    return HuggingFaceHub(
        repo_id=mdl,
        huggingfacehub_api_token=st.session_state.hf_api_key,
        model_kwargs={
            "temperature": t,
            "top_p": top_p,
            "do_sample": True,
            "max_new_tokens": 1024,
        },
    )


def create_ConversationalRetrievalChain(retriever, language="english"):
    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the conversation and a follow-up question, rephrase the follow-up to a standalone question in its original language.

Chat History:
{chat_history}

Follow-up: {question}

Standalone question:""",
    )
    answer_prompt = ChatPromptTemplate.from_template(
        answer_template(language)
    )

    memory = create_memory(st.session_state.selected_model)
    llm_standalone = create_chat_llm()
    llm_answer = create_chat_llm()

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=llm_standalone,
        llm=llm_answer,
        memory=memory,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return chain, memory

####################################################################
#        Vectorstore creation logic (button handler)
####################################################################
def chain_RAG_blocks():
    errs = []
    if not any(
        [
            st.session_state.openai_api_key,
            st.session_state.google_api_key,
            st.session_state.hf_api_key,
        ]
    ):
        errs.append(f"insert your {st.session_state.LLM_provider} API key")
    if (
        st.session_state.retriever_type == "Cohere reranker"
        and not st.session_state.cohere_api_key
    ):
        errs.append("insert your Cohere API key")
    if not st.session_state.uploaded_file_list:
        errs.append("select documents to upload")
    if not st.session_state.vector_store_name.strip():
        errs.append("provide a Vectorstore name")

    if errs:
        st.session_state.error_message = "Please " + ", and ".join(errs) + "."
        return

    st.session_state.error_message = ""
    with st.spinner("Creating vectorstore…"):
        try:
            delte_temp_files()
            # save uploads
            for f in st.session_state.uploaded_file_list:
                with open(TMP_DIR / f.name, "wb") as tmp:
                    tmp.write(f.read())

            documents = langchain_document_loader()
            chunks = split_documents_to_chunks(documents)
            embeddings = select_embeddings_model()

            persist_dir = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            st.session_state.vector_store = Chroma.from_documents(
                chunks, embeddings, persist_directory=persist_dir.as_posix()
            )

            st.session_state.retriever = create_retriever(
                st.session_state.vector_store,
                embeddings,
                st.session_state.retriever_type,
            )
            (
                st.session_state.chain,
                st.session_state.memory,
            ) = create_ConversationalRetrievalChain(
                st.session_state.retriever,
                language=st.session_state.assistant_language,
            )
            clear_chat_history()
            st.success(
                f"Vectorstore **{st.session_state.vector_store_name}** created!"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

####################################################################
#        Chat helpers
####################################################################
def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[
                st.session_state.assistant_language
            ],
        }
    ]
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    try:
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]
        if st.session_state.LLM_provider == "HuggingFace":
            if "\nAnswer: " in answer:
                answer = answer.split("\nAnswer: ", 1)[1]

        st.session_state.messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        )
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("**Source documents**"):
                for doc in response["source_documents"]:
                    page = (
                        f" (Page {doc.metadata.get('page')})"
                        if "page" in doc.metadata
                        else ""
                    )
                    st.markdown(
                        f"**Source:** {doc.metadata.get('source')}{page}\n\n{doc.page_content}\n"
                    )

    except Exception as e:
        st.warning(e)

####################################################################
#                         Main Chatbot
####################################################################
def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    col1.subheader("Chat with your data")
    col2.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        clear_chat_history()

    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input():
        if not any(
            [
                st.session_state.openai_api_key,
                st.session_state.google_api_key,
                st.session_state.hf_api_key,
            ]
        ):
            st.info(
                f"Please insert your {st.session_state.LLM_provider} API key."
            )
            return
        with st.spinner("Running…"):
            get_response_from_LLM(prompt)

if __name__ == "__main__":
    chatbot()
