import json
import os
import time
from collections import OrderedDict

import openai
import pandas as pd
import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from modules.utils import add_bg_from_local, local_css, set_page_config

# from gtts import gTTS
# import speech_recognition as s_r

# os.environ["AZURE_S2T_KEY"] = st.secrets["AZURE_S2T_KEY"]
os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
STREAMING_INTERVAL = 0.01


if "messages" not in st.session_state:
    st.session_state.messages = OrderedDict()


def is_api_key_valid(model_host: str, api_key: str):
    if api_key is None:
        st.sidebar.warning("Lütfen geçerli bir API keyi girin!", icon="⚠")
        return False
    elif model_host == "openai" and not api_key.startswith("sk-"):
        st.sidebar.warning(
            "Lütfen geçerli bir OpenAI API keyi girin!", icon="⚠"
        )
        return False
    elif model_host == "huggingface" and not api_key.startswith("hf_"):
        st.sidebar.warning(
            "Lütfen geçerli bir HuggingFace API keyi girin!", icon="⚠"
        )
        return False
    else:
        if model_host == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
        else:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        return True


# def speech2text():
# r = s_r.Recognizer()
# my_mic = s_r.Microphone(
#    device_index=1
# )  # my device index is 1, you have to put your device index
# if st.button("Konuşarak sorun 🎙️"):
#    with my_mic as source:
#        audio = r.listen(source)
#        return r.recognize_google(audio)
# return None


def transform_question(model_host, question):
    if model_host != "openai":
        return question

    system_message = """Bu görevde yapman gereken bu şey, kullanıcı sorularını arama sorgularına dönüştürmektir. Bir kullanıcı
     soru sorduğunda, soruyu, kullanıcının bilmek istediği bilgileri getirecek bir Google arama sorgusuna dönüştürmelisin. Eğer soru türkçe
     ise türkçe, ingilizce ise ingilizce bir cevap üret ve cevabı json formatında döndür. Json formatı şöyle olmalı:
     {"query": output}
     """
    user_message = f"""Dönüştürmen gereken soru, tek tırnak işaretleri arasındadır:
     '{question}'
     Verdiğin cevap da yalnızca arama sorgusu yer almalı, başka herhangi bir şey yazmamalı ve tırnak işareti gibi
     bir noktalama işareti de eklememelisin. Sonucu json formatında dönmelisin.
     Json formatı şöyle olmalı:
     {{"query": output}}"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    response = completion.choices[0].message.content
    json_object = json.loads(response)
    return json_object["query"]


def search_web(query, link_count: int = 3):
    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=lambda query: search.results(query, link_count),
    )
    return tool.run(query)


def create_query_vector_store(model_host, results):
    urls = [val["link"] for val in results]
    loader = WebBaseLoader(urls)
    documents = loader.load()
    for doc in documents:
        doc.metadata = {"url": doc.metadata["source"]}

    if model_host == "openai":
        char_text_splitter = MarkdownTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
        )
    else:
        char_text_splitter = MarkdownTextSplitter(
            chunk_size=256,
            chunk_overlap=32,
        )
    texts = char_text_splitter.split_documents(documents)

    embeddings = (
        OpenAIEmbeddings()
        if model_host == "openai"
        else HuggingFaceHubEmbeddings()
    )
    vector_store = Chroma.from_documents(texts, embeddings)
    return [vector_store.as_retriever(), urls]


def create_document_vector_store(model_host):
    excel = pd.read_excel(
        io="./static/Links.xlsx",
        sheet_name="Sheet1",
    )
    links = excel.Links.values.tolist()
    documents = []
    for i, link in enumerate(links):
        print(i, link)
        try:
            loader = WebBaseLoader(link)
            documents.extend(loader.load())
        except Exception:
            continue

    for doc in documents:
        doc.metadata = {"url": doc.metadata["source"]}

    if model_host == "openai":
        char_text_splitter = MarkdownTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
        )
    else:
        char_text_splitter = MarkdownTextSplitter(
            chunk_size=256,
            chunk_overlap=32,
        )
    texts = char_text_splitter.split_documents(documents)

    embeddings = (
        OpenAIEmbeddings()
        if model_host == "openai"
        else HuggingFaceHubEmbeddings()
    )
    persist_directory = f"./sigma_chatbot/chroma_db_{model_host}"
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.persist()
    return vector_store.as_retriever()


def load_document_vector_store(model_host):
    embeddings = (
        OpenAIEmbeddings()
        if model_host == "openai"
        else HuggingFaceHubEmbeddings()
    )
    persist_directory = f"./sigma_chatbot/chroma_db_{model_host}"
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store.as_retriever()


def create_llm(model):
    return (
        ChatOpenAI(
            model_name=model.split("/")[1],
            temperature=0,
        )
        if model.startswith("openai")
        else HuggingFaceHub(
            repo_id=model,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 4096,
            },
        )
    )


def create_main_prompt():
    return """
    <|SYSTEM|>#
    - Eğer sorulan soru doğrudan TOBB ETÜ (TOBB Ekonomi ve Teknoloji Üniversitesi) ile ilgili değilse
     "Üzgünüm, bu soru TOBB ETÜ ile ilgili olmadığından cevaplayamıyorum. Lütfen başka bir soru sormayı
      deneyin." diye yanıt vermelisin ve başka herhangi bir şey söylememelisin.
    - Sen Türkçe konuşan bir botsun. Soru Türkçe ise her zaman Türkçe cevap vermelisin.
    - If the question is in English, then answer in English. If the question is Turkish, then answer in Turkish.
    - Sen çok yardımsever, nazik, gerçek dünyaya ait bilgilere dayalı olarak soru cevaplayan bir sohbet botusun.
    - Cevapların açıklayıcı olmalı. Soru soran kişiye istediği tüm bilgiyi net bir şekilde vermelisin. Gerekirse uzun bir mesaj yazmaktan
    da çekinme.
    Yalnızca TOBB ETÜ Üniversitesi ile ilgili sorulara cevap verebilirsin, asla başka bir soruya cevap vermemelisin.
    <|USER|>
    Şimdi kullanıcı sana bir soru soruyor. Bu soruyu sana verilen bağlam ve sohbet geçmişindeki bilgilerinden faydalanarak
    açık ve net bir biçimde yanıtla.

    SORU: {question}
    BAĞLAM:
    {context}

    CEVAP: <|ASSISTANT|>
    """


def create_retrieval_qa(llm, prompt_template, retriever):
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    combine_docs_chain_kwargs = {"prompt": PROMPT}
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        memory=memory,
    )


def main():
    set_page_config()

    background_img_path = os.path.join("static", "background", "Sky BG.png")
    sidebar_background_img_path = os.path.join(
        "static", "background", "Lila Gradient.png"
    )
    page_markdown = add_bg_from_local(
        background_img_path=background_img_path,
        sidebar_background_img_path=sidebar_background_img_path,
    )
    st.markdown(page_markdown, unsafe_allow_html=True)

    css_file = os.path.join("static", "style.css")
    local_css(css_file)

    st.markdown(
        """<h1 style='text-align: center; color: black; font-size: 60px;'> 🤖 TOBB ETÜ Sohbet Botu </h1>
        <br>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "<center><h1>Sohbet Botu Ayarları</h1></center> <br>",
        unsafe_allow_html=True,
    )

    model = st.sidebar.selectbox(
        "Lütfen bir LLM seçin",
        [
            "<Seçiniz>",
            "openai/gpt-3.5-turbo",
            "google/flan-t5-xxl",
            "databricks/dolly-v2-3b",
            "Writer/camel-5b-hf",
            "Salesforce/xgen-7b-8k-base",
            "tiiuae/falcon-40b",
            "bigscience/bloom",
        ],
    )
    if model == "<Seçiniz>":
        st.sidebar.warning("Lütfen bir model seçin.")
        _, center_war_col, _ = st.columns([2, 5, 1])
        center_war_col.warning(
            "Lütfen sol taraftaki panelden bot için gerekli ayarlamaları yapın."
        )
        return
    else:
        api_key = st.sidebar.text_input(
            f"Lütfen {model} API keyini girin",
        )
        model_host = "openai" if model.startswith("openai") else "huggingface"
        if is_api_key_valid(model_host, api_key):
            st.sidebar.success("API keyi başarıyla alındı.")
        else:
            _, center_war_col, _ = st.columns([2, 5, 1])
            center_war_col.warning(
                "Lütfen sol taraftaki panelden bot için gerekli ayarlamaları yapın."
            )
            return

    with st.sidebar:
        choice = st.radio(
            "Botun nasıl çalışacağını seçin",
            ["İnternetteki sayfalar ile", "Hazır dokümanlar ile"],
        )

    for user_message, assistant_message in st.session_state.messages.items():
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_message)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(assistant_message)
            # sound_file = BytesIO()
            # tts = gTTS(assistant_message, lang="tr")
            # tts.write_to_fp(sound_file)
            # st.audio(sound_file)

    text_input = st.chat_input(
        placeholder="Yazarak sorun ✍️",
        key="text_box",
        max_chars=100,
    )
    # voice_input = speech2text()
    # user_input = voice_input or text_input
    user_input = text_input

    try:
        if user_input:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)

            if choice == "İnternetteki sayfalar ile":
                with st.spinner("Soru internet üzerinde aranıyor"):
                    query = transform_question(model_host, user_input)
                    query = query.replace('"', "").replace("'", "")
                    results = search_web(query)

                with st.spinner("Toplanan bilgiler derleniyor"):
                    retriever, urls = create_query_vector_store(
                        model_host, results
                    )
            elif choice == "Hazır dokümanlar ile":
                if os.path.exists(f"./sigma_chatbot/chroma_db_{model_host}"):
                    with st.spinner(
                        "TOBB ETÜ'ye ait geçmiş tarihte taranmış sayfalar yükleniyor"
                    ):
                        retriever = load_document_vector_store(model_host)
                else:
                    with st.spinner(
                        "TOBB ETÜ'ye ait 100e yakın sayfa taranıyor ve işleniyor"
                    ):
                        retriever = create_document_vector_store(model_host)

            with st.spinner("Soru cevaplanıyor"):
                llm = create_llm(model)
                prompt_template = create_main_prompt()
                qa = create_retrieval_qa(llm, prompt_template, retriever)
                response = qa.run(user_input)

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()

                if (
                    not (
                        response.startswith("Üzgünüm")
                        or response.startswith("I'm sorry")
                    )
                    and choice == "İnternetteki sayfalar ile"
                ):
                    source_output = " \n \n Soru, şu kaynaklardan yararlanarak cevaplandı: \n \n"
                    for url in urls:
                        source_output += url + " \n \n "
                    response += source_output
                llm_output = ""
                for i in range(len(response)):
                    llm_output += response[i]
                    message_placeholder.write(f"{llm_output}▌")
                    time.sleep(STREAMING_INTERVAL)
                message_placeholder.write(llm_output)
                # sound_file = BytesIO()
                # tts = gTTS(llm_output, lang="tr")
                # tts.write_to_fp(sound_file)
                # st.audio(sound_file)

            if user_input not in st.session_state.messages:
                assistant_message = llm_output
                st.session_state.messages[user_input] = assistant_message

    except Exception as e:
        _, center_err_col, _ = st.columns([1, 8, 1])
        center_err_col.error(
            "\n Sorunuz cevaplanamadı. Lütfen başka bir soru sormayı deneyin. Teşekkürler ;]"
        )
        print(f"An error occurred: {type(e).__name__}")
        print(e)


if __name__ == "__main__":
    main()
