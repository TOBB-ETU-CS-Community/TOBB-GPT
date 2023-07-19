import json
import os
import time

import openai
import requests
import speech_recognition as s_r
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from modules.utils import add_bg_from_local, set_page_config

# os.environ["AZURE_S2T_KEY"] = st.secrets["AZURE_S2T_KEY"]
os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


# Storing the chat
if "user" not in st.session_state:
    st.session_state.user = []
if "bot" not in st.session_state:
    st.session_state.bot = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None


def get_speech() -> bool:
    """takes user voice input

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if user voice is taken successfully else False
    """
    if audio_bytes := audio_recorder(
        text="Lütfen sesli bir soru sormak için sağdaki ikona tıklayın ve konuşmaya başlayın",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="user",
        icon_size="4x",
    ):
        # st.audio(audio_bytes, format="audio/wav")
        # write('output.wav', 44100, audio_bytes)
        temp_file = os.getcwd()
        temp_dir = temp_file.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, "output.wav")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_bytes)
            return True
    return False


def gets():
    r = s_r.Recognizer()
    my_mic = s_r.Microphone(
        device_index=1
    )  # my device index is 1, you have to put your device index
    if st.button("Konuşarak sorun 🎙️"):
        with my_mic as source:
            audio = r.listen(source)
            return r.recognize_google(audio)
    return None


def speech2text(subscription_key, region) -> str:
    """convert speech to text

    Parameters
    ----------
    subscription_key : str
        Openai api key

    Returns
    -------
    str
        Text generated from speech
    """
    url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=tr-TR"
    headers = {
        "Content-type": 'audio/wav;codec="audio/pcm";',
        # 'Ocp-Apim-Subscription-Key': subscription_key,
        "Authorization": get_token(subscription_key, region),
    }
    with open("output.wav", "rb") as payload:
        response = requests.request("POST", url, headers=headers, data=payload)
        # st.write(response)
        text = json.loads(response.text)
        if "DisplayText" in text.keys():
            return text["DisplayText"]


def get_token(subscription_key, region) -> str:
    """get access token for the given subscription key

    Parameters
    ----------
    subscription_key : str
        Openai api key

    Returns
    -------
    str
        access token
    """
    fetch_token_url = (
        f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    )
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    response = requests.post(fetch_token_url, headers=headers)
    return str(response.text)


def create_vector_store_retriever(query):
    search = GoogleSearchAPIWrapper()

    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=lambda query: search.results(query, 5),
    )
    result = tool.run(query)
    # st.write(result)
    urls = [val["link"] for val in result]
    # st.write(urls)
    loader = WebBaseLoader(urls)
    documents = loader.load()
    for doc in documents:
        doc.page_content = doc.page_content
        doc.metadata = {"url": doc.metadata["source"]}
    # st.write(documents)
    if st.session_state.model == "openai":
        char_text_splitter = MarkdownTextSplitter(
            chunk_size=2048,
            chunk_overlap=256,
        )
    else:
        char_text_splitter = MarkdownTextSplitter(
            chunk_size=256,
            chunk_overlap=32,
        )
    texts = char_text_splitter.split_documents(documents)

    embeddings = (
        OpenAIEmbeddings()
        if st.session_state.model == "openai"
        else HuggingFaceEmbeddings()
    )
    vector_store = Chroma.from_documents(texts, embeddings)
    return vector_store.as_retriever(), urls


def transform_question(question):
    user_message = f"""Dönüştürmen gereken soru, tek tırnak işaretleri arasındadır:
     '{question}'
     Verdiğin cevap da yalnızca arama sorgusu yer almalı, başka herhangi bir şey yazmamalı ve tırnak işareti gibi
     bir noktalama işareti de eklememelisin. Sonucu json formatında dönmelisin."""

    user_message += """Json formatı şöyle olmalı:
     {"query": output}
     """
    if st.session_state.model != "openai":
        return question
    system_message = """Bu görevde yapman gereken bu şey, kullanıcı sorularını arama sorgularına dönüştürmektir. Bir kullanıcı
     soru sorduğunda, soruyu, kullanıcının bilmek istediği bilgileri getiren bir Google arama sorgusuna dönüştürürsün. Eğer soru türkçe
     ise türkçe, ingilizce ise ingilizce bir cevap üret ve cevabı json formatında döndür. Json formatı şöyle olmalı:
     {"query": output}
     """
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


def create_retrieval_qa(prompt_template, llm, retriever):
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


def is_api_key_valid(model: str, api_key: str):
    if api_key is None:
        st.sidebar.warning("Lütfen geçerli bir API keyi girin!", icon="⚠")
        return False
    elif model == "openai" and not api_key.startswith("sk-"):
        st.sidebar.warning("Lütfen geçerli bir API keyi girin!", icon="⚠")
        return False
    elif model == "huggingface" and not api_key.startswith("hf_"):
        st.sidebar.warning("Lütfen geçerli bir API keyi girin!", icon="⚠")
        return False
    else:
        key = (
            "OPENAI_API_KEY"
            if model == "openai"
            else "HUGGINGFACEHUB_API_TOKEN"
        )
        os.environ[key] = api_key
        return True


def show_sidebar():
    st.sidebar.markdown(
        "<center><h1>Sohbet Botu Ayarları</h1></center> <br>",
        unsafe_allow_html=True,
    )

    llm = st.sidebar.selectbox(
        "Lütfen bir LLM seçin:",
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
    st.session_state.model = llm
    if llm != "<Seçiniz>":
        st.sidebar.text_input(f"Lütfen {llm} API keyini girin:", key="api_key")
        model = "openai" if llm.startswith("openai") else "huggingface"
        if is_api_key_valid(model, st.session_state.api_key):
            st.sidebar.success("API keyi başarıyla alındı.")
            return True
    return False


def start_chat():
    if st.session_state.model.startswith("openai"):
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
        )
    else:
        llm = HuggingFaceHub(
            repo_id=st.session_state.model,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 4096,
            },
        )

    prompt_template = """
    <|SYSTEM|>#
    - Eğer sorulan soru doğrudan üniversiteleri liseler, lise eğitimi ve üniversite eğitimi ile ilgili değilse
     "Üzgünüm, bu soru liseler ya da üniversiteler ile ilgili olmadığından cevaplayamıyorum. Lütfen başka bir soru sormayı
      deneyin." diye yanıt vermelisin ve başka
      herhangi bir şey söylememelisin.
    - Sen Türkçe konuşan bir botsun. Soru Türkçe ise her zaman Türkçe cevap vermelisin.
    - If the question is in English, then answer in English. If the question is Turkish, then answer in Turkish.
    - Sen yardımsever, nazik, gerçek dünyaya ait bilgilere dayalı olarak soru cevaplayan bir sohbet botusun.
    Yalnızca üniversiteler ile ilgili sorulara cevap verebilirsin, asla başka bir soruya cevap vermemelisin.
    <|USER|>
    Şimdi kullanıcı sana bir soru soruyor. Bu soruyu sana verilen bağlam ve sohbet geçmişindeki bilgilerinden faydalanarak yanıtla.

    SORU: {question}
    BAĞLAM:
    {context}

    CEVAP: <|ASSISTANT|>
    """
    # Transform output to json
    user_input = ""
    # region = "switzerlandwest"  # huseyin
    # region = "eastus"  # ata
    gets()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # config = sdk.SpeechConfig(subscription=os.environ["AZURE_S2T_KEY"], region=region)
    # config.speech_synthesis_language = "tr-TR"
    # config.speech_synthesis_voice_name = "en-US-JennyNeural"
    # speech_synthesizer = sdk.SpeechSynthesizer(speech_config=config, audio_config=None)
    try:
        if user_input := st.chat_input(
            # label="🎙️ ya da ✍️",
            # value="" if speech is None else speech,
            placeholder="Yazarak sorun ✍️",
            key="text_box",
            max_chars=100,
        ):
            # user_input = st.session_state.text_box
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            with st.spinner("Soru internet üzerinde aranıyor"):
                query = transform_question(st.session_state.text_box)
                query = query.replace('"', "").replace("'", "")

            with st.spinner("Toplanan bilgiler derleniyor"):
                retriever, urls = create_vector_store_retriever(query)
                qa = create_retrieval_qa(prompt_template, llm, retriever)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Soru cevaplanıyor"):
                    response = qa.run(user_input)

                source_output = " \n \n Soru, şu kaynaklardan yararlanarak cevaplandı: \n \n"
                for url in urls:
                    source_output += url + " \n \n "
                response += source_output
                llm_output = ""
                for i in range(len(response)):
                    llm_output += response[i]
                    message_placeholder.write(f"{llm_output}▌")
                    time.sleep(0.01)
                message_placeholder.write(llm_output)

            st.session_state.messages.append(
                {"role": "assistant", "content": llm_output}
            )

    except Exception as e:
        _, center_err_col, _ = st.columns([1, 8, 1])
        center_err_col.markdown("<br>", unsafe_allow_html=True)
        # center_err_col.error(f"An error occurred: {type(e).__name__}")
        print(e)
        center_err_col.error(
            "\nLütfen biz hatayı çözerken bekleyin. Teşekkürler ;]"
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

    # css_file = os.path.join("style", "style.css")
    # local_css(css_file)

    st.markdown(
        """<h1 style='text-align: center; color: black; font-size: 60px;'> 🤖 Üniversite Sohbet Botu </h1>
        <br>""",
        unsafe_allow_html=True,
    )

    if show_sidebar():
        start_chat()


if __name__ == "__main__":
    main()
