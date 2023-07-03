import json
import os

import azure.cognitiveservices.speech as sdk
import openai
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from modules.configurations import add_bg_from_local
from streamlit_chat import message

os.environ["AZURE_S2T_KEY"] = st.secrets["AZURE_S2T_KEY"]
os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


# Storing the chat
if "user" not in st.session_state:
    st.session_state.user = []
if "bot" not in st.session_state:
    st.session_state.bot = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None


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
        with open("output.wav", mode="bw") as f:
            f.write(audio_bytes)
            return True
    return False


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
        func=lambda query: search.results(query, 3),
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
    char_text_splitter = MarkdownTextSplitter(
        chunk_size=2048,
        chunk_overlap=128,
    )
    texts = char_text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    return vector_store.as_retriever(), urls


def transform_question(question):
    system_message = """Bu görevde yapman gereken bu şey, kullanıcı sorularını arama sorgularına dönüştürmektir. Bir kullanıcı
     soru sorduğunda, soruyu, kullanıcının bilmek istediği bilgileri getiren bir Google arama sorgusuna dönüştürürsün. Eğer soru türkçe
     ise türkçe, ingilizce ise ingilizce bir cevap üret."""
    user_message = """Dönüştürmen gereken soru, tek tırnak işaretleri arasındadır:
     '{question}'
     Verdiğin cevap da yalnızca arama sorgusu yer almalı, başka herhangi bir şey yazmamalı ve tırnak işareti gibi
     bir noktalama işareti de eklememelisin.
     """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    response = openai.Completion.create(
        engine="chatgpt-3.5-turbo", prompt=messages, max_tokens=100
    )
    return response.choices[0].text


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


def is_api_key_valid(openai_api_key: str):
    if openai_api_key is None or not openai_api_key.startswith("sk-"):
        st.warning("Lütfen geçerli bir OpenAI API Key'i girin!", icon="⚠")
        return False
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True


def show_chat_ui():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

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
    region = "switzerlandwest"  # huseyin
    # region = "eastus"  # ata
    speech = None
    if get_speech():
        speech = speech2text(os.environ["AZURE_S2T_KEY"], region)
    st.text_input(
        label="",
        value="" if speech is None else speech,
        placeholder="Sorunuzu buraya yazabilirsiniz:",
        key="text_box",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col5:
        answer = st.button("Cevapla")

    config = sdk.SpeechConfig(
        subscription=os.environ["AZURE_S2T_KEY"], region=region
    )
    config.speech_synthesis_language = "tr-TR"
    # config.speech_synthesis_voice_name = "en-US-JennyNeural"
    speech_synthesizer = sdk.SpeechSynthesizer(
        speech_config=config, audio_config=None
    )

    try:
        if answer:
            with st.spinner("Soru internet üzerinde aranıyor:"):
                query = transform_question(st.session_state.text_box)
                query = query.replace('"', "").replace("'", "")
                retriever, urls = create_vector_store_retriever(query)
                qa = create_retrieval_qa(prompt_template, llm, retriever)

            with st.spinner(
                "Soru internet üzerindeki kaynaklar ile cevaplanıyor:"
            ):
                user_input = st.session_state.text_box
                output = qa.run(user_input)
            output += (
                f"\n\nSoru, şu kaynaklardan yararlanarak cevaplandı: \n {urls}"
            )

            if user_input not in st.session_state.user:
                st.session_state.user.append(user_input)
            if output not in st.session_state.bot:
                st.session_state.bot.append(output)

        if st.session_state["bot"]:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for i in range(len(st.session_state["bot"])):
                message(
                    st.session_state["user"][i],
                    is_user=True,
                    key=f"{str(i)}_user",
                )
                message(st.session_state["bot"][i], key=str(i))
                result = speech_synthesizer.speak_text(
                    st.session_state["bot"][i]
                )
                st.audio(result.audio_data)
    except Exception as e:
        _, center_err_col, _ = st.columns([1, 8, 1])
        center_err_col.markdown("<br>", unsafe_allow_html=True)
        center_err_col.error(f"An error occurred: {type(e).__name__}")
        print(e)
        center_err_col.error(
            "\nLütfen biz hatayı çözerken bekleyin. Teşekkürler ;]"
        )


def main():
    st.set_page_config(
        page_title="🤖 ChatBot",
        page_icon="🤖",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/olympian-21",
            "Report a bug": None,
            "About": "This is a chat bot for university students",
        },
    )

    st.markdown(
        "<center><h1>Üniversite Sohbet Botu</h1></center> <br> <br>",
        unsafe_allow_html=True,
    )

    add_bg_from_local("input/main.png", "input/sidebar.png")

    st.sidebar.markdown(
        "<center><h3>Sohbet Botu Ayarları</h3></center> <br> <br>",
        unsafe_allow_html=True,
    )

    st.sidebar.text_input("Lütfen OpenAI API Key'ini girin:", key="openai_api")
    # if st.sidebar.button("Bu OpenAI API Key'ini kullan"):
    # openai_api_key = st.session_state.openai_api

    if is_api_key_valid(st.session_state.openai_api):
        st.sidebar.success("Bu OpenAI API Key'i başarıyla alındı.")
        show_chat_ui()

    st.sidebar.markdown("<br> " * 3, unsafe_allow_html=True)
    st.sidebar.write("Hüseyin Pekkan & Ata Turhan tarafından geliştirilmiştir")


if __name__ == "__main__":
    main()
