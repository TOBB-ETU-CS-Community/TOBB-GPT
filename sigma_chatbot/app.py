import json
import os
import time

import openai
import requests
import speech_recognition as s_r
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
     ise türkçe, ingilizce ise ingilizce bir cevap üret ve cevabı json formatında döndür. Json formatı şöyle olmalı:
     {"query": output}
     """
    user_message = f"""Dönüştürmen gereken soru, tek tırnak işaretleri arasındadır:
     '{question}'
     Verdiğin cevap da yalnızca arama sorgusu yer almalı, başka herhangi bir şey yazmamalı ve tırnak işareti gibi
     bir noktalama işareti de eklememelisin. Sonucu json formatında dönmelisin."""

    user_message += """Json formatı şöyle olmalı:
     {"query": output}
     """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    # response = openai.Completion.create(
    #    engine="gpt-3.5-turbo", prompt=messages, max_tokens=100
    # )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    json_object = json.loads(response.choices[0].message.content)
    # st.write(json_object["query"])
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


def is_api_key_valid(openai_api_key: str):
    if openai_api_key is None or not openai_api_key.startswith("sk-"):
        st.warning("Lütfen geçerli bir OpenAI API Key'i girin!", icon="⚠")
        return False
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True


def show_chat_ui():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
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

            with st.spinner("Soru internet üzerinde aranıyor:"):
                query = transform_question(st.session_state.text_box)
                query = query.replace('"', "").replace("'", "")

            with st.spinner(
                "Soru internet üzerindeki kaynaklar ile cevaplanıyor:"
            ):
                retriever, urls = create_vector_store_retriever(query)
                qa = create_retrieval_qa(prompt_template, llm, retriever)

            llm_output = ""
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                responses = qa.run(user_input)

                source_output = (
                    "\n\n Soru, şu kaynaklardan yararlanarak cevaplandı: \n\n"
                )
                for url in urls:
                    source_output += url + " \n\n "
                responses += source_output
                for response in responses:
                    llm_output += response
                    message_placeholder.markdown(f"{llm_output}▌")
                    time.sleep(0.01)
                message_placeholder.markdown(llm_output)

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

    # add_bg_from_local("input/main.png", "input/sidebar.png")

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
