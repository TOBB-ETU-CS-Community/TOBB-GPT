import json
import os
from io import BytesIO

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

if "audio_recorded" not in st.session_state:
    st.session_state.audio_recorded = False

if "text_received" not in st.session_state:
    st.session_state.text_received = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None


def get_text(
    prompt: str = "Bana TOBB Üniversitesi hakkında bilgi verebilir misin?",
) -> str:
    """generates a message for the conversation

    Parameters
    ----------
    prompt : str
        Default message.

    Returns
    -------
    str
        User input for prompting the model
    """
    st.session_state.text_received = True
    return st.text_input("You: ", prompt, key="input")


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
    if audio_bytes := audio_recorder():
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
    url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US"
    headers = {
        "Content-type": 'audio/wav;codec="audio/pcm";',
        # 'Ocp-Apim-Subscription-Key': subscription_key,
        "Authorization": get_token(subscription_key, region),
    }
    with open("output.wav", "rb") as payload:
        response = requests.request("POST", url, headers=headers, data=payload)
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
    st.write("google namussuz")
    result = tool.run(query)
    st.write(result)
    urls = [val["link"] for val in result]
    st.write(urls)
    st.write("google şerefsiz")
    loader = WebBaseLoader(urls)
    documents = loader.load()
    st.write(documents)
    char_text_splitter = MarkdownTextSplitter(
        chunk_size=2048,
        chunk_overlap=128,
    )
    texts = char_text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    return vector_store.as_retriever()


def transform_question(question):
    prompt = f"""Bu görevde yapman gereken bu şey, kullanıcı sorularını arama sorgularına dönüştürmektir. Bir kullanıcı soru sorduğunda,
      soruyu, kullanıcının bilmek istediği bilgileri getiren bir Google arama sorgusuna dönüştürürsünüz.
      Dönüştürmen gereken soru, tek tırnak işaretleri arasındadır:
     '{question}'
     Verdiğin cevap da yalnızca arama sorgusu yer almalı, başka herhangi bir şey yazmamalısın.
     """
    model = "text-davinci-003"
    response = openai.Completion.create(
        engine=model, prompt=prompt, max_tokens=100
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
        st.warning("Please enter a valid OpenAI API key!", icon="⚠")
        return False
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True


def show_chat_ui():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt_template = """
    <|SYSTEM|>#
    - Sen Türkçe konuşan bir botsun. Soru Türkçe ise her zaman Türkçe cevap vermelisin.
    - If the question is in English, then answer in English. If the question is Turkish, then answer in Turkish.
    - You are a helpful, polite, fact-based agent for answering questions about TOBB University based on provided context and chat history.
    - The user just asked you a question about this context or chat history. Answer it using the information contained in the context or chat history.
    - If the question is not about universities, say that you only answer questions about universities.
    <|USER|>
    Please answer the following question using the context provided or chat history.

    QUESTION: {question}
    CONTEXT:
    {context}

    ANSWER: <|ASSISTANT|>
    """

    user_input = ""
    # region = "switzerlandwest"#huseyin
    region = "eastus"  # ata
    chosen_way = st.radio(
        "How do you want to ask the questions?", ("Text", "Speech")
    )
    if chosen_way == "Text":
        user_input = st.text_input(
            "Please write your question in the textbox below: ", key="input"
        )
    elif chosen_way == "Speech":
        if get_speech():
            user_input = speech2text(os.environ["AZURE_S2T_KEY"], region)
            st.write(user_input)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col5:
        answer = st.button("Answer")
    key = os.environ["AZURE_S2T_KEY"]
    config = sdk.SpeechConfig(subscription=key, region=region)
    config.speech_synthesis_language = "en-US"
    config.speech_synthesis_voice_name = "en-US-JennyNeural"
    speech_synthesizer = sdk.SpeechSynthesizer(
        speech_config=config, audio_config=None
    )

    # synthesizer = sdk.SpeechSynthesizer(speech_config=config)

    # input_text = st.text_input(
    #    "Please write a text to convert it to a speech:"
    # )
    # if st.button("test azure text to speech") and input_text is not None:
    #    result = speech_synthesizer.speak_text(input_text)
    #    st.audio(result.audio_data)
    # audioStream = sdk.AudioDataStream(result)
    # display(audioElement)

    try:
        if answer:
            st.session_state.text_received, st.session_state.audio_recorded = (
                False,
                False,
            )
            query = transform_question(st.session_state.input)
            user_input = st.session_state.input
            # query = st.session_state.input
            st.write(query)
            query = query.replace('"', "").replace("'", "")
            st.write(query)
            retriever = create_vector_store_retriever(query)
            qa = create_retrieval_qa(prompt_template, llm, retriever)
            output = qa.run(user_input)
            # store the output
            st.session_state.user.append(user_input)
            st.session_state.bot.append(output)

        BytesIO()
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
        center_err_col.error(
            "\nPlease wait while we are solving the problem. Thank you ;]"
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
        "<center><h1>Sigma ChatBot</h1></center> <br> <br>",
        unsafe_allow_html=True,
    )

    add_bg_from_local("input/main.png", "input/sidebar.png")

    st.sidebar.markdown(
        "<center><h3>Configurations for ChatBot</h3></center> <br> <br>",
        unsafe_allow_html=True,
    )
    openai_api_key = st.sidebar.text_input("Please enter the OpenAI API Key:")
    if st.sidebar.button("Use this OPEN AI Api Key"):
        st.session_state.openai_api_key = openai_api_key

    if is_api_key_valid(st.session_state.openai_api_key):
        st.sidebar.success("This OpenAI Api Key was used successfully.")
        show_chat_ui()

    st.sidebar.markdown("<br> " * 3, unsafe_allow_html=True)
    st.sidebar.write("Developed by Hüseyin Pekkan & Ata Turhan")


if __name__ == "__main__":
    main()
