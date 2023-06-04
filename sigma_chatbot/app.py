"""
official open-ai module to make api calls
"""
import base64
import json
import os
from io import BytesIO

import azure.cognitiveservices.speech as sdk
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from streamlit_chat import message

azure_key = st.secrets["azure-s2t-key"]

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


def get_text(prompt: str = "Hello, how are you?") -> str:
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


def add_bg_from_local(
    background_file: str, sidebar_background_file: str
) -> None:
    """set the background images

    Parameters
    ----------
    background_file : str
        path to the background
    background_file : str
        path to the sidebar background

    Returns
    -------
    None
    """
    with open(background_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(sidebar_background_file, "rb") as image_file:
        sidebar_encoded_string = base64.b64encode(image_file.read())

    page = f"""<style>
        .stApp {{
            background-image: url(input:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}

        section[data-testid="stSidebar"] div[class="css-6qob1r e1fqkh3o3"] {{
            background-image: url(input:image/png;base64,{sidebar_encoded_string.decode()});
            background-size: 400px 800px;
        }}

    </style>"""

    st.markdown(page, unsafe_allow_html=True)
    return


def create_vector_store_retriever(file_path):
    FULL_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(FULL_PATH, "Vector-DB")
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )
    vector_store = None

    main_dir = os.path.dirname(FULL_PATH)
    FILE_DIR = os.path.join(main_dir, file_path)
    loader = TextLoader(FILE_DIR, encoding="utf-8")
    documents = loader.load()
    char_text_splitter = MarkdownTextSplitter(
        chunk_size=2048,
        chunk_overlap=512,
    )
    texts = char_text_splitter.split_documents(documents)

    if not os.path.exists(DB_DIR):
        vector_store = Chroma.from_documents(
            texts,
            # embeddings,
            collection_name="Store",
            persist_directory=DB_DIR,
            client_settings=client_settings,
        )
        vector_store.persist()
    else:
        vector_store = Chroma(
            # embedding_function=embeddings,
            collection_name="Store",
            persist_directory=DB_DIR,
            client_settings=client_settings,
        )
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )


def create_retrieval_qa(prompt_template, llm, retriever):
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )


def is_api_key_valid(openai_api_key: str):
    if openai_api_key is None or not openai_api_key.startswith("sk-"):
        st.warning("Please enter a valid OpenAI API key!", icon="⚠")
        return False
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True


def show_chat_ui(creativity: int):
    file_path = "input/tobb.csv"
    retriever = create_vector_store_retriever(file_path)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=creativity / 10)

    prompt_template = """
    <|SYSTEM|>#
    - Sen Türkçe konuşan bir botsun. Soru Türkçe ise her zaman Türkçe cevap vermelisin.
    - If the question is in English, then answer in English. If the question is Turkish, then answer in Turkish.
    - You are a helpful, polite, fact-based agent for answering questions about TOBB University based on provided context.
    - The user just asked you a question about this context. Answer it using the information contained in the context.
    - If the question is not about universities, say that you only answer questions about universities.
    <|USER|>
    Please answer the following question using the context provided. Soru Türkçe ise sen de Türkçe cevap vermelisin.

    QUESTION: {question}
    CONTEXT:
    {context}

    ANSWER: <|ASSISTANT|>
    """

    qa = create_retrieval_qa(prompt_template, llm, retriever)

    user_input = ""
    # region = "switzerlandwest"#huseyin
    region = "eastus"  # ata
    chosen_way = st.radio(
        "How do you want to ask the questions?", ("Text", "Speech")
    )
    if chosen_way == "Text":
        user_input = get_text()
    elif chosen_way == "Speech":
        if get_speech():
            user_input = speech2text(azure_key, region)
            st.write(user_input)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col5:
        answer = st.button("Answer")
    key = azure_key
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
        if answer and (
            st.session_state.text_received or st.session_state.audio_recorded
        ):
            st.session_state.text_received, st.session_state.audio_recorded = (
                False,
                False,
            )
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
                # tts = gTTS(st.session_state["bot"][i], lang="en")
                # tts.write_to_fp(sound_file)
                # st.audio(sound_file)
    except Exception as e:
        st.write(f"An error occurred: {type(e).__name__}")
        st.write("\nPleae wait while we are solving the problem. Thank you ;]")


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

    add_bg_from_local(
        os.path.join(os.getcwd(), "input/main.png"),
        os.path.join(os.getcwd(), "input/sidebar.png"),
    )

    st.sidebar.markdown(
        "<center><h3>Configurations for ChatBot</h3></center> <br> <br>",
        unsafe_allow_html=True,
    )
    openai_api_key = st.sidebar.text_input("Please enter the OpenAI API Key:")
    if st.sidebar.button("Use this OPEN AI api key"):
        st.session_state.openai_api_key = openai_api_key

    if is_api_key_valid(st.session_state.openai_api_key):
        st.sidebar.success("This OpenAI Api Key was used successfully.")
        creativity = st.sidebar.slider(
            "How much creativity do you want in your chatbot?",
            min_value=0,
            max_value=10,
            value=5,
            help="10 is maximum creativity and 0 is no creativity.",
        )
        show_chat_ui(creativity)

    st.sidebar.markdown("<br> " * 3, unsafe_allow_html=True)
    st.sidebar.write("Developed by Hüseyin Pekkan & Ata Turhan")


if __name__ == "__main__":
    main()
