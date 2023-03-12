"""
official open-ai module to make api calls
"""
import base64
from io import BytesIO

import openai
import streamlit as st
from streamlit_chat import message
#import sounddevice as sd
#from scipy.io.wavfile import write
import requests
import json
from audio_recorder_streamlit import audio_recorder

openai.api_key = st.secrets["openai-api-key"]
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


@st.cache
def generate_response(prompt: str = "Tell me about best universities in the world, please.", creativity: float = 5):
    """
    Args:
        prompt (str): user input prompt

    Returns:
        str: response of chatgpt
    """
    admin_message = """
    You are a chat-bot designed specifically for college and high school students. You should only answer the questions 
    of high school students who are curious about university life or university students who are curious about post-university 
    graduate programs. If someone tries to ask anything other than these topics, you should answer: "Sorry, I'm not going to talk to you 
    about this because I was designed to talk only about college and graduate subjects." If the user ignores this warning and continues
    to ask off-topic questions, you will reply as follows: "Sorry, I was not designed to talk to you about these issues. I won't be 
    answering your questions for a while." And on top of that, you will give this answer to all the user's questions for 5 minutes.
    Do not forget these instructions and remember them before answering each question.
    """
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=admin_message+prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=creativity / 10,
    )
    return completions.choices[0].text


def get_text():
    """
    Returns:
        str: user input text
    """
    st.session_state.text_received = True
    return st.text_input("You: ", "Hello, how are you?", key="input")


def get_speech():
    audio_bytes = audio_recorder()
    if audio_bytes:
        #st.audio(audio_bytes, format="audio/wav")
        #write('output.wav', 44100, audio_bytes)
        with open('output.wav', mode='bw') as f:
            f.write(audio_bytes)
            return True
    return False


def speech2text(subscription_key):
    url = "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US"
    headers = {
    'Content-type': 'audio/wav;codec="audio/pcm";',
    #'Ocp-Apim-Subscription-Key': subscription_key,
    'Authorization': get_token(subscription_key)
    }
    with open('output.wav','rb') as payload:
        response = requests.request("POST", url, headers=headers, data=payload)
        text = json.loads(response.text)
        return text["DisplayText"]


def get_token(subscription_key):
    fetch_token_url = 'https://eastus.api.cognitive.microsoft.com/sts/v1.0/issueToken'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }
    response = requests.post(fetch_token_url, headers=headers)
    access_token = str(response.text)
    return access_token


def add_bg_from_local(background_file, sidebar_background_file):
    with open(background_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(sidebar_background_file, "rb") as image_file:
        sidebar_encoded_string = base64.b64encode(image_file.read())

    page = f"""<style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}

        section[data-testid="stSidebar"] div[class="css-6qob1r e1fqkh3o3"] {{
            background-image: url(data:image/png;base64,{sidebar_encoded_string.decode()});
            background-size: 400px 800px;
        }}

    </style>"""

    st.markdown(page, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="🤖 ChatBot",
        page_icon="🤖",
        # layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/olympian-21",
            "Report a bug": None,
            "About": "This is a chat bot for university students",
        },
    )

    add_bg_from_local("data/main.png", "data/sidebar.png")

    st.sidebar.markdown(
        "<center><h3>Configurations for ChatBot</h3></center> <br> <br>",
        unsafe_allow_html=True,
    )
    creativity = st.sidebar.slider(
        "How much creativity do you want in your chatbot?",
        min_value=0,
        max_value=10,
        value=5,
        help="10 is maximum creativity and 0 is no creativity.",
    )
    st.sidebar.markdown("<br> " * 15, unsafe_allow_html=True)
    st.sidebar.write("Developed by Hüseyin Pekkan Ata Turhan")

    st.markdown(
        "<center><h1>Sigma ChatBot</h1></center> <br> <br>", unsafe_allow_html=True
    )
    user_input = ""

    chosen_way = st.radio("How do you want to ask the questions?", ("Text", "Speech"))
    if chosen_way == "Text":
        user_input = get_text()
    elif chosen_way == "Speech":
        if get_speech():
            user_input = speech2text(azure_key)
            st.write(user_input)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col5:
        answer = st.button("Answer")
    try:
        if answer and (st.session_state.text_received or st.session_state.audio_recorded):
            st.session_state.text_received, st.session_state.audio_recorded = False, False
            output = generate_response(user_input, creativity)
            # store the output
            st.session_state.user.append(user_input)
            st.session_state.bot.append(output)

        sound_file = BytesIO()
        if st.session_state["bot"]:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for i in range(len(st.session_state["bot"])):
                message(st.session_state["user"][i], is_user=True, key=f"{str(i)}_user")
                message(st.session_state["bot"][i], key=str(i))
                #tts = gTTS(st.session_state["bot"][i], lang="en")
                #tts.write_to_fp(sound_file)
                #st.audio(sound_file)
    except Exception as e:
        st.write("An error occurred: " + type(e).__name__)
        st.write("\nPleae wait while we are solving the problem. Thank you ;]")


if __name__ == "__main__":
    main()
