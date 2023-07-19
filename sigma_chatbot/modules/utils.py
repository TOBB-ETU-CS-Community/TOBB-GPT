import base64

import streamlit as st


@st.cache_data
def add_bg_from_local(background_img_path, sidebar_background_img_path):
    with open(background_img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(sidebar_background_img_path, "rb") as image_file:
        sidebar_encoded_string = base64.b64encode(image_file.read())

    return f"""<style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}

        section[data-testid="stSidebar"] {{
            background-image: url(data:image/png;base64,{sidebar_encoded_string.decode()});
            background-size: cover;
        }}
        div[class="stChatFloatingInputContainer css-90vs21 ehod42b2"]
            {{
                background: url(data:image/png;base64,{encoded_string.decode()});
                background-size: cover;
                z-index: 1;
            }}
    </style>"""


def set_page_config():
    st.set_page_config(
        page_title="Sigma ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/olympian-21",
            "Report a bug": "https://github.com/olympian-21",
            "About": "This is a chatbot that answers questions scraping the web.",
        },
    )


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    style = """<style>
        .row-widget.stButton {
            text-align: center;
            position: fixed;
            bottom: 0;
            z-index: 2;
            }
    </style>"""
    st.markdown(style, unsafe_allow_html=True)
