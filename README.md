---

<div align="center">

[![Sourcery](https://img.shields.io/badge/Sourcery-enabled-brightgreen)](https://sourcery.ai)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

<hr>

# Sigma ChatBot

It is a chatbot powered by OpenAI, Langchain, ChromeDB, and Azure APIs to educate students about specific colleges.

https://sigma-chatbot.streamlit.app/

https://sigma-chatbot.azurewebsites.net/

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Prerequisites

Before setting up the project, ensure that you have the following prerequisites installed on your machine:

- Python (version 3.10.10):
  (https://www.python.org/downloads/)
- Poetry (version 1.4.0):
  (https://python-poetry.org/docs/#installation)

## Installation

1. Clone the project repository:

```
git clone https://github.com/TOBB-ETU-CS-Community/Sigma-Chatbot.git
```

2. Navigate to the project directory:

```
cd Sigma-Chatbot
```

3. Install project dependencies using Poetry:

```
poetry install
```

This command will create a virtual environment and install all the necessary dependencies specified in the `pyproject.toml` file.

## Usage

1. Activate the virtual environment created by Poetry:

```
poetry shell
```

2. Run the project:

```
streamlit run "src/sigma_chatbot/app.py"
```
