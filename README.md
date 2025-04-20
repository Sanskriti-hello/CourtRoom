# CourtRoom
CourtRoom is a Python-based application designed to simulate courtroom trials, providing an interactive platform for legal education, mock trials, or entertainment purposes. Utilizing Streamlit for the user interface, it offers a dynamic environment to model courtroom proceedings.

## 🚀 Features

- 📜 **Case-Based Scene Generation**  
  Upload a CSV of past case details and input a new case paragraph to generate realistic courtroom scenes.

- 🧠 **LLM Integration**  
  Powered by HuggingFace models for intelligent scene generation.

- 🧑‍⚖️ **Agent-Based Simulation**  
  Different courtroom roles (Judge, Plaintiff, Defendant) represented as agents.

- ⚡ **FAISS or Similar Search DB**  
  Fast and relevant case retrieval from uploaded database for context-aware generation.

  https://courtroom-o9uubxkeuyeraqfp3exnpt.streamlit.app/
  

  ## 📂 Project Structure

  CourtRoom/
├── agents/                 # Contains agent definitions and logic
├── database/               # Database models and interactions
├── __init__.py             # Package initializer
├── requirements.txt        # Python dependencies
├── run_trial.py            # Script to initiate trial simulations
├── streamlit_app.py        # Main Streamlit application
└── README.md               # Project documentation

✨ Acknowledgements
HuggingFace Transformers

FAISS for vector similarity search

Streamlit for UI


