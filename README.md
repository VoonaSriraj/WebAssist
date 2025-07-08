
#  WebAssist – RAG-Powered AI Web Assistant

WebAssist is a **Retrieval-Augmented Generation (RAG)** based chatbot application built using **Streamlit**, **LangChain**, **FAISS**, and **ChatGroq's LLMs**. This app allows users to ask questions related to content scraped from the web (e.g., GeeksforGeeks C++ documentation) and get context-aware answers in real time.


---

##  Features

*  **Web Scraping** using `WebBaseLoader`
*  **HuggingFace Embeddings** for semantic understanding
*  **FAISS Vector Store** for efficient document retrieval
*  **ChatGroq (LLaMA-3)** based LLM responses
*  Fast and reliable Q\&A experience
*  View relevant document chunks used in answers
*  Built with **LangChain** pipelines and RAG architecture

---

##  Tech Stack

| Component       | Technology                           |
| --------------- | ------------------------------------ |
| UI Framework    | Streamlit                            |
| Embeddings      | HuggingFace (`all-MiniLM-L6-v2`)     |
| Vector Store    | FAISS                                |
| Language Model  | ChatGroq (`llama-3.3-70b-versatile`) |
| Text Splitter   | RecursiveCharacterTextSplitter       |
| Prompt Template | LangChain's ChatPromptTemplate       |
| Docs Source     | GeeksforGeeks - C++ Page             |

---

##  Installation

### Prerequisites

* Python 3.8+
* `pip`
* `virtualenv` (optional)
* `groq` API key (free from [https://console.groq.com](https://console.groq.com))

### 1. Clone the Repository

```bash
git clone https://github.com/VoonaSriraj/WebAssist.git
cd WebAssist
```

### 2. Create and Activate Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup `.env` File

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

##  Running the App

```bash
streamlit run app.py
```

Open the URL provided in the terminal to interact with the chatbot.

---

##  Usage

1. Enter a question in the text input box (e.g., "What is a pointer in C++?")
2. The model will search the GeeksforGeeks C++ documentation using vector similarity.
3. It will generate an answer based on the most relevant chunks.
4. You can expand the "📄 View Source Documents" section to view retrieved context.

---

##  Project Structure

```
WebAssist/
├── app.py               # Main Streamlit app
├── .env                 # Contains API key (not committed)
├── requirements.txt     # Python dependencies
└── README.md            # You're here
```

---

##  Contributing

Pull requests are welcome! Feel free to open issues for feature requests or bug reports.

---

##  License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [FAISS](https://github.com/facebookresearch/faiss)
* [HuggingFace Transformers](https://huggingface.co)
* [Groq](https://console.groq.com)
* [Streamlit](https://streamlit.io)

---

