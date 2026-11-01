# SmartNotes AI – Your Intelligent Study Assistant

Upload PDF study notes and let AI summarize, explain and quiz you.
Built with **Python**, **Streamlit**, the **Groq API** and **pypdf**.

## Features

- PDF upload with text extraction (invalid, locked and scanned PDFs are handled with clear messages)
- AI summary: short, medium or detailed, in simple English with headings
- Important points: key concepts, definitions and key terms
- MCQ generator: four options, correct answers and explanations, with scoring
- Ask Your Notes: chat that answers only from your PDF and says so when the answer is missing
- Flashcards: study one card at a time
- TXT downloads for every result and one complete study pack

## Project structure

```
smartnotes-ai/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml          (theme colours – safe to upload)
│   └── secrets.toml         (YOUR API KEY – create it yourself, never upload it)
└── utils/
    ├── __init__.py
    ├── ai_engine.py
    └── pdf_reader.py
```

## Run locally

1. Install Python 3.10 or newer from https://www.python.org/downloads/ (tick "Add Python to PATH" on Windows).
2. Open a terminal inside the `smartnotes-ai` folder.
3. Create and activate a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install the libraries: `pip install -r requirements.txt`
5. Get a free Groq key at https://console.groq.com/keys (sign in, click **Create API Key**, copy it).
6. Create the file `.streamlit/secrets.toml` and put this inside (use your own key):
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
7. Start the app: `streamlit run app.py` – it opens at http://localhost:8501

## Deploy on Streamlit Community Cloud

1. Upload the project to a **public or private GitHub repository** (do not upload `secrets.toml`; `.gitignore` already blocks it).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**. After a minute your app has a public link.

## Optional settings

Add this line to your secrets to choose a different Groq model:

```toml
GROQ_MODEL = "llama-3.1-8b-instant"
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "API key is not set up" | Check the name is exactly `GROQ_API_KEY` in secrets. |
| "free Groq limit was reached" | Wait one minute and try again. |
| "No readable text was found" | The PDF is scanned images. Use a PDF with selectable text. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with your virtual environment active. |
