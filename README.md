# 🎓 EduPlanner AI
> Local Student Assistant & Dynamic Quiz Generator  
> **Stack:** Python 3.10+ · Streamlit · Google Gemini 1.5 Flash · SQLite3

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Run with API key in environment (recommended)
GOOGLE_API_KEY="AIzaSy..." streamlit run app.py

# 2b. OR run and enter key in the sidebar
streamlit run app.py
```

Get a **free** Gemini API key at: https://aistudio.google.com/app/apikey

---

## Features

| Feature | Details |
|---|---|
| 📂 **High-Capacity Ingestion** | Support for manual text pasting or uploading local academic files **up to 200MB** directly from PC storage. |
| 📖 **Study Planner** | Analyzes document difficulty → generates a tailored 3-step study plan with dynamic insights. |
| 🧠 **Interactive Quiz** | 5 MCQs (2 easy · 2 medium · 1 hard) with radio buttons, no pre-selection (`index=None`), and instant ✅/❌ visual feedback. |
| 💾 **Markdown Export (.md)** | Allows users to instantly download their generated study plans as local `.md` files for syncing with Obsidian, Notion, or Logseq. |
| 📊 **Progress Tracker** | SQLite3 stores every quiz result; sidebar seamlessly displays score history, averages, and timestamps. |
| 🔐 **API Key Security** | Supports standard environment variables (`GOOGLE_API_KEY`) or sidebar password-masked input. |
| 🔄 **State Management** | `st.session_state` structures ensure quiz selections and user states survive native Page Reruns. |
| 🛡️ **Robust Error Handling** | 3-strategy JSON parser (direct → regex → fence-strip), heavy validation, and strict `try/except` safety blocks on all LLM pipelines. |

---

## Architecture

