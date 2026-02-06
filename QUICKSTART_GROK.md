# 🚀 Quick Start Guide - Run with Grok

## ✅ What's Already Done

1. ✅ Python dependencies installed
2. ✅ `.env` file configured for Grok
3. ✅ Grok setup script created (`setup_rag_grok.py`)
4. ✅ Documentation created (`README_GROK.md`)

## 🔑 What You Need To Do

### Step 1: Get Your FREE Grok API Key

1. Visit: **https://console.x.ai/**
2. Sign in with your X (Twitter) account  
3. Click "Create API Key"
4. Copy the API key (starts with `xai-`)

### Step 2: Add API Key to .env File

Open `.env` and replace this line:
```
OPENAI_API_KEY=your-grok-api-key-here
```

With your actual key:
```
OPENAI_API_KEY=xai-your-actual-key-here
```

**Important:** Keep these lines in the .env file:
```
OPENAI_API_BASE=https://api.x.ai/v1
LLM_MODEL=grok-beta
USE_QDRANT=true
```

### Step 3: Install sentence-transformers

Run this command:
```powershell
pip install sentence-transformers
```

### Step 4: Run the Setup

Choose one of these options:

**Quick Test (Recommended First):**
```powershell
python setup_rag_grok.py --quick
```
This processes 10 transcripts in ~5-10 minutes

**Medium Test:**
```powershell
python setup_rag_grok.py --max 50
```
This processes 50 transcripts in ~30-40 minutes

**Full Build:**
```powershell
python setup_rag_grok.py
```
This processes all 297 transcripts in ~2-3 hours

### Step 5: Launch the App

After setup completes, run:
```powershell
streamlit run streamlit_app.py
```

The web interface will open at http://localhost:8501

## 🎯 What to Expect

### During Setup
1. Script checks your Grok API key
2. Installs sentence-transformers (if needed)
3. Installs and starts Qdrant vector database
4. Downloads embedding model (~90MB, first run only)
5. Processes transcripts one by one
6. Tests with a sample query

### After Setup
You can:
- Use the beautiful Streamlit web interface
- Query from command line
- View the knowledge graph
- Browse transcripts

## 💰 Costs

**Total Cost: $0 (COMPLETELY FREE!)**

- Grok API: FREE ✨
- Local embeddings: FREE
- Qdrant: FREE
- Transcripts: FREE

## 📚 Full Documentation

Full details in: **README_GROK.md**

## ⚡ Need Help?

Common issues:

**"Grok API key not set"**
→ Make sure you added your key to `.env` file

**"sentence-transformers not found"**
→ Run: `pip install sentence-transformers`

**"Qdrant not accessible"**
→ Run: `.\install_qdrant_windows.ps1` then restart setup

## 🎉 That's It!

You're all set to run a production-ready RAG system for FREE using Grok!
