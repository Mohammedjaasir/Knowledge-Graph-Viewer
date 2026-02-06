# 🚀 Running LennyHub RAG with Grok (FREE!)

This guide shows you how to run the LennyHub RAG system using **Grok's FREE API** instead of OpenAI.

## 🎁 What's Different?

**Original Setup:**
- OpenAI GPT-4o-mini (paid)
- OpenAI embeddings (paid)
- Cost: ~$7 to build, $0.001-0.01 per query

**Grok Setup:**
- **Grok API for LLM** (FREE! ✨)
- **Local embeddings** (sentence-transformers, FREE!)
- **Cost: $0** 🎉

## 📋 Quick Start

### Step 1: Get Your FREE Grok API Key

1. Go to [https://console.x.ai/](https://console.x.ai/)
2. Sign in with your X (Twitter) account
3. Create an API key
4. Copy the API key

### Step 2: Configure the Environment

Open the `.env` file and add your Grok API key:

```bash
# Your Grok API Key
OPENAI_API_KEY=xai-your-grok-api-key-here

# Grok API Base URL (required!)
OPENAI_API_BASE=https://api.x.ai/v1

# Model to use
LLM_MODEL=grok-beta

# Keep these settings
USE_QDRANT=true
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=lennyhub
```

### Step 3: Install Additional Dependencies

The Grok setup uses local embeddings, so we need sentence-transformers:

```powershell
pip install sentence-transformers
```

### Step 4: Run the Grok Setup Script

```powershell
# Quick test (first 10 transcripts)
python setup_rag_grok.py --quick

# Process first 50 transcripts
python setup_rag_grok.py --max 50

# Process all 297 transcripts
python setup_rag_grok.py
```

**What this does:**
- ✅ Checks your Grok API key
- ✅ Installs sentence-transformers (if needed)
- ✅ Installs and starts Qdrant
- ✅ Builds the RAG system using Grok
- ✅ Tests with a sample query

### Step 5: Use the System

Once setup is complete, you can query the system:

**Via Streamlit Web UI:**
```powershell
streamlit run streamlit_app.py
```

**Via Command Line:**
```powershell
python query_rag.py "What is a curiosity loop?"
python query_rag.py --interactive
```

## 🔧 Technical Details

### What Changed?

1. **LLM (Text Generation):** 
   - Instead of OpenAI's GPT-4o-mini
   - Now using Grok's API (compatible with OpenAI client)

2. **Embeddings (Vector Search):**
   - Instead of OpenAI's text-embedding-3-small (1536 dims)
   - Now using sentence-transformers/all-MiniLM-L6-v2 (384 dims)
   - Runs locally, no API calls needed

3. **API Configuration:**
   - Changed `openai.base_url` to point to `https://api.x.ai/v1`
   - Model name changed to `grok-beta` or `grok-2-1212`

### Available Grok Models

- `grok-beta` - Latest beta version (recommended, free)
- `grok-2-1212` - Stable release (December 2024)
- Check [XAI docs](https://docs.x.ai/) for latest models

### Local Embedding Models

The `sentence-transformers/all-MiniLM-L6-v2` model is used because:
- ✅ **Fast**: 120 sentences/sec on CPU
- ✅ **Small**: Only ~90MB download
- ✅ **Accurate**: Good balance of speed vs quality
- ✅ **Free**: Runs completely offline

**Alternatives you can try:**
- `all-mpnet-base-v2` - More accurate, slower (768 dims)
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support
- Any model from [HuggingFace](https://huggingface.co/sentence-transformers)

To change the model, edit `setup_rag_grok.py` line 137:
```python
embedding_model = SentenceTransformer('your-model-name-here')
```

## ⚙️ Configuration Options

### Environment Variables (.env)

```bash
# Required - Your Grok API key
OPENAI_API_KEY=xai-your-key-here

# Required - Grok API endpoint
OPENAI_API_BASE=https://api.x.ai/v1

# Optional - Model selection
LLM_MODEL=grok-beta

# Optional - Qdrant settings (recommended)
USE_QDRANT=true
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=lennyhub
```

## 🐛 Troubleshooting

### "Grok API key not set"

Make sure you've:
1. Added your key to `.env`
2. Set `OPENAI_API_BASE=https://api.x.ai/v1`
3. The key starts with `xai-`

### "sentence-transformers not found"

Install it:
```powershell
pip install sentence-transformers
```

### "Qdrant not accessible"

Install Qdrant:
```powershell
# Windows
.\\install_qdrant_windows.ps1

# Mac/Linux
./install_qdrant_local.sh
```

Then start it:
```powershell
# Windows
cd $HOME\\.qdrant
.\\qdrant.exe

# Mac/Linux
./start_qdrant.sh
```

### Slow embedding generation

First-time embedding generation downloads the model (~90MB). Subsequent runs are much faster.

For even faster embeddings, reduce batch size or use a smaller model:
```python
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller, faster
```

### API errors with Grok

Check that:
1. Your API key is valid
2. You have sufficient quota (free tier limits)
3. The model name is correct (try `grok-beta`)

## 📊 Performance Comparison

### OpenAI vs Grok Setup

| Metric | OpenAI | Grok Setup |
|--------|--------|------------|
| **LLM Cost** | $0.20-6.00 | **FREE** |
| **Embedding Cost** | $0.04-1.20 | **FREE** |
| **Total Build Cost** | ~$7.20 | **$0.00** |
| **Query Cost** | $0.001-0.01 | **$0.00** |
| **Embedding Dims** | 1536 | 384 |
| **Embedding Speed** | API latency | Local (fast) |
| **LLM Quality** | GPT-4o-mini | Grok-beta |

### Build Time Estimates

| Transcripts | Sequential | 
|------------|------------|
| 10 (quick) | ~5-10 min |
| 50 | ~30-40 min |
| 297 (all) | ~2-3 hours |

*Note: Local embeddings are actually faster than API calls since there's no network latency!*

## 🎯 Next Steps

After setup:

1. **Try the web interface:**
   ```powershell
   streamlit run streamlit_app.py
   ```

2. **Query from CLI:**
   ```powershell
   python query_rag.py "What are the best product management frameworks?"
   ```

3. **View the knowledge graph:**
   ```powershell
   python serve_graph.py
   ```

4. **Check Qdrant dashboard:**
   Open [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## 💡 Tips

- Start with `--quick` to test the setup (10 transcripts in ~5 min)
- Use `--max 50` for a good balance of coverage and speed
- The first run downloads the embedding model (~90MB)
- Subsequent queries are cached and instant
- You can still use the original OpenAI setup if needed

## 🆘 Need Help?

- Grok API docs: [https://docs.x.ai/](https://docs.x.ai/)
- Grok console: [https://console.x.ai/](https://console.x.ai/)
- Sentence-transformers: [https://www.sbert.net/](https://www.sbert.net/)
- Original README: [README.md](README.md)

## 🎉 Enjoy Your FREE RAG System!

You now have a production-ready RAG system powered by:
- 🤖 **Grok** (cutting-edge LLM from xAI)
- 📊 **Local embeddings** (fast & free)
- 🗄️ **Qdrant** (production vector DB)
- 📚 **297 podcast transcripts** (Lenny's Newsletter)

All for **$0**! 🎊
