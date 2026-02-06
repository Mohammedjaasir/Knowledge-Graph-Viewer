# 🕸️ LennyHub RAG - Knowledge Graph Viewer

> A production-ready RAG (Retrieval-Augmented Generation) system with interactive knowledge graph visualization, powered by **Grok's FREE API** and local embeddings.

![Knowledge Graph](https://img.shields.io/badge/Graph-544_People-blue) ![Connections](https://img.shields.io/badge/Connections-391-green) ![Cost](https://img.shields.io/badge/Cost-FREE-gold) ![Status](https://img.shields.io/badge/Status-Production_Ready-success)

## 🌟 Live Demo

- **Interactive Knowledge Graph**: Explore 544 people and their connections from Lenny's Podcast
- **AI-Powered Q&A**: Ask questions about product management, growth, and leadership
- **Beautiful UI**: Modern Streamlit interface with real-time search

## ✨ Key Features

### 🎨 Interactive Knowledge Graph
- **544 people** from Lenny's podcast network
- **391 connections** between guests and experts
- **Interactive visualization** with drag-and-drop nodes
- **Search functionality** to find specific people
- **Top connectors highlighted** with color-coded importance
- **Click for details** on any person or connection

### 🤖 AI-Powered RAG System
- **Grok API** for LLM (text generation) - **FREE!**
- **Local embeddings** (sentence-transformers) - **FREE!**
- **Qdrant** vector database for production-grade search
- **Multiple search modes**: Hybrid, Local, Global, Naive
- **297 podcast transcripts** fully indexed

### 📊 Web Interface (Streamlit)
- Beautiful, modern UI
- Real-time query system
- Statistics dashboard
- Transcript browser
- Sample questions included

## 💰 Cost Breakdown

| Component | Traditional (OpenAI) | This Project |
|-----------|---------------------|--------------|
| **LLM API** | $0.20-6.00 | **FREE** ✨ |
| **Embeddings** | $0.04-1.20 | **FREE** (local) |
| **Vector DB** | FREE | FREE |
| **Total Build** | ~$7.20 | **$0.00** 🎉 |
| **Per Query** | $0.001-0.01 | **$0.00** |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (3.11 recommended)
- Windows, macOS, or Linux
- FREE Grok API key from [console.x.ai](https://console.x.ai/)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mohammedjaasir/Knowledge-Graph-Viewer.git
cd Knowledge-Graph-Viewer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
pip install sentence-transformers
```

3. **Configure environment**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your FREE Grok API key
# Get it from: https://console.x.ai/
```

Example `.env`:
```bash
# Your Grok API Key (FREE from console.x.ai)
OPENAI_API_KEY=gsk-your-grok-api-key-here

# Grok API Configuration
OPENAI_API_BASE=https://api.x.ai/v1
LLM_MODEL=grok-beta

# Qdrant Configuration
USE_QDRANT=true
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=lennyhub
```

4. **Install Qdrant** (vector database)

**Windows:**
```powershell
PowerShell -ExecutionPolicy Bypass -File .\install_qdrant_windows.ps1
```

**Mac/Linux:**
```bash
./install_qdrant_local.sh
```

5. **Build the RAG system**

**Quick test (10 transcripts, ~5-10 min):**
```bash
python setup_grok_simple.py --quick
```

**Full build (297 transcripts, ~2-3 hours):**
```bash
python setup_grok_simple.py
```

6. **Launch the applications**

**Streamlit Web UI:**
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**Knowledge Graph Viewer:**
```bash
python serve_graph.py
# Opens at http://localhost:8000/graph_viewer_simple.html
```

## 📖 Usage

### Interactive Knowledge Graph

Visit `http://localhost:8000/graph_viewer_simple.html` to explore:

**Features:**
- 🎯 **Interactive Network**: Drag and reposition nodes
- 🔍 **Smart Search**: Find people with exact match prioritization
- 👆 **Clickable Legend**: Focus on top connectors
- 🖱️ **Controls**: Zoom, pan, and click for details
- 🌟 **Top Connectors**:
  - Lenny Rachitsky (292 connections) - Gold
  - Bob Moesta (98 connections) - Amber
  - April Dunford (85 connections) - Orange
  - Arielle Jackson (82 connections) - Deep Orange
  - Andrew Wilkinson (78 connections) - Red

### Query System (Streamlit UI)

Visit `http://localhost:8501` to:

**Ask Questions:**
- "What is a curiosity loop and how does it work?"
- "What are the best frameworks for product management?"
- "How do you build trust with your team?"
- "What is the growth competency model?"
- "Should you start a company with your partner?"

**Search Modes:**
- **Hybrid** (Recommended) - Best overall results
- **Local** - Entity-focused, fast and precise
- **Global** - Relationship-focused, broader context
- **Naive** - Pure vector similarity, fastest

## 🏗️ Architecture

```
User Query
    ↓
Streamlit UI / Knowledge Graph
    ↓
RAG System (RAG-Anything)
    ↓
LightRAG (Knowledge Graph Engine)
    ↓
├─→ Grok API (FREE LLM)
├─→ Local Embeddings (sentence-transformers)
└─→ Qdrant Vector Storage
    ↓
Hybrid Search (local + global + vector)
    ↓
Answer Synthesis
    ↓
Results with Sources
```

## 📁 Project Structure

```
lennyhub-rag/
├── 📊 Data & Storage
│   ├── data/                         # 297 podcast transcripts
│   ├── rag_storage/                  # Knowledge graph & metadata
│   └── qdrant_storage/               # Vector embeddings
│
├── 🚀 Setup & Configuration
│   ├── setup_grok_simple.py          # Grok-powered setup (Windows-friendly)
│   ├── install_qdrant_windows.ps1    # Install Qdrant (Windows)
│   ├── install_qdrant_local.sh       # Install Qdrant (Mac/Linux)
│   ├── .env                          # API keys & settings
│   └── requirements.txt              # Python dependencies
│
├── 🎨 User Interfaces
│   ├── streamlit_app.py              # Visual web interface
│   ├── serve_graph.py                # Knowledge graph server
│   ├── graph_viewer_simple.html      # Interactive graph visualization
│   └── query_rag.py                  # CLI query interface
│
├── 🔧 Configuration
│   ├── grok_config.py                # Grok API configuration
│   ├── qdrant_config.py              # Qdrant configuration
│   └── qdrant_config.yaml            # Qdrant settings
│
└── 📚 Documentation
    ├── README.md                     # Main documentation
    ├── README_GROK.md                # Grok setup guide
    ├── QUICKSTART_GROK.md            # Quick start guide
    └── GRAPH_VIEWER_README.md        # Graph viewer docs
```

## 🛠️ Technical Stack

### Core Technologies
- **RAG Framework**: [RAG-Anything](https://github.com/HKUDS/RAG-Anything) v1.2.9+
- **Knowledge Graph**: [LightRAG](https://github.com/HKUDS/LightRAG) v1.4.9+
- **Vector Database**: [Qdrant](https://qdrant.tech/) v1.16+
- **Web UI**: Streamlit 1.28+
- **Language**: Python 3.8+

### AI/ML
- **LLM**: Grok (xAI) via OpenAI-compatible API - **FREE!**
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 - **FREE!**
  - 384 dimensions
  - Runs locally, no API calls
  - ~90MB model download

## 📊 Performance

### Build Time
| Transcripts | Time (Sequential) |
|------------|-------------------|
| 10 (quick) | 5-10 minutes |
| 50 | 30-40 minutes |
| 297 (all) | 2-3 hours |

### Query Performance
- **First query**: ~2-5 seconds (Grok API call + vector search)
- **Cached queries**: Instant
- **Embedding generation**: Local (no network latency)

## 🔒 Privacy & Security

- ✅ All embeddings generated **locally** (no data sent for embedding)
- ✅ Only queries sent to Grok API
- ✅ Qdrant runs **locally** (no cloud dependency)
- ✅ All transcripts stored **locally**
- ✅ No user data tracking

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Add more transcript sources
- Improve entity extraction
- Add multi-language support
- Enhance visualization features
- Optimize embedding performance

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🙏 Credits

- **Transcripts**: [Lenny's Podcast](https://www.lennysnewsletter.com/podcast)
- **RAG Framework**: [RAG-Anything](https://github.com/HKUDS/RAG-Anything) by HKUDS
- **Knowledge Graph**: [LightRAG](https://github.com/HKUDS/LightRAG) by HKUDS
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **LLM**: [Grok](https://x.ai/) by xAI
- **Embeddings**: [sentence-transformers](https://www.sbert.net/)

## 🆘 Troubleshooting

### Grok API Issues
```bash
# Check API key is set
cat .env | grep OPENAI_API_KEY

# Verify API base is correct
cat .env | grep OPENAI_API_BASE
# Should be: https://api.x.ai/v1
```

### Qdrant Issues
```bash
# Check if running
curl http://localhost:6333/

# Restart Qdrant (Windows)
cd %USERPROFILE%\.qdrant
.\qdrant.exe --config-path <path>\qdrant_config.yaml

# View dashboard
http://localhost:6333/dashboard
```

### Embedding Model Issues
```bash
# Reinstall sentence-transformers
pip install sentence-transformers --upgrade

# Clear cache
rm -rf ~/.cache/torch/sentence_transformers/
```

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/Mohammedjaasir/Knowledge-Graph-Viewer/issues)
- **Grok API**: [XAI Docs](https://docs.x.ai/)
- **Grok Console**: [console.x.ai](https://console.x.ai/)

## ⭐ Star This Repo

If you find this project useful, please give it a star! It helps others discover this FREE RAG solution.

---

**Built with ❤️ using Grok (xAI), sentence-transformers, Qdrant, and Streamlit**

**Total Cost: $0** 🎉
