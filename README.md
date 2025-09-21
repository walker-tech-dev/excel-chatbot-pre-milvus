# 📊 Excel Financial Chatbot - Pre-Milvus Version

## 🎯 Overview

This is the **original version** of the Excel Financial Chatbot before vector database (Milvus) integration was added. This version represents the foundational chatbot with basic RAG architecture using simple file processing and session state management.

## 🌟 Features

### ✅ What This Version Includes:
- **Basic Streamlit Interface** - Clean, user-friendly web interface
- **Excel File Processing** - Upload and process multiple Excel/CSV files
- **Simple RAG Architecture** - Basic retrieval and generation without vector storage
- **Ollama Integration** - Local LLM (Llama 3.2 3B) for chat responses
- **Session State Management** - Data stored in browser session
- **Multi-file Support** - Process multiple financial datasets
- **Real-time Chat** - Interactive Q&A about your data

### ❌ What This Version Doesn't Include:
- Vector database (Milvus) storage
- Advanced embedding strategies
- Persistent data storage
- Performance optimizations
- Multi-tier processing options
- Enhanced business intelligence features

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed with Llama 3.2 3B model
- Streamlit

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/walker-tech-dev/excel-chatbot-pre-milvus.git
cd excel-chatbot-pre-milvus
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start Ollama:**
```bash
ollama pull llama3.2:3b
ollama serve
```

4. **Run the application:**
```bash
streamlit run streamlit_app.py
```

5. **Open your browser:**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
excel-chatbot-pre-milvus/
├── streamlit_app.py          # Main Streamlit application (41KB)
├── agno_app.py              # Core application logic
├── agno_integration.py      # Business logic integration
├── requirements.txt         # Python dependencies
├── Excel/                   # Sample Excel files directory
├── README.md               # This file
└── docker-compose.yml      # Docker setup (optional)
```

## 💡 Usage

1. **Upload Files:** Use the file uploader to add your Excel/CSV files
2. **Process Data:** Click "Process Files" to analyze your data
3. **Ask Questions:** Type questions about your financial data
4. **Get Insights:** Receive AI-powered responses based on your data

### Example Questions:
- "What is the total revenue by customer?"
- "Which customers have the highest support tickets?"
- "Show me customer health scores"
- "What are the main product lines?"

## 🔄 Evolution Path

This repository represents the **starting point** of the Excel Financial Chatbot. For the full-featured version with:

- **Vector Database Integration** (Milvus)
- **Advanced Performance Optimizations** 
- **Multi-tier Processing** (Ultra-fast, Fast, Enhanced)
- **Business Intelligence Features**
- **Cross-dataset Correlation**

See the main repository: [excel-financial-chatbot](https://github.com/walker-tech-dev/excel-financial-chatbot)

## 📊 Technical Details

### Architecture
- **Frontend:** Streamlit web interface
- **Backend:** Python with Pandas for data processing
- **AI Model:** Ollama Llama 3.2 3B (local)
- **Data Storage:** Browser session state (temporary)
- **File Support:** Excel (.xlsx) and CSV files

### Limitations
- Data is not persistent (lost on page refresh)
- No vector similarity search
- Basic text processing without embeddings
- Limited cross-file correlation
- No performance optimizations for large datasets

## 🛠️ Development

This version is ideal for:
- **Learning RAG fundamentals**
- **Understanding basic chatbot architecture**
- **Prototyping financial data analysis**
- **Educational purposes**
- **Simple Excel data querying**

## 🔗 Related Projects

- **Main Repository:** [excel-financial-chatbot](https://github.com/walker-tech-dev/excel-financial-chatbot) - Full-featured version with Milvus integration
- **Comparison:** This pre-Milvus version vs the enhanced version shows the evolution from basic to advanced RAG architecture

## 📈 Performance

- **Processing Speed:** Fast for small datasets (< 1000 rows)
- **Memory Usage:** Low (session state only)
- **Scalability:** Limited (no persistent storage)
- **Query Speed:** Good for simple questions

## 🆚 Version Comparison

| Feature | Pre-Milvus (This Repo) | Full Version |
|---------|----------------------|--------------|
| Storage | Session State | Vector Database |
| Speed | Basic | Optimized (3 tiers) |
| Persistence | ❌ | ✅ |
| Embeddings | ❌ | ✅ Advanced |
| Business Intelligence | Basic | ✅ Enhanced |
| Performance | Good for small data | Scales to 32K+ records |
| Setup Complexity | Simple | Advanced |
| Learning Curve | Easy | Moderate |

## 📝 License

MIT License - Feel free to use and modify for your projects.

## 👨‍💻 Author

**walker-tech-dev**
- GitHub: [@walker-tech-dev](https://github.com/walker-tech-dev)

## 🙏 Acknowledgments

- Ollama team for the excellent local LLM framework
- Streamlit for the fantastic web framework
- Meta for the Llama models

---

*This is the foundational version - simple, educational, and effective for basic Excel data analysis with AI.*