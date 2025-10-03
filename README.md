# RAG Automotive Analysis

A Retrieval-Augmented Generation (RAG) system for analyzing automotive company annual reports. This application allows you to query financial information from BMW, Tesla, and Ford annual reports using natural language.

## Features

- **Natural Language Queries**: Ask questions in plain English about financial metrics
- **Multi-Company Support**: Query data across BMW, Tesla, and Ford
- **Multi-Year Analysis**: Compare financial data across different years
- **Conversational Interface**: Follow-up questions with context awareness
- **Accurate Retrieval**: Vector-based semantic search for relevant information

## Project Structure

```
rag-automotive-analysis/
├── data/
│   ├── raw/                    # Original PDFs
│   │   ├── BMW/*.pdf
│   │   ├── Tesla/*.pdf
│   │   └── Ford/*.pdf
│   └── processed/              # Vector store
│       └── chroma_db/
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # PDF loading & chunking
│   ├── vector_store.py         # Embedding & retrieval
│   ├── rag_engine.py           # Main RAG logic
│   └── chat_interface.py       # Terminal interface
├── .env.template               # template for setting environment variables
├── requirements.txt
├── .gitignore
├── setup.py                    # One-time setup script
├── main.py                     # Entry point to chat interface
├── README.md
└── test_queries.py             # Script to run all provided test queestions
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Annual reports for BMW, Tesla, and Ford

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rishideepc/rag-automotive-analysis
   cd rag-automotive-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows Powershell: .\venv\Scripts\activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file from template
   cp .env.template .env
   
   # Edit .env and add your OpenAI API key
   nano .env  # example
   ```

5. **Add PDF files**
   Place your annual report PDFs in the following structure:
   ```
   data/raw/
   ├── BMW/
   │   ├── BMW_Annual_Report_2021.pdf
   │   ├── BMW_Annual_Report_2022.pdf
   │   └── BMW_Annual_Report_2023.pdf
   ├── Tesla/
   │   ├── Tesla_Annual_Report_2022.pdf
   │   └── Tesla_Annual_Report_2023.pdf
   └── Ford/
       ├── Ford_Annual_Report_2021.pdf
       ├── Ford_Annual_Report_2022.pdf
       └── Ford_Annual_Report_2023.pdf
   ```

6. **Run setup**
   ```bash
   python setup.py
   ```
   This will:
   - Verify the environment
   - Process all PDF documents
   - Create embeddings
   - Build the vector store

## Usage

### Option 1: Interactive Chat Interface (Recommended)

Start the application:
```bash
python main.py
```

### Option 2: Test All Sample Questions

Run all test queries from the provided sample:
```bash
python test_queries.py
```

### Example Queries

**Simple Factual Queries:**
- "What was BMW's total revenue in 2023?"
- "How much revenue did Tesla generate in 2023?"
- "What was Ford's revenue for the year 2020?"

**Comparison Queries:**
- "Between Tesla and Ford, which company achieved higher profits in 2022?"
- "Compare BMW and Tesla revenue in 2023"
- "Which company had better profitability in 2022?"

**Trend & Summary Queries:**
- "What were the growth trends for BMW from 2020 to 2023?"
- "Provide a summary of revenue for all companies over the past three years"
- "How has Tesla's profit changed from 2022 to 2023?"

**Qualitative Queries:**
- "What key economic factors influenced Ford's performance in 2021?"
- "Which Tesla product is currently in development?"
- "What are BMW's strategic priorities for 2023?"

### Interactive Commands

In the chat interface, you can use these commands:
- Type your question and press Enter to get an answer
- `exit`, `quit`, or `q` - Close the application
- `clear` - Clear conversation history
- `examples` - Show example questions
- `help` - Display help information

## Configuration

Edit `.env` to customize:

```bash
# Model Configuration
OPENAI_MODEL=gpt-4o                  # or any other model
EMBEDDING_MODEL=text-embedding-ada-002

# Chunking Configuration
CHUNK_SIZE=1000                      # example size of text chunks
CHUNK_OVERLAP=200                    # example overlap between chunks
```

## Troubleshooting

### No PDF files found
- Ensure PDFs are in the correct directory structure
- Check that filenames match the expected pattern

### OpenAI API errors
- Verify your API key is correct in `.env`
- Check you have sufficient API credits
- Ensure you're not rate limited

### Vector store issues
- Delete `data/processed/chroma_db/` and run `python setup.py` again
- Check disk space

## Technical Details

### Technology Stack
- **LangChain**: RAG framework
- **ChromaDB**: Vector database
- **OpenAI**: Embeddings and LLM
- **PyPDF**: PDF parsing

### How It Works
1. **Document Processing**: PDFs are loaded and split into chunks
2. **Embedding**: Text chunks are converted to vector embeddings
3. **Storage**: Embeddings are stored in ChromaDB
4. **Query**: User question is converted to embedding
5. **Retrieval**: Most relevant chunks are retrieved
6. **Generation**: LLM generates answer based on retrieved context
