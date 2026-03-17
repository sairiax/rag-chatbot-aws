# 🎓 RAG University

A production-ready **Retrieval-Augmented Generation (RAG)** system built with LangChain, AWS Bedrock and ChromaDB. Supports multi-format document ingestion (PDF, Word, PowerPoint, images with OCR), rich metadata indexing, conversational memory and a Streamlit chatbot interface.

---

## 💼 Business Context & Use Case

### 🎯 Target Group
- **Legal Departments**: Corporate legal teams managing high volumes of correspondence and contracts.
- **Law Firms**: Attorneys performing due diligence or litigation discovery across multiple document types.
- **Compliance Teams**: Professionals auditing communications for regulatory requirements.

### 🛠️ Jobs to be Done (JTBD)
*“Help me extract actionable insights from thousands of disparate emails and documents, so that I can reduce manual review time by 80% and focus on legal strategy rather than data retrieval.”*

### 🚀 Key Use Cases
1. **Litigation Support**: Quickly find specific mentions of concepts or evidence across years of email history.
2. **Due Diligence**: Analyze data rooms (PDFs, PPTs, Docs) for specific clauses or risks during M&A.
3. **Knowledge Management**: Create a centralized, searchable brain from historical legal archives.

---

## 🔄 Application Flow

The system follows a linear pipeline from "messy" data to conversational insights:

1.  **Data Prep (Cleaning)**: Raw CSV exports are converted into standardized `.txt` files to preserve metadata and structure.
2.  **Multimodal Ingestion**: Smart loaders handle PDF, Word, PowerPoint, and Text.
3.  **OCR Engine (AWS Textract)**: Automatically triggers for scanned documents or images.
4.  **Vector Store (ChromaDB)**: High-performance indexing using AWS Bedrock Embeddings.
5.  **Conversational RAG**: Claude 3.5 Sonnet generates answers with full source citations and session memory.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│              app/streamlit_app.py                       │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   ConversationalRAGChain    │  ← LangChain LCEL + memory
          │   src/chains/rag_chain.py   │
          └──┬──────────────────────┬───┘
             │                      │
  ┌──────────▼──────┐     ┌─────────▼────────┐
  │  SmartRetriever │     │  AWS Bedrock LLM  │
  │  (with filter)  │     │  (Claude / Nova)  │
  └──────────┬──────┘     └──────────────────┘
             │
  ┌──────────▼──────────┐
  │   ChromaVectorStore  │  ← persistent on disk
  └──────────┬──────────┘
             │
  ┌──────────▼──────────┐
  │  AWS Bedrock Embed.  │  ← Titan / Cohere
  └─────────────────────┘

Ingestion pipeline (offline):
  Documents → MultiFormatDocumentLoader → OCRProcessor (optional)
            → MetadataAwareTextSplitter → ChromaVectorStore
```

---

## Project structure

```
rag-university/
├── .env.example              # Template for environment variables
├── .gitignore
├── requirements.txt
│
├── config/
│   └── settings.py           # Pydantic-settings config (reads from .env)
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py  # PDF · Word · PPT · TXT · images
│   │   ├── ocr_processor.py    # Tesseract OCR (scanned docs & images)
│   │   └── text_splitter.py    # Recursive splitter + metadata enrichment
│   │
│   ├── clean/
│   │   └── data_cleaning.py    # NEW: CSV to TXT converter for legacy emails
│   │
│   ├── embeddings/
│   │   └── aws_embeddings.py   # BedrockEmbeddings factory
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py     # ChromaDB wrapper (CRUD + retriever)
│   │
│   ├── retrieval/
│   │   └── retriever.py        # SmartRetriever with metadata filters
│   │
│   ├── chains/
│   │   └── rag_chain.py        # Conversational RAG + per-session memory
│   │
│   └── utils/
│       └── helpers.py          # Format helpers, filter builder
│
├── app/
│   └── streamlit_app.py        # Full Streamlit UI
│
├── scripts/
│   └── index_documents.py      # CLI batch indexing tool
│
└── data/
    ├── raw/                    # Put raw CSV/files here
    ├── clean/                  # Target for cleaned files (TXT)
    └── chroma_db/              # ChromaDB persisted storage
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.11 and 3.12 |
| AWS account | With Amazon Bedrock model access enabled |
| AWS Textract | Needed for OCR on scanned PDFs / image files |
| Poppler | Required by `pdf2image` for PDF→image conversion |

### Enable AWS Textract

Textract is a managed AWS service, so no local OCR binary is required.

1. Ensure the AWS user/role has `textract:DetectDocumentText` permission.
2. Use a region where Textract is available.
3. Keep your AWS credentials configured in `.env`.

### Install Poppler

**Ubuntu / Debian**
```bash
sudo apt-get install poppler-utils
```

**macOS**
```bash
brew install poppler
```

**Windows** — Download from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows) and add `bin/` to PATH.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/rag-university
cd rag-university

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your AWS credentials and model IDs
```

### Enable AWS Bedrock model access

Go to **AWS Console → Bedrock → Model access** and request access for:
- `amazon.titan-embed-text-v2:0` (or your chosen embedding model)
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (or your chosen LLM)

---

## 🧹 Data Pre-processing

If you are working with legacy email exports (CSVs), you must clean and standardize them before indexing:

1. Place your CSV files (e.g., `correos_legales.csv`) in `data/raw/`.
2. Run the cleaning script:
   ```bash
   python3 src/clean/data_cleaning.py
   ```
3. The script will generate standardized `.txt` files in `data/clean/`, ready for indexing.

---

## Usage

### Option A — Streamlit chatbot (recommended)

```bash
streamlit run app/streamlit_app.py
```

1. Use the sidebar to **upload documents** (PDF, Word, PowerPoint, TXT, images).
2. Toggle **OCR** if you have scanned docs.
3. Click **Indexar documentos** to embed and store them.
4. Start chatting! The bot remembers the conversation.
5. Use the **filter** dropdown to restrict retrieval to a specific file type.

### Option B — CLI batch indexing

```bash
# Index cleaned documents in data/clean/
python scripts/index_documents.py --dir data/clean

# Index raw documents (with OCR)
python scripts/index_documents.py --dir data/raw --ocr

# Clear existing index and re-index
python scripts/index_documents.py --dir data/clean --clear
```

---

## Supported document formats

| Format | Extension(s) | OCR fallback |
|---|---|---|
| PDF | `.pdf` | ✅ (scanned pages detected automatically) |
| Word | `.docx`, `.doc` | ❌ |
| PowerPoint | `.pptx`, `.ppt` | ❌ |
| Plain text | `.txt`, `.md` | ❌ |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` | ✅ (required) |

---

## Metadata indexed per chunk

Every chunk stored in ChromaDB carries:

| Field | Description |
|---|---|
| `source` | Full path or original filename |
| `filename` | Base filename |
| `file_type` | Extension without dot (`pdf`, `docx`, …) |
| `page` | Page number (PDFs) |
| `slide_number` | Slide number (PowerPoint) |
| `slide_title` | Slide title (PowerPoint) |
| `chunk_index` | Position within the parent document |
| `total_chunks` | Total chunks from the parent document |
| `chunk_size` | Character count of this chunk |
| `ocr_applied` | `True` if OCR was used to extract text |

Use metadata filters in retrieval:
```python
from src.utils.helpers import build_metadata_filter

filter_dict = build_metadata_filter(file_type="pdf")
retriever = smart_retriever.get_retriever(filter_dict=filter_dict)
```

---

## Configuration reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | — | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `LLM_MODEL_ID` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Bedrock LLM |
| `LLM_TEMPERATURE` | `0.7` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Max tokens in response |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `rag_university` | Collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Documents retrieved per query |

---

## Extending the project

**Add a new document format**
1. Implement a `_load_<ext>` method in `MultiFormatDocumentLoader`.
2. Register the extension in `_EXT_TO_METHOD`.

**Switch to a different vector store** (e.g. Pinecone, OpenSearch)
- Replace `ChromaVectorStore` with a new class exposing the same interface (`add_documents`, `as_retriever`, `get_collection_stats`).

**Use a different LLM / embeddings provider**
- Update `src/embeddings/aws_embeddings.py` and `src/chains/rag_chain.py` to use the desired `langchain_*` integration package.

**Add reranking**
- Wrap the retriever in `langchain.retrievers.ContextualCompressionRetriever` with a cross-encoder reranker.

---

## License

MIT
