Carina Rodriguez

EN.705.743.8VL.FA25

11/21/2025

Note: This project was developed with the assistance of the Cursor AI code editor.


# Retrieval-Augmented Generation Strategies for Text-to-SQL with CodeLlama: A Comparative Analysis

This project evaluates Retrieval-Augmented Generation (RAG) for Text-to-SQL using the Spider dataset. I tested several forms of retrieved context, including schema-only input, retrieved examples, and combined prompts. I also compared dense semantic retrieval with BM25 and evaluated the effect of Chain-of-Thought (CoT) prompting. All experiments used a locally deployed CodeLlama model.

## Project Overview

This system translates natural language database questions into SQL queries using:
- **Retrieval-Augmented Generation (RAG)** to provide relevant context
- **Two retrieval methods**: dense embeddings and BM25 keyword matching
- **Chain-of-Thought prompting (CoT)** for improved reasoning
- **Local LLM integration** via Ollama (CodeLlama)

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- ~2GB disk space for Spider dataset
- GPU recommended for embeddings (but CPU works)

### Step 1: Setup Environment

```bash
# Navigate to project directory
cd text2sql

# Create virtual environment
python -m venv text2sql_env

# Activate environment
# Windows PowerShell:
.\text2sql_env\Scripts\Activate

# Linux/Mac:
source text2sql_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 3: Download Spider Dataset

1. Download Spider dataset from: https://yale-lily.github.io/spider
2. Extract to `spider_data/` folder in project root
3. Verify structure:
```
spider_data/
├── dev.json              # Development set (1,034 examples)
├── train_spider.json     # Training set (7,000 examples)
├── tables.json           # Database schemas
└── database/             # SQLite databases
    ├── concert_singer/
    ├── pets_1/
    └── ... (166 databases total)
```

### Step 4: Install and Configure Ollama

```bash
# Install Ollama from https://ollama.ai

# Pull CodeLlama model (optimized for code/SQL generation)
ollama pull codellama

# Verify it's running
ollama ps
```

## Project Architecture

### Core Source Code (`src/`)

#### **Data Processing Module** (`src/data_processing/`)
- `load_spider.py`: Handles all Spider dataset operations
  - Loads train/dev/test splits
  - Parses database schemas from JSON
  - Formats schemas into readable text for LLM context
  - Maps questions to their databases

#### **Retrieval Module** (`src/retrieval/`)
- `embedder.py`: Creates semantic embeddings using sentence-transformers
  - Converts text to dense vectors (384 dimensions)
  - Supports batch processing for efficiency
  - GPU acceleration when available
  
- `vector_store.py`: FAISS-based vector storage and similarity search
  - Builds efficient index for 7,000+ training examples
  - Performs fast k-NN search for similar questions
  - Handles index serialization/loading
  
- `retriever.py`: Coordinates retrieval pipeline
  - Combines embedder and vector store
  - Filters results by database
  - Formats retrieved examples for prompts
  
- `bm25_retriever.py`: Keyword-based BM25 retrieval
  - Traditional IR approach for comparison
  - Term frequency-based matching
  - No neural networks required

#### **Generation Module** (`src/generation/`)
- `ollama_client.py`: Interface to Ollama LLM
  - Sends prompts to CodeLlama model
  - Extracts SQL from model responses
  - Handles connection and error cases
  
- `prompt_builder.py`: Constructs standard prompts
  - Baseline prompts (question only)
  - RAG prompts (with schema/examples)
  - Consistent formatting across experiments
  
- `cot_prompt_builder.py`: Chain-of-Thought prompting
  - Guides model through reasoning steps
  - Encourages systematic query construction
  - Reduces common SQL errors

#### **Evaluation Module** (`src/evaluation/`)
- `sql_executor.py`: Executes and validates SQL
  - Runs queries on actual SQLite databases
  - Compares result sets for correctness
  - Handles execution errors gracefully

#### **Main Pipeline** (`src/`)
- `text2sql_pipeline.py`: Orchestrates the complete system
  - Combines all modules into unified pipeline
  - Supports multiple configuration modes
  - Handles both baseline and RAG approaches

### Scripts (`scripts/`)

#### **Index Building Scripts**
- `build_index.py`: Creates FAISS index from training data
  - Embeds all 7,000 training questions
  - Builds optimized search structure
  - Saves to `indices/` folder
  
- `build_bm25_index.py`: Creates BM25 index
  - Tokenizes training questions
  - Builds term frequency index
  - Saves as pickle file

#### **Experiment Scripts**
- `run_all_experiments.py`: Comprehensive evaluation
  - Tests all 6 methods systematically
  - Saves results after each method
  - Generates comparison summary
  
- `test_bm25.py`: BM25-specific evaluation
  - Tests keyword retrieval approach
  - Detailed per-example results
  
- `test_cot.py`: Chain-of-Thought evaluation
  - Tests reasoning-based prompting
  - Compares against standard prompting
  
- `test_k_values.py`: Parameter optimization
  - Tests different numbers of retrieved examples
  - Finds optimal k value

#### **Analysis Scripts**
- `analyze_all_patterns.py`: Cross-method analysis
  - Identifies which queries each method solves
  - Finds universally hard/easy problems
  - Generates detailed comparison metrics

### Data Directories

- `spider_data/`: Spider dataset (not included in repo)
- `indices/`: Saved search indices
  - `faiss_index.bin`: Dense embedding index
  - `metadata.pkl`: Associated metadata
  - `bm25_index.pkl`: BM25 search index
- `results/`: Experiment outputs
  - JSON files with detailed results
  - Per-method and summary statistics

## 🔄 Workflow

### 1. One-Time Setup
```bash
# Build search indices from training data
python scripts/build_index.py
python scripts/build_bm25_index.py
```

### 2. Run Experiments
```bash
# Test all methods
python scripts/run_all_experiments.py

# Or test specific approaches
python scripts/test_bm25.py
python scripts/test_cot.py
```

### 3. Analyze Results
```bash
python scripts/analyze_all_patterns.py
```

## Methodology

### Retrieval Strategies
1. **Dense Embeddings**: Semantic similarity using neural networks
2. **BM25**: Classic keyword matching using term statistics

### Generation Configurations
1. **Baseline**: No context, direct question → SQL
2. **Schema Only**: Database structure provided
3. **Retrieval Without Context(Examples Only)**: Similar question-SQL pairs provided
4. **Full RAG**: Both schema and examples
5. **BM25 RAG**: Same as RAG but with keyword retrieval
6. **CoT RAG**: RAG with step-by-step reasoning

### Evaluation Approach
- Execute generated SQL on actual databases
- Compare result sets (not SQL syntax)
- Measure execution accuracy

### Results Analysis
- Analyze success/failure patterns using detailed results from all methods.


## Contributing

Areas for potential improvement:
- Fine-tuning models specifically on Spider
- Advanced prompting techniques
- Schema linking algorithms
- Query decomposition strategies
- Error analysis and recovery

## References

- [Spider Dataset](https://yale-lily.github.io/spider): Yale's Text-to-SQL benchmark
- [Sentence Transformers](https://www.sbert.net/): Semantic embeddings
- [FAISS](https://github.com/facebookresearch/faiss): Efficient similarity search
- [Ollama](https://ollama.ai/): Local LLM deployment
- [BM25](https://en.wikipedia.org/wiki/Okapi_BM25): Classic retrieval algorithm

