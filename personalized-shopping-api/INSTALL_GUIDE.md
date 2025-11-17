# Installation Guide

Step-by-step installation instructions for the Personalized Shopping Assistant.

## System Requirements

### Hardware
- **RAM:** Minimum 8GB (16GB recommended for LLM inference)
- **CPU:** Modern multi-core processor (4+ cores)
- **Disk:** 10GB free space (for models and data)
- **GPU:** Optional (CUDA-compatible for FAISS-GPU)

### Software
- **OS:** Linux, macOS, or Windows (WSL recommended)
- **Python:** 3.10 or higher
- **Ollama:** Latest version

## Step 1: Install System Dependencies

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y python3.10 python3-pip python3-venv git
```

### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install git
brew install git
```

### Windows (WSL)
```bash
# Open WSL terminal
sudo apt update
sudo apt install -y python3.10 python3-pip python3-venv git
```

## Step 2: Install Ollama

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS
```bash
# Download from https://ollama.com/download
# Or use Homebrew
brew install ollama
```

### Verify Ollama Installation
```bash
ollama --version
```

## Step 3: Pull Required Models

```bash
# Pull Llama 3.2 3B (fast model for profiling)
ollama pull llama3.2:3b

# Pull Llama 3.1 8B (quality model for sentiment & response)
ollama pull llama3.1:8b

# Verify models
ollama list
```

Expected output:
```
NAME                ID              SIZE      MODIFIED
llama3.2:3b         a80c4f17acd5    2.0 GB    2 minutes ago
llama3.1:8b         4fa551d4f938    4.7 GB    5 minutes ago
```

## Step 4: Clone Repository

```bash
# Clone the project (or extract if received as ZIP)
cd /your/workspace/
git clone <repository-url> personalized-shopping-assistant

# Or if you have the folder already:
cd personalized-shopping-assistant
```

## Step 5: Setup Python Environment

```bash
# Navigate to project directory
cd personalized-shopping-assistant

# Run automated setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# The script will:
# - Create virtual environment
# - Install dependencies
# - Create data directories
# - Copy .env.example to .env
```

### Manual Setup (Alternative)

If the script fails, run manually:

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{raw,processed,embeddings}
mkdir -p logs

# Copy environment template
cp .env.example .env
```

## Step 6: Configure Environment

Edit `.env` file:

```bash
nano .env  # or use your preferred editor
```

Required settings:

```env
# Ollama endpoint (default for local installation)
OLLAMA_BASE_URL=http://localhost:11434

# Data paths (adjust if different)
PURCHASE_DATA_PATH=./data/raw/customer_purchase_data.csv
REVIEW_DATA_PATH=./data/raw/customer_reviews_data.csv

# Model names (must match pulled models)
PROFILING_MODEL=llama3.2:3b
SENTIMENT_MODEL=llama3.1:8b
RECOMMENDATION_MODEL=llama3.1:8b
RESPONSE_MODEL=llama3.1:8b

# Embedding model (will auto-download from HuggingFace)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

## Step 7: Prepare Data Files

### Data Format Requirements

Create two CSV files in `data/raw/`:

**1. customer_purchase_data.csv**

Required columns:
```csv
TransactionID,CustomerID,CustomerName,ProductID,ProductName,ProductCategory,PurchaseQuantity,PurchasePrice,PurchaseDate,Country
1,887,Kenneth Martinez,101,Laptop,Electronics,1,699,2024-01-15,USA
2,887,Kenneth Martinez,102,Mouse,Electronics,2,29.99,2024-01-20,USA
```

**2. customer_reviews_data.csv**

Required columns:
```csv
ReviewID,ProductID,ReviewText,ReviewDate
1,101,Great laptop! Very fast and reliable,2024-01-16
2,101,Excellent product worth every penny,2024-01-17
```

### Sample Data

If you don't have real data, create sample files:

```bash
# Run this in project root
cat > data/raw/customer_purchase_data.csv << 'EOF'
TransactionID,CustomerID,CustomerName,ProductID,ProductName,ProductCategory,PurchaseQuantity,PurchasePrice,PurchaseDate,Country
1,887,Kenneth Martinez,101,Laptop,Electronics,1,699,2024-01-15,USA
2,887,Kenneth Martinez,102,Mouse,Electronics,2,29.99,2024-01-20,USA
3,560,Jane Smith,103,Keyboard,Electronics,1,89.99,2024-01-18,USA
4,560,Jane Smith,101,Laptop,Electronics,1,520,2024-02-01,USA
