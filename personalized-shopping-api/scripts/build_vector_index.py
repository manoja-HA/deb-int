"""Build ChromaDB vector index from customer purchase data"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_customer_embeddings():
    """Build customer behavior embeddings and ChromaDB collection"""

    # Load purchase data
    logger.info(f"Loading purchase data from {settings.PURCHASE_DATA_PATH}")
    df = pd.read_csv(settings.PURCHASE_DATA_PATH)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

    # Group by customer
    logger.info("Aggregating customer data...")
    customer_data = df.groupby('customerid').agg({
        'customername': 'first',
        'productcategory': lambda x: list(x),
        'purchaseprice': ['mean', 'sum', 'count'],
        'purchasequantity': 'sum',
        'transactionid': 'count',
        'country': 'first'
    }).reset_index()

    customer_data.columns = ['customer_id', 'customer_name', 'categories', 'avg_price', 'total_spent', 'price_count', 'total_quantity', 'total_purchases', 'country']

    logger.info(f"Found {len(customer_data)} unique customers")

    # Create behavior text representations
    logger.info("Creating behavior text representations...")
    behavior_texts = []
    customer_ids = []
    metadatas = []

    for _, row in customer_data.iterrows():
        # Categorize frequency (behavioral pattern)
        if row['total_purchases'] > 10:
            frequency = "high"
        elif row['total_purchases'] >= 3:
            frequency = "medium"
        else:
            frequency = "low"

        # Categorize price segment (spending behavior)
        if row['avg_price'] < 200:
            price_segment = "budget"
        elif row['avg_price'] < 600:
            price_segment = "value"
        else:
            price_segment = "premium"

        # Quantity buyer classification
        if row['total_quantity'] >= 5:
            buyer_type = "quantity buyer"
        else:
            buyer_type = "selective buyer"

        # Get top categories
        from collections import Counter
        category_counts = Counter(row['categories'])
        top_categories = [cat for cat, _ in category_counts.most_common(3)]
        categories_str = ", ".join(top_categories)

        # Create rich behavior text for embedding
        # This captures: purchasing pattern, price sensitivity, category preference, quantity behavior
        behavior_text = (
            f"{row['customer_name']} is a {frequency} frequency {price_segment} price segment {buyer_type}. "
            f"Purchases in categories: {categories_str}. "
            f"Average purchase: ${row['avg_price']:.2f}, Total spent: ${row['total_spent']:.2f}, "
            f"Typical quantity: {row['total_quantity']} units across {row['total_purchases']} transactions. "
            f"Location: {row['country']}"
        )
        behavior_texts.append(behavior_text)

        # Store customer ID and metadata
        customer_ids.append(str(row['customer_id']))
        metadatas.append({
            'customer_name': row['customer_name'],
            'total_purchases': int(row['total_purchases']),
            'avg_price': float(row['avg_price']),
            'total_spent': float(row['total_spent']),
            'total_quantity': int(row['total_quantity']),
            'frequency': frequency,
            'price_segment': price_segment,
            'buyer_type': buyer_type,
            'top_category': top_categories[0] if top_categories else 'Unknown',
            'country': row['country']
        })

    # Initialize ChromaDB
    logger.info("Initializing ChromaDB...")
    chroma_dir = settings.EMBEDDINGS_DIR / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Delete existing collection if it exists
    try:
        client.delete_collection(name="customers")
        logger.info("Deleted existing collection")
    except:
        pass

    # Create collection with custom embedding function
    logger.info(f"Creating ChromaDB collection with embedding model: {settings.EMBEDDING_MODEL}")

    from chromadb.utils import embedding_functions
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )

    collection = client.create_collection(
        name="customers",
        embedding_function=embedding_function,
        metadata={"description": "Customer behavioral embeddings"}
    )

    # Add documents to collection
    logger.info(f"Adding {len(customer_ids)} customers to collection...")
    collection.add(
        documents=behavior_texts,
        metadatas=metadatas,
        ids=customer_ids
    )

    logger.info(f"Successfully added {collection.count()} customers to ChromaDB")

    # Save metadata CSV for reference
    metadata_path = settings.EMBEDDINGS_DIR / "customer_metadata.csv"
    logger.info(f"Saving metadata to {metadata_path}")

    metadata_df = customer_data[['customer_id', 'customer_name', 'total_purchases', 'avg_price']]
    metadata_df.to_csv(metadata_path, index=False)

    logger.info("Vector index build completed successfully!")
    logger.info(f"ChromaDB path: {chroma_dir}")
    logger.info(f"Metadata path: {metadata_path}")
    logger.info(f"Total customers indexed: {collection.count()}")


if __name__ == "__main__":
    try:
        build_customer_embeddings()
    except Exception as e:
        logger.error(f"Failed to build vector index: {e}", exc_info=True)
        sys.exit(1)
