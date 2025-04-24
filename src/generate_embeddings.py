from pinecone import Pinecone, ServerlessSpec
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from decimal import Decimal
from datetime import date, time, datetime
import json

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "cenomi_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "your_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}
print(DB_CONFIG)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists and create it if not
INDEX_NAME = "cenomicore"
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Load multilingual model for embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Function to connect to PostgreSQL and fetch data
def fetch_data(query, batch_size=100):
    all_data = []
    offset = 0
    while True:
        paginated_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute(paginated_query)
            rows = cur.fetchall()
            if not rows:
                break
            columns = [desc[0] for desc in cur.description]
            batch = [dict(zip(columns, row)) for row in rows]
            all_data.extend(batch)
            print(f"Fetched {len(batch)} rows at offset {offset}, total so far: {len(all_data)}")
            offset += batch_size
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Database error at offset {offset}: {e}")
            break
    return all_data

def convert_metadata(metadata):
    converted = {}
    for key, value in metadata.items():
        if value is None:
            converted[key] = ""
        elif isinstance(value, Decimal):
            converted[key] = float(value)
        elif isinstance(value, (date, time, datetime)):
            converted[key] = value.isoformat()
        elif isinstance(value, dict) or isinstance(value, list):
            # Convert complex JSON objects (dicts and lists) to strings
            try:
                converted[key] = json.dumps(value)
            except:
                # If serialization fails, store as empty string
                converted[key] = ""
        else:
            converted[key] = value
    return converted

def upsert_embeddings(data, id_prefix, text_field_en, text_field_ar, metadata_fields):
    vectors = []
    for item in data:
        text_en = item[text_field_en] if item[text_field_en] is not None else ""
        text_ar = item[text_field_ar] if item[text_field_ar] is not None else ""

        embedding_en = model.encode(text_en).tolist()
        vector_id_en = f"{id_prefix}_{item['id']}_en"
        metadata_en = {k: item[k] for k in metadata_fields if k in item}
        metadata_en.update({"lang": "en", "type": id_prefix})
        metadata_en = convert_metadata(metadata_en)
        vectors.append({"id": vector_id_en, "values": embedding_en, "metadata": metadata_en})

        embedding_ar = model.encode(text_ar).tolist()
        vector_id_ar = f"{id_prefix}_{item['id']}_ar"
        metadata_ar = {k: item[k] for k in metadata_fields if k in item}
        metadata_ar.update({"lang": "ar", "type": id_prefix})
        metadata_ar = convert_metadata(metadata_ar)
        vectors.append({"id": vector_id_ar, "values": embedding_ar, "metadata": metadata_ar})

    batch_size = 100
    try:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        print(f"Upserted {len(vectors)} embeddings for {id_prefix}")
    except Exception as e:
        print(f"Error upserting {id_prefix} embeddings: {e}")

def main():
    # 1. Malls (previously unique_properties)
    malls_query = """
        SELECT id, unique_property_id, marketing_name AS name_en, marketing_name_ar AS name_ar, 
        city, country, mall_information
        FROM malls
    """
    malls = fetch_data(malls_query)
    upsert_embeddings(
        malls, "mall", "name_en", "name_ar",
        ["id", "unique_property_id", "name_en", "name_ar", "city", "country", "mall_information"]
    )

    # 2. Brands (previously stores)
    brands_query = """
        SELECT b.id, b.brand_id, b.brand_name_en AS name_en, b.brand_name_ar AS name_ar, 
        b.category_name AS category_en, b.category_name_ar AS category_ar,
        b.description_en, b.description_ar, bma.unique_property_id AS mall_id
        FROM brands b
        JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
    """
    brands = fetch_data(brands_query)
    upsert_embeddings(
        brands, "store", "name_en", "name_ar",
        ["id", "brand_id", "name_en", "name_ar", "category_en", "category_ar", 
         "description_en", "description_ar", "mall_id"]
    )

    # 3. Products
    products_query = """
        SELECT p.id, p.name, p.category, p.brand_id, b.brand_name_en, bma.unique_property_id AS mall_id
        FROM products p
        JOIN brands b ON p.brand_id = b.brand_id
        JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
    """
    products = fetch_data(products_query)
    # Since products table doesn't seem to have AR fields, we'll use the same field for both
    upsert_embeddings(
        products, "product", "name", "name",
        ["id", "brand_id", "name", "category", "brand_name_en", "mall_id"]
    )

    # 4. Engagements (previously offers/events)
    engagements_query = """
        SELECT e.id, e.engagement_id, e.brand_id, e.unique_property_id AS mall_id,
        e.title_en AS name_en, e.title_ar AS name_ar, e.type,
        e.description_en, e.description_ar, e.start_date, e.end_date
        FROM engagements e
    """
    engagements = fetch_data(engagements_query)
    upsert_embeddings(
        engagements, "engagement", "name_en", "name_ar",
        ["id", "engagement_id", "brand_id", "mall_id", "name_en", "name_ar", "type",
         "description_en", "description_ar", "start_date", "end_date"]
    )

    # 5. Services
    services_query = """
        SELECT s.id, s.name, s.unique_property_id AS mall_id
        FROM services s
    """
    services = fetch_data(services_query)
    # Since services table doesn't seem to have AR fields, we'll use the same field for both
    upsert_embeddings(
        services, "service", "name", "name",
        ["id", "name", "mall_id"]
    )

    print("All embeddings successfully uploaded to Pinecone!")

if __name__ == "__main__":
    main()