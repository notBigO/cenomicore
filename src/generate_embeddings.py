from pinecone import Pinecone, ServerlessSpec
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from decimal import Decimal
from datetime import date, time

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}
print(DB_CONFIG)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists and create it if not
INDEX_NAME = "cenomi"
if INDEX_NAME not in pc.list_indexes().names():
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
def fetch_data(query):
    try:
        conn = psycopg2.connect(**DB_CONFIG)  # type: ignore
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        print(f"Database error: {e}")
        return []

def convert_metadata(metadata):
    converted = {}
    for key, value in metadata.items():
        if value is None:
            converted[key] = ""
        elif isinstance(value, Decimal):
            converted[key] = float(value)
        elif isinstance(value, (date, time)):
            converted[key] = str(value)
        else:
            converted[key] = value
    return converted

# Function to generate and upsert embeddings
def upsert_embeddings(data, id_prefix, text_field_en, text_field_ar, metadata_fields):
    vectors = []
    for item in data:
        text_en = item[text_field_en] if item[text_field_en] is not None else ""
        text_ar = item[text_field_ar] if item[text_field_ar] is not None else ""

        # English embedding
        embedding_en = model.encode(text_en).tolist()
        vector_id_en = f"{id_prefix}_{item['id']}_en"
        metadata_en = {k: item[k] for k in metadata_fields if k in item}
        metadata_en.update({"lang": "en", "type": id_prefix})
        metadata_en = convert_metadata(metadata_en)
        vectors.append({"id": vector_id_en, "values": embedding_en, "metadata": metadata_en})

        # Arabic embedding
        embedding_ar = model.encode(text_ar).tolist()
        vector_id_ar = f"{id_prefix}_{item['id']}_ar"
        metadata_ar = {k: item[k] for k in metadata_fields if k in item}
        metadata_ar.update({"lang": "ar", "type": id_prefix})
        metadata_ar = convert_metadata(metadata_ar)
        vectors.append({"id": vector_id_ar, "values": embedding_ar, "metadata": metadata_ar})

    # Upsert to Pinecone in batches
    batch_size = 100
    try:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        print(f"Upserted {len(vectors)} embeddings for {id_prefix}")
    except Exception as e:
        print(f"Error upserting {id_prefix} embeddings: {e}")

def main():
    # 1. Malls
    malls_query = """
        SELECT mall_id AS id, name_en, name_ar, location_en, location_ar, description_en, description_ar 
        FROM malls
    """
    malls = fetch_data(malls_query)
    upsert_embeddings(
        malls, "mall", "name_en", "name_ar",
        ["id", "name_en", "name_ar", "location_en", "location_ar", "description_en", "description_ar"]
    )

    # 2. Stores
    stores_query = """
        SELECT store_id AS id, mall_id, tenant_id, name_en, name_ar, category_en, category_ar, 
               location_en, location_ar, description_en, description_ar 
        FROM stores
    """
    stores = fetch_data(stores_query)
    upsert_embeddings(
        stores, "store", "name_en", "name_ar",
        ["id", "mall_id", "tenant_id", "name_en", "name_ar", "category_en", "category_ar", 
         "location_en", "location_ar", "description_en", "description_ar"]
    )

    # 3. Products
    products_query = """
        SELECT product_id AS id, store_id, name_en, name_ar, description_en, description_ar, price, currency 
        FROM products
    """
    products = fetch_data(products_query)
    upsert_embeddings(
        products, "product", "name_en", "name_ar",
        ["id", "store_id", "name_en", "name_ar", "description_en", "description_ar", "price", "currency"]
    )

    # 4. Events
    events_query = """
        SELECT event_id AS id, mall_id, name_en, name_ar, description_en, description_ar, 
               start_time, end_time, location_en, location_ar 
        FROM events
    """
    events = fetch_data(events_query)
    upsert_embeddings(
        events, "event", "name_en", "name_ar",
        ["id", "mall_id", "name_en", "name_ar", "description_en", "description_ar", 
         "start_time", "end_time", "location_en", "location_ar"]
    )

    # 5. Offers
    offers_query = """
        SELECT offer_id AS id, store_id, description_en, description_ar, start_date, end_date 
        FROM offers
    """
    offers = fetch_data(offers_query)
    upsert_embeddings(
        offers, "offer", "description_en", "description_ar",
        ["id", "store_id", "description_en", "description_ar", "start_date", "end_date"]
    )

    # 6. Services
    services_query = """
        SELECT service_id AS id, mall_id, name_en, name_ar, description_en, description_ar 
        FROM services
    """
    services = fetch_data(services_query)
    upsert_embeddings(
        services, "service", "name_en", "name_ar",
        ["id", "mall_id", "name_en", "name_ar", "description_en", "description_ar"]
    )

    # 7. Loyalty Programs
    loyalty_query = """
        SELECT loyalty_id AS id, mall_id, name_en, name_ar, description_en, description_ar, 
               points_per_purchase, redemption_rules_en, redemption_rules_ar 
        FROM loyalty_programs
    """
    loyalty_programs = fetch_data(loyalty_query)
    upsert_embeddings(
        loyalty_programs, "loyalty", "name_en", "name_ar",
        ["id", "mall_id", "name_en", "name_ar", "description_en", "description_ar", 
         "points_per_purchase", "redemption_rules_en", "redemption_rules_ar"]
    )

    # 8. Amenities
    amenities_query = """
        SELECT amenity_id AS id, mall_id, name_en, name_ar, location_en, location_ar, 
               description_en, description_ar 
        FROM amenities
    """
    amenities = fetch_data(amenities_query)
    upsert_embeddings(
        amenities, "amenity", "name_en", "name_ar",
        ["id", "mall_id", "name_en", "name_ar", "location_en", "location_ar", 
         "description_en", "description_ar"]
    )

    print("All embeddings successfully uploaded to Pinecone!")

if __name__ == "__main__":
    main()