from pinecone import Pinecone, ServerlessSpec
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from decimal import Decimal
from datetime import date, time, datetime
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

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

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "cenomicore"
qdrant_client = QdrantClient(url=QDRANT_URL)
if COLLECTION_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE)
    )

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

def upsert_embeddings(data, id_prefix, text_field_en, text_field_ar, metadata_fields=None):
    points = []
    for item in data:
        text_en = item[text_field_en] if item[text_field_en] is not None else ""
        text_ar = item[text_field_ar] if item[text_field_ar] is not None else ""

        if metadata_fields is None:
            metadata_fields = item.keys()

        embedding_en = model.encode(text_en).tolist()
        vector_id_en = f"{id_prefix}_{item['id']}_en"
        metadata_en = {k: item[k] for k in metadata_fields if k in item}
        metadata_en.update({"lang": "en", "type": id_prefix})
        metadata_en = convert_metadata(metadata_en)
        points.append(qdrant_models.PointStruct(
            id=vector_id_en,
            vector=embedding_en,
            payload=metadata_en
        ))

        embedding_ar = model.encode(text_ar).tolist()
        vector_id_ar = f"{id_prefix}_{item['id']}_ar"
        metadata_ar = {k: item[k] for k in metadata_fields if k in item}
        metadata_ar.update({"lang": "ar", "type": id_prefix})
        metadata_ar = convert_metadata(metadata_ar)
        points.append(qdrant_models.PointStruct(
            id=vector_id_ar,
            vector=embedding_ar,
            payload=metadata_ar
        ))

    batch_size = 100
    try:
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        print(f"Upserted {len(points)} embeddings for {id_prefix}")
    except Exception as e:
        print(f"Error upserting {id_prefix} embeddings: {e}")

def main():
    # 1. Malls (previously unique_properties)
    malls_query = """
        SELECT id, unique_property_id, marketing_name AS name_en, marketing_name_ar AS name_ar, 
        city, country, mall_information, image, gps_coordinates, property_group_id,
        created_at, updated_at
        FROM malls
    """
    malls = fetch_data(malls_query)
    upsert_embeddings(
        malls, "mall", "name_en", "name_ar",
        ["id", "unique_property_id", "name_en", "name_ar", "city", "country", "mall_information", 
         "image", "gps_coordinates", "property_group_id", "created_at", "updated_at"]
    )

    # 2. Brands (previously stores)
    brands_query = """
        SELECT b.id, b.brand_id, b.brand_name_en AS name_en, b.brand_name_ar AS name_ar, 
        b.category_name AS category_en, b.category_name_ar AS category_ar,
        b.description_en, b.description_ar, b.company_name_en, b.company_name_ar,
        b.group_name, b.group_name_ar, b.tenant_profile_id, b.brand_profile_id,
        b.store_phone_code, b.store_phone_number, b.store_email, b.store_website,
        b.publish_date, b.is_published, b.anchor_brand, b.brand_logo,
        b.social_tiktok, b.social_instagram, b.social_facebook, b.social_threads,
        b.social_twitter, b.social_snapchat, b.social_youtube,
        b.banner_en, b.banner_ar, b.images_en, b.images_ar, b.tags_en, b.tags_ar,
        b.pms_unit_codes, b.created_at, b.updated_at,
        bma.unique_property_id AS mall_id
        FROM brands b
        JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
    """
    brands = fetch_data(brands_query)
    upsert_embeddings(
        brands, "store", "name_en", "name_ar",
        ["id", "brand_id", "name_en", "name_ar", "category_en", "category_ar", 
         "description_en", "description_ar", "company_name_en", "company_name_ar",
         "group_name", "group_name_ar", "tenant_profile_id", "brand_profile_id",
         "store_phone_code", "store_phone_number", "store_email", "store_website",
         "publish_date", "is_published", "anchor_brand", "brand_logo",
         "social_tiktok", "social_instagram", "social_facebook", "social_threads",
         "social_twitter", "social_snapchat", "social_youtube",
         "banner_en", "banner_ar", "images_en", "images_ar", "tags_en", "tags_ar", 
         "pms_unit_codes", "created_at", "updated_at", "mall_id"]
    )

    # 3. Products
    products_query = """
        SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
        p.is_featured, p.in_stock, p.image_url, p.attributes, p.created_at, p.updated_at,
        b.brand_name_en, bma.unique_property_id AS mall_id
        FROM products p
        JOIN brands b ON p.brand_id = b.brand_id
        JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
    """
    products = fetch_data(products_query)
    # Since products table doesn't seem to have AR fields, we'll use the same field for both
    upsert_embeddings(
        products, "product", "name", "name",
        ["id", "brand_id", "name", "description", "price", "category", "brand_name_en", "mall_id",
         "is_featured", "in_stock", "image_url", "attributes", "created_at", "updated_at"]
    )

    # 4. Engagements (previously offers/events)
    engagements_query = """
        SELECT e.id, e.engagement_id, e.brand_id, e.unique_property_id AS mall_id,
        e.title_en AS name_en, e.title_ar AS name_ar, e.type,
        e.description_en, e.description_ar, e.terms_conditions_en, e.terms_conditions_ar,
        e.start_date, e.end_date, e.publish_date, e.is_exclusive, e.ext_url,
        e.home_banner_disp, e.images_en, e.images_ar, e.tags_en, e.tags_ar,
        e.tenant_profile_id, e.created_at, e.updated_at
        FROM engagements e
    """
    engagements = fetch_data(engagements_query)
    upsert_embeddings(
        engagements, "engagement", "name_en", "name_ar",
        ["id", "engagement_id", "brand_id", "mall_id", "name_en", "name_ar", "type",
         "description_en", "description_ar", "terms_conditions_en", "terms_conditions_ar",
         "start_date", "end_date", "publish_date", "is_exclusive", "ext_url", 
         "home_banner_disp", "images_en", "images_ar", "tags_en", "tags_ar",
         "tenant_profile_id", "created_at", "updated_at"]
    )

    # 5. Services
    services_query = """
        SELECT s.id, s.name, s.name_ar, s.description, s.description_ar, s.icon_url,
        s.is_available, s.location, s.unique_property_id AS mall_id,
        s.created_at, s.updated_at
        FROM services s
    """
    services = fetch_data(services_query)
    # Using name_ar if available, falling back to name if not
    upsert_embeddings(
        services, "service", "name", "name_ar",
        ["id", "name", "name_ar", "description", "description_ar", "icon_url",
         "is_available", "location", "mall_id", "created_at", "updated_at"]
    )

    print("All embeddings successfully uploaded to Pinecone!")

if __name__ == "__main__":
    main()