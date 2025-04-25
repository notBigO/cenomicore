import os
import asyncio
import httpx
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, Table, UniqueConstraint, Float
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_utils import database_exists, create_database
from dotenv import load_dotenv
import re
import sys
import uuid
from sqlalchemy import Text
import random

# Import data from data.py
from al_nakheel import malls_data, brands_data, engagement_data
# Import data from u_walk.py
from u_walk import u_walk_mall_data, u_walk_brands_data, u_walk_engagement_data

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL")
API_TOKEN = os.getenv("API_TOKEN")

# Get database URL and fix connection issues
DATABASE_URL = str(os.getenv("DATABASE_URL"))

# Fix the driver issue - change from asyncpg to psycopg2 for synchronous operations
if "postgresql+asyncpg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
    print("Changed database driver from asyncpg to psycopg2 for better compatibility")

# Parse connection parameters that need special handling
connect_args = {}
if "postgresql" in DATABASE_URL:
    # Extract and remove problematic parameters
    params_to_extract = ["sslmode", "sslrootcert", "sslcert", "sslkey"]
    
    for param in params_to_extract:
        pattern = f"[?&]{param}=([^&]+)"
        match = re.search(pattern, DATABASE_URL)
        if match:
            param_value = match.group(1)
            DATABASE_URL = re.sub(pattern, '', DATABASE_URL)
            
            # Handle SSL parameters appropriately for psycopg2
            if param == "sslmode":
                connect_args["sslmode"] = param_value
            else:
                connect_args[param] = param_value

# Create the engine with the fixed URL and connect_args
if connect_args:
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
    print(f"Using database connection with SSL parameters: {connect_args}")
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create database if it doesn't exist
try:
    if not database_exists(engine.url):
        create_database(engine.url)
except Exception as e:
    print(f"Database connection error: {str(e)}")
    print(f"Current DATABASE_URL: {re.sub(r':[^:@]+@', ':***@', DATABASE_URL)}")
    print("Please check your .env file and ensure DATABASE_URL is correctly formatted.")
    print("Recommended format: postgresql://username:password@hostname/database")
    sys.exit(1)

Base = declarative_base()

# Create a many-to-many association table for brands and malls
brand_mall_association = Table(
    "brand_mall_association",
    Base.metadata,
    Column("brand_id", Integer, ForeignKey("brands.brand_id"), primary_key=True),
    Column("unique_property_id", Integer, ForeignKey("malls.unique_property_id"), primary_key=True)
)

class Mall(Base):
    __tablename__ = "malls"

    id = Column(Integer, primary_key=True)
    unique_property_id = Column(Integer, unique=True, nullable=False)
    property_group_id = Column(Integer)
    marketing_name = Column(String)
    marketing_name_ar = Column(String)
    city = Column(String)
    country = Column(String)
    mall_information = Column(JSONB)
    image = Column(String)
    gps_coordinates = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Many-to-many relationship with brands
    brands = relationship("Brand", secondary=brand_mall_association, back_populates="malls")
    # One-to-many relationship with engagements
    engagements = relationship("Engagement", back_populates="mall")
    # One-to-many relationship with services
    services = relationship("Service", back_populates="mall")


class Brand(Base):
    __tablename__ = "brands"

    id = Column(Integer, primary_key=True)
    brand_id = Column(Integer, unique=True, nullable=False)
    # Remove the direct foreign key to malls - now handled through the association table
    # unique_property_id = Column(Integer, ForeignKey("malls.unique_property_id"))
    anchor_brand = Column(Integer)
    tenant_profile_id = Column(Integer)
    brand_name_en = Column(String)
    brand_name_ar = Column(String)
    brand_logo = Column(String)
    company_name_en = Column(String)
    company_name_ar = Column(String)
    category_name = Column(String)
    category_name_ar = Column(String)
    group_name = Column(String)
    group_name_ar = Column(String)
    brand_profile_id = Column(Integer)
    store_phone_code = Column(String)
    store_phone_number = Column(String)
    store_email = Column(String)
    store_website = Column(String)
    publish_date = Column(String)
    is_published = Column(Boolean)
    social_tiktok = Column(String)
    social_instagram = Column(String)
    social_facebook = Column(String)
    social_threads = Column(String)
    social_twitter = Column(String)
    social_snapchat = Column(String)
    social_youtube = Column(String)
    description_en = Column(String)
    description_ar = Column(String)
    banner_en = Column(String)
    banner_ar = Column(String)
    images_en = Column(JSONB)
    images_ar = Column(JSONB)
    tags_en = Column(JSONB)
    tags_ar = Column(JSONB)
    pms_unit_codes = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Many-to-many relationship with malls
    malls = relationship("Mall", secondary=brand_mall_association, back_populates="brands")
    # One-to-many relationship with engagements
    engagements = relationship("Engagement", back_populates="brand")
    # One-to-many relationship with products
    products = relationship("Product", back_populates="brand")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    price = Column(Float, nullable=False)
    image_url = Column(String)
    category = Column(String)
    brand_id = Column(Integer, ForeignKey("brands.brand_id"), nullable=False)
    is_featured = Column(Boolean, default=False)
    in_stock = Column(Boolean, default=True)
    attributes = Column(JSONB)  # Store product attributes like color, size, material, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to brand
    brand = relationship("Brand", back_populates="products")


class Engagement(Base):
    __tablename__ = "engagements"

    id = Column(Integer, primary_key=True)
    # Remove unique constraint from engagement_id
    engagement_id = Column(Integer, nullable=False, index=True)
    brand_id = Column(Integer, ForeignKey("brands.brand_id"))
    unique_property_id = Column(Integer, ForeignKey("malls.unique_property_id"))
    # Add a unique constraint on the combination of engagement_id and unique_property_id
    __table_args__ = (
        UniqueConstraint('engagement_id', 'unique_property_id', name='uix_engagement_mall'),
    )
    tenant_profile_id = Column(Integer)
    title_en = Column(String)
    title_ar = Column(String)
    type = Column(String)
    description_en = Column(String)
    description_ar = Column(String)
    terms_conditions_en = Column(String)
    terms_conditions_ar = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    publish_date = Column(String)
    is_exclusive = Column(Integer)
    images_en = Column(JSONB)
    images_ar = Column(JSONB)
    tags_en = Column(JSONB)
    tags_ar = Column(JSONB)
    ext_url = Column(String)
    home_banner_disp = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    mall = relationship("Mall", back_populates="engagements")
    brand = relationship("Brand", back_populates="engagements")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = Column(Text, nullable=True)
    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan")

# Define the ConversationMessage model
class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    message_index = Column(Integer, nullable=False)  # For ordering messages

    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")


class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    name_ar = Column(String)
    description = Column(String)
    description_ar = Column(String)
    icon_url = Column(String)
    is_available = Column(Boolean, default=True)
    location = Column(String)
    unique_property_id = Column(Integer, ForeignKey("malls.unique_property_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to mall
    mall = relationship("Mall", back_populates="services")


def reset_database():
    """Drop all tables and recreate them"""
    print("Resetting database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database schema recreated.")


def generate_products_for_brand(brand):
    """Generate relevant products based on the brand's category"""
    products = []
    
    # Map brand categories to product types
    category_mapping = {
        # Fashion and clothing categories
        "Full range sports wear and equipment": ["Running Shoes", "Athletic T-Shirt", "Sports Shorts", "Training Jacket", "Fitness Tracker"],
        "Fashion accessories": ["Designer Handbag", "Luxury Wallet", "Statement Necklace", "Leather Belt", "Sunglasses"],
        "Women's shoes": ["High Heel Sandals", "Ballet Flats", "Ankle Boots", "Platform Sneakers", "Espadrilles"],
        "Lingerie - In home lounge wear": ["Silk Pajama Set", "Cotton Nightgown", "Lace Bralette", "Satin Robe", "Loungewear Set"],
        "Men's Arabic - Thobes": ["Classic Thobe", "Modern Thobe", "Embroidered Bisht", "Traditional Headwear", "Formal Sandals"],
        
        # Beauty and cosmetics categories
        "Cosmetics": ["Liquid Foundation", "Eyeshadow Palette", "Lipstick Collection", "Mascara", "Skincare Set"],
        "Pharmacy": ["Vitamin Supplements", "Skincare Products", "Hair Care Set", "Personal Care Kit", "First Aid Essentials"],
        
        # Specialty retail categories
        "Watches": ["Luxury Watch", "Sports Chronograph", "Classic Timepiece", "Smart Watch", "Limited Edition Collection"],
        "Optical": ["Designer Eyeglasses", "Prescription Sunglasses", "Contact Lenses Pack", "Blue Light Glasses", "Reading Glasses"],
        "Electronic games": ["Console Game", "Gaming Headset", "Controller", "Gaming Keyboard", "Virtual Reality Set"],
        
        # Food categories
        "Chicken Cuisine": ["Signature Chicken Meal", "Family Bucket", "Spicy Wings", "Chicken Sandwich", "Combo Meal"]
    }
    
    # Define attribute generators based on product type
    def generate_footwear_attributes():
        colors = ["Black", "White", "Red", "Blue", "Grey", "Navy", "Green", "Brown", "Multi-color"]
        sizes = ["36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46"]
        materials = ["Leather", "Canvas", "Synthetic", "Mesh", "Suede", "Gore-Tex", "Rubber", "Knit"]
        styles = ["Casual", "Formal", "Athletic", "Outdoor", "Fashion"]
        return {
            "color": random.choice(colors),
            "size": random.sample(sizes, random.randint(3, 8)),
            "material": random.choice(materials),
            "style": random.choice(styles),
            "closure_type": random.choice(["Lace-up", "Slip-on", "Zipper", "Velcro", "Buckle"]),
            "sole_material": random.choice(["Rubber", "EVA", "Phylon", "PU", "TPU"]),
            "water_resistant": random.choice([True, False]),
            "gender": random.choice(["Men", "Women", "Unisex"])
        }
    
    def generate_clothing_attributes():
        colors = ["Black", "White", "Blue", "Red", "Grey", "Navy", "Green", "Beige", "Pink", "Purple", "Yellow"]
        sizes = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
        materials = ["Cotton", "Polyester", "Nylon", "Wool", "Silk", "Linen", "Denim", "Cashmere", "Viscose", "Elastane"]
        return {
            "color": random.choice(colors),
            "size": random.sample(sizes, random.randint(3, 7)),
            "material": random.choice(materials),
            "fit": random.choice(["Regular", "Slim", "Relaxed", "Athletic", "Loose"]),
            "pattern": random.choice(["Solid", "Striped", "Checkered", "Printed", "Graphic", "Plain"]),
            "season": random.choice(["All Season", "Summer", "Winter", "Spring", "Fall"]),
            "care": "Machine wash cold, tumble dry low",
            "gender": random.choice(["Men", "Women", "Unisex"])
        }
    
    def generate_accessory_attributes():
        colors = ["Black", "Brown", "White", "Red", "Blue", "Gold", "Silver", "Multi-color"]
        materials = ["Leather", "Canvas", "Nylon", "Metal", "Fabric", "Stainless Steel", "Gold Plated", "Silver Plated"]
        return {
            "color": random.choice(colors),
            "material": random.choice(materials),
            "style": random.choice(["Casual", "Formal", "Luxury", "Vintage", "Modern", "Classic", "Bohemian"]),
            "dimensions": f"{random.randint(15, 40)}cm x {random.randint(10, 30)}cm x {random.randint(5, 15)}cm",
            "weight": f"{random.randint(100, 1500)}g",
            "gender": random.choice(["Men", "Women", "Unisex"])
        }
    
    def generate_watch_attributes():
        colors = ["Black", "Silver", "Gold", "Rose Gold", "Blue", "Brown"]
        materials = ["Stainless Steel", "Leather", "Rubber", "Ceramic", "Titanium", "Gold", "Silicone"]
        return {
            "case_color": random.choice(colors),
            "band_color": random.choice(colors),
            "case_material": random.choice(materials[:5]),
            "band_material": random.choice(materials),
            "case_diameter": f"{random.randint(30, 45)}mm",
            "water_resistance": f"{random.choice(['30', '50', '100', '200', '300'])}m",
            "movement": random.choice(["Quartz", "Automatic", "Mechanical", "Solar", "Hybrid"]),
            "features": random.sample(["Date Display", "Chronograph", "Luminous Hands", "Tachymeter", "Dual Time", "Alarm", "Compass"], random.randint(1, 4)),
            "style": random.choice(["Dress", "Casual", "Sport", "Luxury", "Diving", "Pilot", "Field"]),
            "gender": random.choice(["Men", "Women", "Unisex"])
        }
    
    def generate_cosmetic_attributes():
        colors = ["Natural", "Rose", "Peach", "Red", "Pink", "Nude", "Berry", "Coral", "Brown", "Black"]
        finishes = ["Matte", "Shimmer", "Satin", "Glossy", "Metallic", "Dewy", "Radiant"]
        return {
            "color": random.choice(colors) if "Lipstick" in product_type or "Eyeshadow" in product_type else None,
            "finish": random.choice(finishes) if "Lipstick" in product_type or "Foundation" in product_type or "Eyeshadow" in product_type else None,
            "skin_type": random.choice(["All Skin Types", "Dry", "Oily", "Combination", "Sensitive"]),
            "ingredients": "Water, Glycerin, Dimethicone, Vitamin E, Hyaluronic Acid, and other quality ingredients",
            "volume_weight": f"{random.randint(5, 200)}{'ml' if 'Liquid' in product_type or 'Set' in product_type else 'g'}",
            "application": random.choice(["Easy application with fingertips", "Use with beauty blender", "Apply with brush", "Gently tap onto skin"]),
            "benefits": random.sample(["Hydrating", "Long-lasting", "Cruelty-free", "Paraben-free", "Vegan", "Dermatologically tested", "Non-comedogenic"], random.randint(2, 5))
        }
    
    def generate_eyewear_attributes():
        colors = ["Black", "Tortoise", "Clear", "Brown", "Blue", "Red", "Gold", "Silver"]
        materials = ["Acetate", "Metal", "TR-90", "Titanium", "Stainless Steel", "Plastic"]
        return {
            "frame_color": random.choice(colors),
            "lens_color": random.choice(["Clear", "Brown", "Grey", "Blue", "Green", "Yellow"]),
            "frame_material": random.choice(materials),
            "frame_shape": random.choice(["Round", "Square", "Rectangle", "Cat Eye", "Aviator", "Wayfarer", "Oval"]),
            "lens_type": random.choice(["Single Vision", "Bifocal", "Progressive", "Blue Light Blocking", "Photochromic", "Polarized"]),
            "frame_width": f"{random.randint(125, 145)}mm",
            "temple_length": f"{random.randint(135, 155)}mm",
            "gender": random.choice(["Men", "Women", "Unisex"])
        }
    
    def generate_electronics_attributes():
        colors = ["Black", "White", "Red", "Blue", "Grey"]
        return {
            "color": random.choice(colors),
            "connectivity": random.choice(["Wired", "Wireless", "Bluetooth", "USB", "USB-C"]),
            "compatibility": random.choice(["PC", "PS5", "Xbox Series X/S", "Nintendo Switch", "All Platforms", "Multi-platform"]),
            "battery_life": f"{random.randint(8, 30)} hours" if "Wireless" in product_type or "Bluetooth" in product_type else None,
            "dimensions": f"{random.randint(10, 30)}cm x {random.randint(5, 20)}cm x {random.randint(2, 10)}cm",
            "weight": f"{random.randint(200, 800)}g",
            "warranty": f"{random.choice(['1', '2', '3'])} years"
        }
    
    def generate_generic_attributes():
        return {
            "description": f"Premium quality {product_type.lower()} from {brand.brand_name_en}",
            "model_number": f"{brand.brand_name_en[:3].upper()}-{random.randint(1000, 9999)}",
            "item_code": f"{random.choice(['A', 'B', 'C', 'D', 'E'])}{random.randint(10000, 99999)}",
            "origin": random.choice(["Italy", "France", "USA", "Germany", "Japan", "South Korea", "UK", "China", "Vietnam"]),
            "warranty": f"{random.choice(['6', '12', '24', '36'])} months"
        }
    
    # Default product types for unknown categories
    default_products = ["Premium Item", "Signature Collection", "Limited Edition", "Best Seller", "New Arrival"]
    
    # Get product types based on brand category
    category = brand.category_name if brand.category_name else "Unknown"
    product_types = category_mapping.get(category, default_products)
    
    # If we didn't find a direct match, try partial matching
    if category not in category_mapping:
        for key in category_mapping:
            if key.lower() in category.lower():
                product_types = category_mapping[key]
                break
    
    # Generate 3 products
    for i in range(3):
        # Select a product type, wrap around if needed
        product_type = product_types[i % len(product_types)]
        
        # Generate a descriptive name based on brand and product type
        name = f"{brand.brand_name_en} {product_type}"
        
        # Generate a description
        description = f"Premium quality {product_type.lower()} from {brand.brand_name_en}, featuring the latest design and excellent craftsmanship."
        
        # Generate a random price between 99 and 2999
        price = round(random.uniform(99, 2999), 2)
        
        # Use brand's logo as image if available, otherwise use a placeholder
        image_url = ""
        if hasattr(brand, 'brand_logo') and brand.brand_logo:
            image_url = brand.brand_logo
        else:
            image_url = "https://via.placeholder.com/300"
        
        # Generate attributes based on product type and category
        attributes = generate_generic_attributes()  # Base attributes for all products
        
        # Add specific attributes based on category/product type
        if "Shoes" in product_type or "Sandals" in product_type or "Boots" in product_type or "Flats" in product_type or "Footwear" in product_type:
            attributes.update(generate_footwear_attributes())
        elif "T-Shirt" in product_type or "Jacket" in product_type or "Shorts" in product_type or "Pajama" in product_type or "Robe" in product_type or "Thobe" in product_type:
            attributes.update(generate_clothing_attributes())
        elif "Handbag" in product_type or "Wallet" in product_type or "Belt" in product_type or "Necklace" in product_type:
            attributes.update(generate_accessory_attributes())
        elif "Watch" in product_type or "Timepiece" in product_type or "Chronograph" in product_type:
            attributes.update(generate_watch_attributes())
        elif "Foundation" in product_type or "Eyeshadow" in product_type or "Lipstick" in product_type or "Mascara" in product_type or "Skincare" in product_type:
            attributes.update(generate_cosmetic_attributes())
        elif "Eyeglasses" in product_type or "Sunglasses" in product_type or "Glasses" in product_type:
            attributes.update(generate_eyewear_attributes())
        elif "Game" in product_type or "Headset" in product_type or "Controller" in product_type or "Keyboard" in product_type:
            attributes.update(generate_electronics_attributes())
        
        # Create the product
        product = {
            "name": name,
            "description": description,
            "price": price,
            "image_url": image_url,
            "category": category,
            "brand_id": brand.brand_id,
            "is_featured": random.choice([True, False]),
            "in_stock": True,
            "attributes": attributes
        }
        
        products.append(product)
    
    return products


def generate_mall_services():
    """Generate common mall services"""
    services = [
        {
            "name": "Free Wi-Fi",
            "name_ar": "واي فاي مجاني",
            "description": "High-speed internet available throughout the mall",
            "description_ar": "إنترنت عالي السرعة متوفر في جميع أنحاء المول",
            "icon_url": "https://example.com/icons/wifi.png",
            "is_available": True,
            "location": "Throughout the mall"
        },
        {
            "name": "Valet Parking",
            "name_ar": "خدمة صف السيارات",
            "description": "Convenient valet parking service available at main entrances",
            "description_ar": "خدمة صف السيارات المريحة متوفرة عند المداخل الرئيسية",
            "icon_url": "https://example.com/icons/valet.png",
            "is_available": True,
            "location": "Main entrances"
        },
        {
            "name": "Prayer Rooms",
            "name_ar": "غرف الصلاة",
            "description": "Dedicated prayer rooms for men and women",
            "description_ar": "غرف صلاة مخصصة للرجال والنساء",
            "icon_url": "https://example.com/icons/prayer.png",
            "is_available": True,
            "location": "Level 1 and Level 2"
        },
        {
            "name": "Customer Service",
            "name_ar": "خدمة العملاء",
            "description": "Assistance with mall information, gift cards, and lost & found",
            "description_ar": "المساعدة في معلومات المول وبطاقات الهدايا والمفقودات",
            "icon_url": "https://example.com/icons/customer-service.png",
            "is_available": True,
            "location": "Main concourse, Level 1"
        },
        {
            "name": "ATM",
            "name_ar": "صراف آلي",
            "description": "Automated Teller Machines available for cash withdrawals",
            "description_ar": "أجهزة الصراف الآلي متوفرة للسحب النقدي",
            "icon_url": "https://example.com/icons/atm.png",
            "is_available": True,
            "location": "Various locations throughout the mall"
        },
        {
            "name": "Children's Play Area",
            "name_ar": "منطقة لعب الأطفال",
            "description": "Supervised play area for children",
            "description_ar": "منطقة لعب للأطفال تحت الإشراف",
            "icon_url": "https://example.com/icons/play-area.png",
            "is_available": True,
            "location": "Level 2, near food court"
        },
        {
            "name": "Wheelchair Access",
            "name_ar": "وصول الكراسي المتحركة",
            "description": "Wheelchair ramps and elevators for accessibility",
            "description_ar": "منحدرات للكراسي المتحركة ومصاعد لسهولة الوصول",
            "icon_url": "https://example.com/icons/wheelchair.png",
            "is_available": True,
            "location": "All entrances and levels"
        },
        {
            "name": "Food Court",
            "name_ar": "ساحة الطعام",
            "description": "Various dining options in a central location",
            "description_ar": "خيارات طعام متنوعة في موقع مركزي",
            "icon_url": "https://example.com/icons/food-court.png",
            "is_available": True,
            "location": "Level 3"
        },
        {
            "name": "Gift Wrapping",
            "name_ar": "تغليف الهدايا",
            "description": "Professional gift wrapping service",
            "description_ar": "خدمة احترافية لتغليف الهدايا",
            "icon_url": "https://example.com/icons/gift-wrap.png",
            "is_available": True,
            "location": "Level 1, next to Customer Service"
        },
        {
            "name": "Car Wash",
            "name_ar": "غسيل السيارات",
            "description": "Vehicle cleaning service while you shop",
            "description_ar": "خدمة تنظيف المركبات أثناء التسوق",
            "icon_url": "https://example.com/icons/car-wash.png",
            "is_available": True,
            "location": "Parking Level P1"
        }
    ]
    
    return services


async def seed_database():
    reset_database()
    db = SessionLocal()
    try:
        # Insert malls from both al_nakheel and u_walk
        mall_objects = {}
        
        # Insert malls from al_nakheel
        for mall_data in malls_data:
            mall = Mall(**mall_data)
            db.add(mall)
            mall_objects[mall.unique_property_id] = mall
        
        # Insert malls from u_walk
        for mall_data in u_walk_mall_data:
            mall = Mall(**mall_data)
            db.add(mall)
            mall_objects[mall.unique_property_id] = mall
            
        db.commit()
        print(f"Malls inserted successfully. Added {len(mall_objects)} malls.")

        # Insert brands and associate with their respective malls
        brand_objects = {}
        
        # Insert brands from al_nakheel
        for brand_data in brands_data[0]:  
            # Create a copy of brand_data and set default values if needed
            brand_dict = dict(brand_data)
            if 'is_published' not in brand_dict:
                brand_dict['is_published'] = True
                
            brand = Brand(**brand_dict)
            db.add(brand)
            brand_objects[brand.brand_id] = brand
            
            # Associate brand with al_nakheel mall
            # Find the al_nakheel mall ID (assuming it's the first mall in the data)
            al_nakheel_id = malls_data[0]["unique_property_id"]
            mall_objects[al_nakheel_id].brands.append(brand)
        
        # Insert brands from u_walk
        for brand_data in u_walk_brands_data:
            # Check if brand already exists (could be in both malls)
            if brand_data["brand_id"] in brand_objects:
                # Brand already exists, just associate it with the u_walk mall
                brand = brand_objects[brand_data["brand_id"]]
            else:
                # Create a new brand
                brand_dict = dict(brand_data)
                if 'is_published' not in brand_dict:
                    brand_dict['is_published'] = True
                    
                brand = Brand(**brand_dict)
                db.add(brand)
                brand_objects[brand.brand_id] = brand
            
            # Associate brand with u_walk mall
            u_walk_id = u_walk_mall_data[0]["unique_property_id"]
            mall_objects[u_walk_id].brands.append(brand)
        
        db.commit()
        print(f"Brands inserted and associated with malls successfully. Added {len(brand_objects)} brands.")

        # Find all missing brand IDs from all engagement_data sources
        missing_brand_ids = []
        
        # Check al_nakheel engagement data
        for engagement_item in engagement_data:
            brand_id = engagement_item.get('brand_id')
            if brand_id and brand_id not in brand_objects and brand_id not in missing_brand_ids:
                missing_brand_ids.append(brand_id)
        
        # Check u_walk engagement data
        for engagement_item in u_walk_engagement_data:
            brand_id = engagement_item.get('brand_id')
            if brand_id and brand_id not in brand_objects and brand_id not in missing_brand_ids:
                missing_brand_ids.append(brand_id)
        
        # Create any missing brands needed for the engagements
        if missing_brand_ids:
            print(f"Creating {len(missing_brand_ids)} missing brands needed for engagements...")
            
            for brand_id in missing_brand_ids:
                # Create a basic brand with the required ID
                brand = Brand(
                    brand_id=brand_id,
                    brand_name_en=f"Brand {brand_id}",
                    brand_name_ar=f"العلامة التجارية {brand_id}",
                    tenant_profile_id=brand_id,
                    brand_logo="https://example.com/placeholder-logo.png",
                    category_name="Uncategorized",
                    category_name_ar="غير مصنف",
                    is_published=True,
                    pms_unit_codes=[]
                )
                db.add(brand)
                brand_objects[brand_id] = brand
            
            db.commit()
            print(f"Created {len(missing_brand_ids)} missing brands.")

        # Insert engagements and associate with their respective malls and brands
        successful_engagements = 0
        
        # Insert engagements from al_nakheel
        al_nakheel_id = malls_data[0]["unique_property_id"]
        for engagement_data_item in engagement_data:
            # Create a copy of engagement_data_item and set default values if needed
            engagement_dict = dict(engagement_data_item)
            if 'home_banner_disp' not in engagement_dict:
                engagement_dict['home_banner_disp'] = 0
                
            engagement = Engagement(**engagement_dict, unique_property_id=al_nakheel_id)
            db.add(engagement)
            successful_engagements += 1
        
        # Insert engagements from u_walk
        u_walk_id = u_walk_mall_data[0]["unique_property_id"]
        
        # Track engagement IDs that have already been added for u_walk mall to avoid duplicates
        processed_engagement_ids = set()
        
        for engagement_data_item in u_walk_engagement_data:
            # Skip if this engagement ID has already been processed for this mall
            engagement_id = engagement_data_item["engagement_id"]
            if engagement_id in processed_engagement_ids:
                print(f"Skipping duplicate engagement: ID {engagement_id}")
                continue
                
            # Mark this engagement ID as processed
            processed_engagement_ids.add(engagement_id)
            
            # Create a copy of engagement_data_item and set default values if needed
            engagement_dict = dict(engagement_data_item)
            if 'home_banner_disp' not in engagement_dict:
                engagement_dict['home_banner_disp'] = 0
                
            engagement = Engagement(**engagement_dict, unique_property_id=u_walk_id)
            db.add(engagement)
            successful_engagements += 1
        
        db.commit()
        print(f"Engagements inserted successfully. Added {successful_engagements} engagements.")

        # Generate and insert products for each brand
        print("Generating and inserting products for brands...")
        total_products = 0
        
        for brand_id, brand in brand_objects.items():
            # Generate relevant products for this brand
            brand_products = generate_products_for_brand(brand)
            
            # Insert products
            for product_data in brand_products:
                product = Product(**product_data)
                db.add(product)
                total_products += 1
        
        db.commit()
        print(f"Products generated and inserted successfully. Added {total_products} products.")

        # Generate and insert services for each mall
        print("Generating and inserting services for malls...")
        services_data = generate_mall_services()
        total_services = 0
        
        for mall_id, mall in mall_objects.items():
            # Add services to this mall
            for service_data in services_data:
                service = Service(**service_data, unique_property_id=mall_id)
                db.add(service)
                total_services += 1
        
        db.commit()
        print(f"Services generated and inserted successfully. Added {total_services} services.")

    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {str(e)}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
    finally:
        db.close()


async def main():
    print("Starting database seeding process...")
    await seed_database()
    print("Database seeding process completed.")


if __name__ == "__main__":
    asyncio.run(main()) 