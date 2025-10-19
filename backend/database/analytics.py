"""
Analytics System - Part 1: Database Connection Boilerplate

This creates the foundation for our analytics system.
We'll add tables incrementally to verify each piece works.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Configuration ==========

# Database location
DB_DIR = Path(__file__).parent.parent / 'database'
DB_FILE = DB_DIR / 'analytics.db'


# ========== Database Connection ==========

class AnalyticsDB:
    """
    SQLite database connection manager.
    Handles connection lifecycle and basic operations.
    """
    
    def __init__(self, db_path: Path = DB_FILE):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analytics DB initialized: {self.db_path}")
    
    
    def connect(self) -> sqlite3.Connection:
        """
        Get database connection.
        
        Returns:
            SQLite connection object
        """
        if self.conn is None:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Allow multi-threaded access
            )
            # Return rows as dictionaries
            self.conn.row_factory = sqlite3.Row
            logger.info("✅ Database connected")
        
        return self.conn
    
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters (for safety)
            
        Returns:
            Cursor object
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor
    
    
    def executemany(self, query: str, params_list: list) -> sqlite3.Cursor:
        """
        Execute a SQL query multiple times.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Cursor object
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        conn.commit()
        return cursor
    
    
    def fetchone(self, query: str, params: tuple = ()) -> Optional[dict]:
        """
        Fetch one row.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Dictionary with row data or None
        """
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    
    def fetchall(self, query: str, params: tuple = ()) -> list:
        """
        Fetch all rows.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries
        """
        cursor = self.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists
        """
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """
        result = self.fetchone(query, (table_name,))
        return result is not None


# ========== Schema Creation ==========

def create_search_events_table(db: AnalyticsDB):
    """
    Create table to track search events.
    
    Tracks every search query: text, image, or hybrid.
    """
    query = """
    CREATE TABLE IF NOT EXISTS search_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query_text TEXT,
        query_type TEXT NOT NULL,
        filters_applied TEXT,
        results_count INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    db.execute(query)
    logger.info("✅ Table created: search_events")


def create_product_views_table(db: AnalyticsDB):
    """
    Create table to track product views.
    
    Tracks when users click/view products.
    """
    query = """
    CREATE TABLE IF NOT EXISTS product_views (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        source TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    db.execute(query)
    logger.info("✅ Table created: product_views")


def create_chat_messages_table(db: AnalyticsDB):
    """
    Create table to track chatbot interactions.
    
    Tracks every message in the chatbot.
    """
    query = """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        message TEXT NOT NULL,
        intent TEXT,
        products_shown TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    db.execute(query)
    logger.info("✅ Table created: chat_messages")


def create_comparisons_table(db: AnalyticsDB):
    """
    Create table to track product comparisons.
    
    Tracks when users compare products.
    """
    query = """
    CREATE TABLE IF NOT EXISTS comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        product_ids TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    db.execute(query)
    logger.info("✅ Table created: comparisons")


# ========== Event Tracking Functions ==========

def track_search(
    db: AnalyticsDB,
    session_id: str,
    query_text: str = None,
    query_type: str = 'text',
    filters_applied: str = None,
    results_count: int = 0
):
    """
    Track a search event.
    
    Args:
        db: Database connection
        session_id: User session identifier
        query_text: Search query (if text search)
        query_type: 'text', 'image', or 'hybrid'
        filters_applied: JSON string of filters
        results_count: Number of results returned
    """
    query = """
    INSERT INTO search_events 
    (session_id, query_text, query_type, filters_applied, results_count)
    VALUES (?, ?, ?, ?, ?)
    """
    
    db.execute(query, (
        session_id,
        query_text,
        query_type,
        filters_applied,
        results_count
    ))
    
    logger.info(f"📊 Search tracked: {query_type} query, {results_count} results")


def track_product_view(
    db: AnalyticsDB,
    session_id: str,
    product_id: str,
    source: str = 'search'
):
    """
    Track a product view event.
    
    Args:
        db: Database connection
        session_id: User session identifier
        product_id: Product unique ID
        source: Where view came from ('search', 'similar', 'recommendation', 'chat')
    """
    query = """
    INSERT INTO product_views 
    (session_id, product_id, source)
    VALUES (?, ?, ?)
    """
    
    db.execute(query, (session_id, product_id, source))
    
    logger.info(f"👁️  Product view tracked: {product_id} from {source}")


def track_chat_message(
    db: AnalyticsDB,
    session_id: str,
    message: str,
    intent: str = None,
    products_shown: str = None
):
    """
    Track a chatbot message.
    
    Args:
        db: Database connection
        session_id: User session identifier
        message: User's message
        intent: Detected intent ('search', 'compare', 'ask')
        products_shown: JSON string of product IDs shown
    """
    query = """
    INSERT INTO chat_messages 
    (session_id, message, intent, products_shown)
    VALUES (?, ?, ?, ?)
    """
    
    db.execute(query, (session_id, message, intent, products_shown))
    
    logger.info(f"💬 Chat message tracked: {intent or 'general'}")


def track_comparison(
    db: AnalyticsDB,
    session_id: str,
    product_ids: str
):
    """
    Track a product comparison event.
    
    Args:
        db: Database connection
        session_id: User session identifier
        product_ids: JSON string of product IDs being compared
    """
    query = """
    INSERT INTO comparisons 
    (session_id, product_ids)
    VALUES (?, ?)
    """
    
    db.execute(query, (session_id, product_ids))
    
    logger.info(f"🔍 Comparison tracked")


# ========== Analytics Queries ==========

def get_recent_searches(db: AnalyticsDB, limit: int = 10) -> list:
    """
    Get most recent searches.
    
    Args:
        db: Database connection
        limit: Number of searches to return
        
    Returns:
        List of search events
    """
    query = """
    SELECT * FROM search_events 
    ORDER BY timestamp DESC 
    LIMIT ?
    """
    return db.fetchall(query, (limit,))


def get_top_search_queries(db: AnalyticsDB, limit: int = 10) -> list:
    """
    Get most popular search queries.
    
    Args:
        db: Database connection
        limit: Number of queries to return
        
    Returns:
        List of queries with counts
    """
    query = """
    SELECT query_text, COUNT(*) as count
    FROM search_events
    WHERE query_text IS NOT NULL
    GROUP BY query_text
    ORDER BY count DESC
    LIMIT ?
    """
    return db.fetchall(query, (limit,))


def get_search_type_breakdown(db: AnalyticsDB) -> list:
    """
    Get breakdown of search types.
    
    Returns:
        List of search types with counts
    """
    query = """
    SELECT query_type, COUNT(*) as count
    FROM search_events
    GROUP BY query_type
    ORDER BY count DESC
    """
    return db.fetchall(query)


def get_most_viewed_products(db: AnalyticsDB, limit: int = 10) -> list:
    """
    Get most viewed products.
    
    Args:
        db: Database connection
        limit: Number of products to return
        
    Returns:
        List of products with view counts
    """
    query = """
    SELECT product_id, COUNT(*) as views, source
    FROM product_views
    GROUP BY product_id
    ORDER BY views DESC
    LIMIT ?
    """
    return db.fetchall(query, (limit,))


def get_view_sources_breakdown(db: AnalyticsDB) -> list:
    """
    Get breakdown of where views came from.
    
    Returns:
        List of sources with counts
    """
    query = """
    SELECT source, COUNT(*) as count
    FROM product_views
    GROUP BY source
    ORDER BY count DESC
    """
    return db.fetchall(query)


def get_total_views(db: AnalyticsDB) -> int:
    """
    Get total product views.
    
    Returns:
        Total view count
    """
    query = "SELECT COUNT(*) as total FROM product_views"
    result = db.fetchone(query)
    return result['total'] if result else 0


def get_chat_statistics(db: AnalyticsDB) -> dict:
    """
    Get chatbot usage statistics.
    
    Returns:
        Dictionary with chat stats
    """
    total_query = "SELECT COUNT(*) as total FROM chat_messages"
    intent_query = """
        SELECT intent, COUNT(*) as count 
        FROM chat_messages 
        WHERE intent IS NOT NULL
        GROUP BY intent
    """
    
    total = db.fetchone(total_query)['total']
    intents = db.fetchall(intent_query)
    
    return {
        'total_messages': total,
        'intents': intents
    }


def get_top_compared_products(db: AnalyticsDB, limit: int = 10) -> list:
    """
    Get most compared products.
    
    Args:
        db: Database connection
        limit: Number to return
        
    Returns:
        List of product comparison counts
    """
    query = """
    SELECT product_ids, COUNT(*) as count
    FROM comparisons
    GROUP BY product_ids
    ORDER BY count DESC
    LIMIT ?
    """
    return db.fetchall(query, (limit,))


def get_total_comparisons(db: AnalyticsDB) -> int:
    """
    Get total number of comparisons.
    
    Returns:
        Total comparison count
    """
    query = "SELECT COUNT(*) as total FROM comparisons"
    result = db.fetchone(query)
    return result['total'] if result else 0


# ========== Initialization ==========

def create_database_connection() -> AnalyticsDB:
    """
    Factory function to create database connection.
    
    Returns:
        AnalyticsDB instance
    """
    return AnalyticsDB()


def initialize_analytics_database():
    """
    Initialize database with all tables.
    Call this once to set up the database.
    """
    db = create_database_connection()
    
    logger.info("🔨 Initializing analytics database...")
    
    # Create all tables
    create_search_events_table(db)
    create_product_views_table(db)
    create_chat_messages_table(db)
    create_comparisons_table(db)
    
    logger.info("✅ Database initialization complete")
    
    return db


# ========== Testing ==========

if __name__ == "__main__":
    """
    Complete analytics system test.
    """
    print("\n" + "="*70)
    print("🧪 COMPLETE ANALYTICS SYSTEM TEST")
    print("="*70)
    
    try:
        # Initialize complete database
        db = initialize_analytics_database()
        
        # Test 1: Verify all tables
        print("\n✅ Test 1: All Tables Created")
        tables = ['search_events', 'product_views', 'chat_messages', 'comparisons']
        for table in tables:
            if db.table_exists(table):
                print(f"   ✅ {table}")
        
        # Test 2: Track chat messages
        print("\n✅ Test 2: Track Chat Messages")
        track_chat_message(db, "session_123", "show me modern sofas", intent="search")
        track_chat_message(db, "session_123", "which is better?", intent="compare")
        track_chat_message(db, "session_456", "I need help choosing", intent="ask")
        print("   ✅ Tracked 3 chat messages")
        
        # Test 3: Chat statistics
        print("\n✅ Test 3: Chat Statistics")
        chat_stats = get_chat_statistics(db)
        print(f"   Total messages: {chat_stats['total_messages']}")
        print(f"   Intents breakdown:")
        for intent in chat_stats['intents']:
            print(f"   - {intent['intent']}: {intent['count']}")
        
        # Test 4: Track comparisons
        print("\n✅ Test 4: Track Comparisons")
        import json
        track_comparison(db, "session_123", json.dumps(["prod_001", "prod_002"]))
        track_comparison(db, "session_456", json.dumps(["prod_003", "prod_004", "prod_005"]))
        print("   ✅ Tracked 2 comparisons")
        
        # Test 5: Comparison statistics
        print("\n✅ Test 5: Comparison Statistics")
        total_comparisons = get_total_comparisons(db)
        print(f"   Total comparisons: {total_comparisons}")
        
        # Summary: Full analytics overview
        print("\n" + "="*70)
        print("📊 ANALYTICS OVERVIEW")
        print("="*70)
        
        total_searches = db.fetchone("SELECT COUNT(*) as count FROM search_events")['count']
        total_views = get_total_views(db)
        total_messages = chat_stats['total_messages']
        total_comps = get_total_comparisons(db)
        
        print(f"\n   Searches:    {total_searches}")
        print(f"   Views:       {total_views}")
        print(f"   Chat msgs:   {total_messages}")
        print(f"   Comparisons: {total_comps}")
        
        # Close connection
        db.close()
        
        print("\n" + "="*70)
        print("✅ COMPLETE ANALYTICS SYSTEM READY!")
        print("="*70)
        print("\n🎉 All 4 tables working!")
        print("📋 Ready to integrate with FastAPI!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise