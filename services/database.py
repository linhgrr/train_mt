import mysql.connector
from mysql.connector import Error
import logging
from typing import Dict, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.connection = None
        self.entity_cache = {}
        self._connect()
        
    def _connect(self):
        """Establish connection to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                charset=settings.DB_CHARSET,
                use_unicode=True,
                collation='utf8mb4_unicode_ci'
            )
            if self.connection.is_connected():
                logger.info(f"Connected to MySQL database {settings.DB_NAME}")
                self._load_entity_mapping()
        except Error as e:
            logger.error(f"Error connecting to MySQL database: {e}")
            self.connection = None
    def _load_entity_mapping(self):
        """Load entity mapping from database into cache"""
        if not self.connection or not self.connection.is_connected():
            return
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT kanji, english FROM train_entity"
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                if row['kanji'] and row['english']:
                    self.entity_cache[row['kanji']] = row['english']
                    
            logger.info(f"Loaded {len(self.entity_cache)} entity mappings from database")
            cursor.close()
            
        except Error as e:
            logger.error(f"Error loading entity mapping: {e}")
    
    def get_entity_translation(self, japanese_entity: str, use_cache: bool = True) -> Optional[str]:
        """
        Get English translation for Japanese entity
        
        Args:
            japanese_entity: The Japanese entity to translate
            use_cache: If True, use cached data. If False, query database directly
        """
        if use_cache:
            return self.entity_cache.get(japanese_entity)
        else:
            return self._query_entity_from_db(japanese_entity)
    
    def _query_entity_from_db(self, japanese_entity: str) -> Optional[str]:
        """Query entity translation directly from database"""
        if not self.connection or not self.connection.is_connected():
            logger.warning("Database not connected, cannot query entity")
            return None
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT english FROM train_entity WHERE kanji = %s LIMIT 1"
            cursor.execute(query, (japanese_entity,))
            result = cursor.fetchone()
            cursor.close()
            
            if result and result['english']:
                logger.debug(f"Found entity translation: {japanese_entity} -> {result['english']}")
                return result['english']
            else:
                logger.debug(f"No translation found for entity: {japanese_entity}")
                return None
                
        except Error as e:
            logger.error(f"Error querying entity translation for '{japanese_entity}': {e}")
            return None
    
    def search_entities(self, search_term: str, limit: int = 10) -> Dict[str, str]:
        """
        Search for entities containing the search term
        
        Args:
            search_term: Term to search for in kanji or english
            limit: Maximum number of results to return
        """
        if not self.connection or not self.connection.is_connected():
            logger.warning("Database not connected, cannot search entities")
            return {}
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT kanji, english FROM train_entity 
                WHERE kanji LIKE %s OR english LIKE %s 
                LIMIT %s
            """
            search_pattern = f"%{search_term}%"
            cursor.execute(query, (search_pattern, search_pattern, limit))
            results = cursor.fetchall()
            cursor.close()
            
            entity_dict = {}
            for row in results:
                if row['kanji'] and row['english']:
                    entity_dict[row['kanji']] = row['english']
                    
            logger.debug(f"Found {len(entity_dict)} entities matching '{search_term}'")
            return entity_dict
            
        except Error as e:
            logger.error(f"Error searching entities for '{search_term}': {e}")
            return {}
    
    def add_entity(self, kanji: str, english: str) -> bool:
        """
        Add new entity translation to database
        
        Args:
            kanji: Japanese entity name
            english: English translation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection or not self.connection.is_connected():
            logger.warning("Database not connected, cannot add entity")
            return False
            
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO train_entity (kanji, english) 
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE english = VALUES(english)
            """
            cursor.execute(query, (kanji, english))
            self.connection.commit()
            cursor.close()
            
            # Update cache if using cache
            self.entity_cache[kanji] = english
            
            logger.info(f"Added/updated entity: {kanji} -> {english}")
            return True
            
        except Error as e:
            logger.error(f"Error adding entity '{kanji}' -> '{english}': {e}")
            return False
    
    def refresh_cache(self):
        """Refresh the entity cache from database"""
        self.entity_cache.clear()
        self._load_entity_mapping()
    
    def is_connected(self) -> bool:
        """Check if database connection is active"""
        return self.connection is not None and self.connection.is_connected()
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")

# Global database service instance
db_service = DatabaseService()
