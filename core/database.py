"""
SecureX-Assist - Database Layer
Secure storage for users, voice embeddings, and sessions
"""

import sqlite3
import json
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager for SecureX-Assist
    Stores: Users, Voice Embeddings, Sessions, Audit Logs
    """
    
    def __init__(self, db_path: str = "securex_db.sqlite"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
        logger.info(f"Initializing database: {db_path}")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Access columns by name
            logger.info("Database connected successfully")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def initialize_schema(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    account_type TEXT DEFAULT 'standard_user',
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    failed_attempts INTEGER DEFAULT 0
                )
            """)
            
            # Voice embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    embedding_data TEXT NOT NULL,
                    embedding_variance TEXT,
                    embedding_type TEXT DEFAULT 'pyannote',
                    quality_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Face embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    embedding_data TEXT NOT NULL,
                    embedding_type TEXT DEFAULT 'face_recognition',
                    quality_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    voice_verified BOOLEAN DEFAULT 0,
                    password_verified BOOLEAN DEFAULT 0,
                    liveness_verified BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Audit logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    status TEXT DEFAULT 'success',
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)
            
            # Liveness challenges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS liveness_challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    challenge_phrase TEXT NOT NULL,
                    transcribed_text TEXT,
                    similarity_score REAL DEFAULT 0.0,
                    passed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # API Keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT UNIQUE NOT NULL,
                    client_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Migration: Add embedding_variance column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE voice_embeddings ADD COLUMN embedding_variance TEXT")
                logger.info("Added embedding_variance column to voice_embeddings table")
            except sqlite3.OperationalError:
                # Column already exists
                pass

            # Migration: Add account_type to users if it doesn't exist
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN account_type TEXT DEFAULT 'standard_user'")
                logger.info("Added account_type column to users table")
            except sqlite3.OperationalError:
                # Column already exists
                pass

            # Backfill account_type for older rows created before migration
            cursor.execute(
                "UPDATE users SET account_type = 'standard_user' WHERE account_type IS NULL OR TRIM(account_type) = ''"
            )
            
            self.conn.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    # ==================== USER OPERATIONS ====================
    
    def create_user(
        self, 
        username: str, 
        password_hash: str, 
        account_type: str = "standard_user",
        email: Optional[str] = None
    ) -> Optional[int]:
        """
        Create a new user
        
        Returns:
            User ID if successful, None otherwise
        """
        try:
            cursor = self.conn.cursor()

            allowed_types = {"administrator", "standard_user", "power_user", "guest"}
            normalized_type = (account_type or "standard_user").strip().lower()
            if normalized_type not in allowed_types:
                normalized_type = "standard_user"
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, account_type, email) VALUES (?, ?, ?, ?)",
                (username, password_hash, normalized_type, email)
            )
            
            self.conn.commit()
            user_id = cursor.lastrowid
            
            logger.info(f"User created: {username} (ID: {user_id}, type: {normalized_type})")
            self.log_audit(user_id, "user_created", f"Username: {username}, Type: {normalized_type}")
            
            return user_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"User already exists: {username}")
            return None
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all active users"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users WHERE is_active = 1 ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get all users: {e}")
            return []
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
    
    def increment_failed_attempts(self, username: str):
        """Increment failed login attempts"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1 WHERE username = ?",
                (username,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to increment failed attempts: {e}")
    
    def reset_failed_attempts(self, username: str):
        """Reset failed login attempts after successful login"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE users SET failed_attempts = 0 WHERE username = ?",
                (username,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to reset failed attempts: {e}")
    
    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user and all associated data
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Get username for audit log before deletion
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            username = row['username'] if row else "unknown"
            
            # Delete user (CASCADE will handle related tables)
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            self.conn.commit()
            
            logger.info(f"User deleted: {username} (ID: {user_id})")
            self.log_audit(None, "user_deleted", f"Username: {username}, ID: {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    # ==================== VOICE EMBEDDING OPERATIONS ====================
    
    def store_voice_embedding(
        self, 
        user_id: int, 
        embedding: np.ndarray,
        variance: Optional[np.ndarray] = None,
        embedding_type: str = "pyannote",
        quality_score: float = 1.0
    ) -> Optional[int]:
        """
        Store voice embedding for a user (mean embedding + variance)
        
        Args:
            user_id: User ID
            embedding: Voice embedding mean (numpy array)
            variance: Voice embedding variance (numpy array, optional)
            embedding_type: Type of model used
            quality_score: Quality assessment score
            
        Returns:
            Embedding ID if successful
        """
        try:
            # Serialize embedding to JSON
            embedding_json = json.dumps(embedding.tolist())
            variance_json = json.dumps(variance.tolist()) if variance is not None else None
            
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO voice_embeddings 
                (user_id, embedding_data, embedding_variance, embedding_type, quality_score) 
                VALUES (?, ?, ?, ?, ?)""",
                (user_id, embedding_json, variance_json, embedding_type, quality_score)
            )
            
            self.conn.commit()
            embedding_id = cursor.lastrowid
            
            logger.info(f"Voice embedding stored for user ID {user_id}")
            self.log_audit(user_id, "voice_enrolled", f"Embedding ID: {embedding_id}")
            
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to store voice embedding: {e}")
            return None
    
    def get_voice_embeddings(self, user_id: int) -> List[Dict]:
        """
        Get all active voice embeddings for a user
        
        Returns:
            List of embedding dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """SELECT * FROM voice_embeddings 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY created_at DESC""",
                (user_id,)
            )
            
            rows = cursor.fetchall()
            embeddings = []
            
            for row in rows:
                embedding_dict = dict(row)
                # Deserialize embedding data
                embedding_dict['embedding_array'] = np.array(
                    json.loads(embedding_dict['embedding_data'])
                )
                # Deserialize variance data if present
                if embedding_dict['embedding_variance'] is not None and embedding_dict['embedding_variance'] != '':
                    try:
                        embedding_dict['embedding_variance'] = np.array(
                            json.loads(embedding_dict['embedding_variance'])
                        )
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.warning(f"Failed to parse embedding_variance for user {user_id}: {e}")
                        embedding_dict['embedding_variance'] = None
                else:
                    embedding_dict['embedding_variance'] = None
                embeddings.append(embedding_dict)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get voice embeddings: {e}")
            return []
    
    def deactivate_old_embeddings(self, user_id: int):
        """Deactivate old voice embeddings when enrolling new ones"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE voice_embeddings SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to deactivate old embeddings: {e}")
    
    def update_voice_embedding(
        self, 
        user_id: int, 
        new_embedding: np.ndarray,
        learning_rate: float = 0.1
    ) -> bool:
        """
        Update voice embedding using adaptive learning
        
        Args:
            user_id: User ID
            new_embedding: New voice embedding from successful verification
            learning_rate: Learning rate for update (0.1 = 10% weight to new)
            
        Returns:
            True if update successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """SELECT * FROM voice_embeddings 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY created_at DESC LIMIT 1""",
                (user_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"No active voice embedding found for user {user_id}")
                return False
            
            # Get current embedding and variance
            current_embedding = np.array(json.loads(row['embedding_data']))
            current_variance = None
            if row['embedding_variance'] is not None and row['embedding_variance'] != '':
                try:
                    current_variance = np.array(json.loads(row['embedding_variance']))
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Could not parse variance for user {user_id}: {e}, continuing without variance update")
            
            # Adaptive update: 0.9 * old + 0.1 * new
            updated_embedding = (1 - learning_rate) * current_embedding + learning_rate * new_embedding
            updated_json = json.dumps(updated_embedding.tolist())
            
            # Update variance if it exists (slight increase to account for new data)
            variance_json = None
            if current_variance is not None:
                # Slightly increase variance to reflect model uncertainty with new data
                updated_variance = current_variance * (1 + learning_rate * 0.1)
                variance_json = json.dumps(updated_variance.tolist())
            
            # Update the database
            if variance_json:
                cursor.execute(
                    """UPDATE voice_embeddings 
                    SET embedding_data = ?, embedding_variance = ?, quality_score = quality_score + 0.01 
                    WHERE id = ?""",
                    (updated_json, variance_json, row['id'])
                )
            else:
                cursor.execute(
                    """UPDATE voice_embeddings 
                    SET embedding_data = ?, quality_score = quality_score + 0.01 
                    WHERE id = ?""",
                    (updated_json, row['id'])
                )
            
            self.conn.commit()
            logger.info(f"Voice embedding updated for user {user_id} with adaptive learning")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update voice embedding: {e}")
            return False
    
    # ==================== FACE EMBEDDING OPERATIONS ====================
    
    def store_face_embedding(
        self, 
        user_id: int, 
        embedding,
        embedding_type: str = "face_recognition",
        quality_score: float = 1.0
    ) -> Optional[int]:
        """
        Store face embedding for a user
        
        Args:
            user_id: User ID
            embedding: Face embedding (list for arrays, string for hashes)
            embedding_type: Type of model used
            quality_score: Quality assessment score
            
        Returns:
            Embedding ID if successful
        """
        try:
            # Serialize embedding to JSON (works for both lists and strings)
            embedding_json = json.dumps(embedding)
            
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO face_embeddings 
                (user_id, embedding_data, embedding_type, quality_score) 
                VALUES (?, ?, ?, ?)""",
                (user_id, embedding_json, embedding_type, quality_score)
            )
            
            self.conn.commit()
            embedding_id = cursor.lastrowid
            
            logger.info(f"Face embedding stored for user ID {user_id}")
            self.log_audit(user_id, "face_enrolled", f"Embedding ID: {embedding_id}")
            
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to store face embedding: {e}")
            return None
    
    def get_face_embeddings(self, user_id: int) -> List[Dict]:
        """
        Get all active face embeddings for a user

        Returns:
            List of embedding dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """SELECT * FROM face_embeddings
                WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC""",
                (user_id,)
            )

            rows = cursor.fetchall()
            embeddings = []

            for row in rows:
                embedding_dict = dict(row)
                # Parse the JSON embedding data
                try:
                    parsed_embedding = json.loads(embedding_dict['embedding_data'])
                    # If it's a string (hash), use it directly
                    if isinstance(parsed_embedding, str):
                        embedding_dict['embedding_data'] = parsed_embedding
                        embedding_dict['embedding_hash'] = parsed_embedding
                    # If it's a list (array), keep as is
                    elif isinstance(parsed_embedding, list):
                        embedding_dict['embedding_data'] = parsed_embedding
                        embedding_dict['embedding_hash'] = json.dumps(parsed_embedding)
                    else:
                        embedding_dict['embedding_data'] = str(parsed_embedding)
                        embedding_dict['embedding_hash'] = str(parsed_embedding)
                except (json.JSONDecodeError, TypeError):
                    # If it's not JSON, treat as raw hash string
                    embedding_dict['embedding_hash'] = embedding_dict['embedding_data']
                
                # Keep backward compatibility with array-based embeddings
                if not isinstance(embedding_dict.get('embedding_array'), list):
                    embedding_dict['embedding_array'] = [0.0] * 128
                
                embeddings.append(embedding_dict)

            return embeddings

        except Exception as e:
            logger.error(f"Failed to get face embeddings: {e}")
            return []
    
    def deactivate_old_face_embeddings(self, user_id: int):
        """Deactivate old face embeddings when enrolling new ones"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE face_embeddings SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to deactivate old face embeddings: {e}")
    
    # ==================== SESSION OPERATIONS ====================
    
    def create_session(self, user_id: int, session_token: str) -> Optional[int]:
        """Create a new session"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (user_id, session_token) VALUES (?, ?)",
                (user_id, session_token)
            )
            self.conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def update_session_verification(
        self, 
        session_token: str,
        voice_verified: bool = False,
        password_verified: bool = False,
        liveness_verified: bool = False
    ):
        """Update session verification status"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """UPDATE sessions 
                SET voice_verified = ?, password_verified = ?, liveness_verified = ?,
                    last_activity = CURRENT_TIMESTAMP
                WHERE session_token = ?""",
                (voice_verified, password_verified, liveness_verified, session_token)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
    
    def invalidate_session(self, session_token: str):
        """Invalidate a session"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
                (session_token,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
    
    # ==================== AUDIT LOG OPERATIONS ====================
    
    def log_audit(
        self, 
        user_id: Optional[int],
        action: str,
        details: Optional[str] = None,
        status: str = "success"
    ):
        """Log an audit event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO audit_logs (user_id, action, details, status) VALUES (?, ?, ?, ?)",
                (user_id, action, details, status)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")
    
    def get_audit_logs(self, user_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Get audit logs"""
        try:
            cursor = self.conn.cursor()
            
            if user_id:
                cursor.execute(
                    "SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
    
    # ==================== LIVENESS CHALLENGE OPERATIONS ====================
    
    def store_liveness_challenge(
        self,
        user_id: int,
        challenge_phrase: str,
        transcribed_text: Optional[str] = None,
        similarity_score: float = 0.0,
        passed: bool = False
    ) -> Optional[int]:
        """Store a liveness challenge attempt"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO liveness_challenges
                (user_id, challenge_phrase, transcribed_text, similarity_score, passed)
                VALUES (?, ?, ?, ?, ?)""",
                (user_id, challenge_phrase, transcribed_text, similarity_score, passed)
            )
            self.conn.commit()

            return cursor.lastrowid

        except Exception as e:
            logger.error(f"Failed to store liveness challenge: {e}")
            return None

    def get_liveness_challenges(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Get liveness challenges for a user

        Args:
            user_id: User ID
            limit: Maximum number of challenges to return

        Returns:
            List of liveness challenge dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """SELECT * FROM liveness_challenges
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit)
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get liveness challenges: {e}")
            return []

    # ==================== API KEY OPERATIONS ====================
    
    def create_api_key(self, key_hash: str, client_name: str, expires_at: Optional[str] = None) -> Optional[int]:
        """Create a new API key"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO api_keys (key_hash, client_name, expires_at) VALUES (?, ?, ?)",
                (key_hash, client_name, expires_at)
            )
            self.conn.commit()
            
            logger.info(f"API key created for client: {client_name}")
            return cursor.lastrowid
            
        except sqlite3.IntegrityError:
            logger.warning(f"API key hash already exists")
            return None
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            return None
            
    def validate_api_key(self, key_hash: str) -> Optional[Dict]:
        """Validate API key hash and return details if valid"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1",
                (key_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                key_dict = dict(row)
                # Check expiration
                if key_dict.get('expires_at'):
                    try:
                        # SQLite might return different ISO formats, be generous
                        expires_dt = datetime.fromisoformat(key_dict['expires_at'].replace('Z', '+00:00'))
                        if datetime.utcnow().timestamp() > expires_dt.timestamp():
                            self.revoke_api_key(key_dict['id'])
                            return None
                    except ValueError as e:
                        logger.warning(f"Failed to parse expiration date: {e}")
                return key_dict
            return None
            
        except Exception as e:
            logger.error(f"Failed to validate API key: {e}")
            return None
            
    def revoke_api_key(self, key_id: int) -> bool:
        """Revoke an API key"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE api_keys SET is_active = 0 WHERE id = ?",
                (key_id,)
            )
            self.conn.commit()
            logger.info(f"Revoked API key ID: {key_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False
            
    def get_all_api_keys(self) -> List[Dict]:
        """Get all API keys"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM api_keys ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get API keys: {e}")
            return []
