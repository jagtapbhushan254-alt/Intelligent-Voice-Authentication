"""
SecureX-Assist - Intent Classifier
Natural language intent recognition using SentenceTransformers
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger("core.intent_classifier")

# --- START: Added basic logging config ---
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
# --- END: Added basic logging config ---

class IntentClassifier:
    """
    Lightweight intent classifier using SentenceTransformers for natural language queries
    """
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model = None
        self.intents = {}
        self.intent_embeddings = {}
        self.threshold = 0.6  # --- FIX: Lowered threshold for better matching ---
        
        # Performance optimization: Cache embeddings for recent queries
        self._embedding_cache = {}  # {text: embedding}
        self._cache_max_size = 100
        
        # Try to load the model, but don't fail if it doesn't work
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Intent classifier model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer model: {e}")
            logger.info("Intent classifier will use keyword matching fallback")
            self.model = None
        
    def add_intent(self, intent_name: str, examples: List[str]):
        """
        Add an intent with example phrases
        """
        self.intents[intent_name] = examples
        if self.model:
            # Compute embeddings for examples
            embeddings = self.model.encode(examples)
            # Store mean embedding for the intent
            self.intent_embeddings[intent_name] = np.mean(embeddings, axis=0)
            logger.info(f"Added intent '{intent_name}' with {len(examples)} examples")
        else:
            # Store examples for keyword matching
            self.intent_embeddings[intent_name] = examples
            logger.info(f"Added intent '{intent_name}' with keyword examples (no ML model)")
    
    def classify(self, text: str) -> Tuple[Optional[str], float]:
        """
        Classify the intent of a text query (optimized with caching)
        """
        if not self.intent_embeddings:
            logger.warning("Classifier has no intents loaded.")
            return None, 0.0
            
        if self.model:
            # Check cache first
            if text in self._embedding_cache:
                text_embedding = self._embedding_cache[text]
            else:
                # Use ML-based classification
                text_embedding = self.model.encode([text])[0]
                
                # Cache the embedding
                if len(self._embedding_cache) >= self._cache_max_size:
                    # Remove oldest entry (simple FIFO)
                    self._embedding_cache.pop(next(iter(self._embedding_cache)))
                self._embedding_cache[text] = text_embedding
            
            # Compute similarities with all intents (vectorized for speed)
            similarities = {}
            for intent_name, intent_embedding in self.intent_embeddings.items():
                # Faster cosine similarity computation
                sim = np.dot(text_embedding, intent_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(intent_embedding)
                )
                similarities[intent_name] = sim
            
            # Find best match
            best_intent = max(similarities, key=similarities.get)
            best_score = similarities[best_intent]
            
            # Check threshold
            if best_score >= self.threshold:
                logger.info(f"Classified '{text}' as intent '{best_intent}' with confidence {best_score:.3f}")
                return best_intent, best_score
            else:
                logger.info(f"No intent matched for '{text}' (best: {best_intent} with {best_score:.3f})")
                return None, best_score
        else:
            # Use keyword-based classification (fast fallback)
            text_lower = text.lower()
            for intent_name, examples in self.intents.items():
                for example in examples:
                    if example.lower() in text_lower:
                        logger.info(f"Keyword matched '{text}' as intent '{intent_name}'")
                        return intent_name, 1.0
            
            logger.info(f"No keyword match for '{text}'")
            return None, 0.0
    
    def setup_default_intents(self):
        """
        Add all default intents for the SecureX Voice Assistant.
        """
        logger.info("Setting up default intents...")

        # --- Greetings & Politeness ---
        self.add_intent("greeting", [
            "hello", "hi", "hey there", "greetings"
        ])
        self.add_intent("goodbye", [
            "goodbye", "bye", "see you later", "exit"
        ])
        self.add_intent("good_morning", ["good morning"])
        self.add_intent("good_afternoon", ["good afternoon"])
        self.add_intent("good_evening", ["good evening"])
        self.add_intent("good_night", ["good night"])
        self.add_intent("thank_you", ["thank you", "thanks"])
        self.add_intent("sorry", ["sorry", "my apologies"])
        self.add_intent("how_are_you", ["how are you", "how are you doing"])

        # --- System Commands ---
        self.add_intent("system_lock", [
            "lock system", "lock my computer", "lock this device", "secure computer"
        ])
        self.add_intent("system_restart", [
            "restart system", "restart my computer", "reboot"
        ])
        self.add_intent("system_shutdown", [
            "shutdown system", "shut down my computer", "turn off computer"
        ])
        self.add_intent("system_info", [
            "system status", "show system info", "how is the computer running", "check cpu"
        ])

        # --- Application Commands ---
        self.add_intent("open_calculator", [
            "open calculator", "launch calculator", "run calculator", "calculator"
        ])
        self.add_intent("open_notepad", [
            "open notepad", "launch notepad", "open text editor", "new note"
        ])
        self.add_intent("open_explorer", [
            "open file explorer", "launch file explorer", "show my files", "open files"
        ])
        self.add_intent("open_app", [
            "open app", "launch application", "open chrome", "launch firefox", "open word"
        ])
        self.add_intent("close_app", [
            "close application", "close this app", "exit program"
        ])
        
        # --- START: Added new intents ---
        self.add_intent("open_youtube", [
            "open youtube", "launch youtube"
        ])
        self.add_intent("play_video", [
            "play this video", "play", "search youtube for", "find this video on youtube", "play on youtube"
        ])
        self.add_intent("open_browser", [
            "open browser", "launch browser", "open google"
        ])
        self.add_intent("search_google", [
            "search for", "google", "search on google", "find me information on"
        ])
        # --- END: Added new intents ---

        # --- Window & Media ---
        self.add_intent("minimize_window", [
            "minimize this", "minimize window", "hide this"
        ])
        self.add_intent("maximize_window", [
            "maximize this", "maximize window", "full screen"
        ])
        self.add_intent("play_music", ["play music", "play a song"])
        self.add_intent("pause_music", ["pause music", "stop music"])
        self.add_intent("mute_audio", ["mute", "mute volume", "silence"])
        self.add_intent("unmute_audio", ["unmute", "restore volume", "speak up"])

        # --- File System ---
        self.add_intent("list_files", [
            "list files", "what files are here", "show files in directory"
        ])
        self.add_intent("delete_file", [
            "delete a file", "remove this file"
        ])
        self.add_intent("open_file", [
            "open a file", "read this document"
        ])
        self.add_intent("search_files", [
            "search for a file", "find a file", "where is my document"
        ])

        # --- Assistant Specific ---
        self.add_intent("what_can_you_do", [
            "what can you do", "help", "show commands", "abilities", "what do you do"
        ])
        self.add_intent("who_are_you", [
            "who are you", "what is your name"
        ])
        self.add_intent("take_screenshot", [
            "take a screenshot", "capture the screen", "save screen"
        ])

        # --- Security & SecureComms ---
        self.add_intent("run_security_scan", [
            "run security scan", "scan for viruses"
        ])
        self.add_intent("show_logs", [
            "show security logs", "view audit trail"
        ])
        self.add_intent("biometric_status", [
            "check biometric status", "is biometrics active"
        ])
        self.add_intent("send_secure_message", [
            "send a secure message", "new message"
        ])
        self.add_intent("check_messages", [
            "check my messages", "do I have new messages"
        ])
        self.add_intent("read_last_message", [
            "read my last message", "what was the last message"
        ])
        self.add_intent("start_voice_call", [
            "start a voice call", "call Aayush"
        ])

        # --- Fun & Interactive ---
        self.add_intent("tell_joke", [
            "tell me a joke", "make me laugh"
        ])
        self.add_intent("play_game", [
            "let's play a game", "play a game"
        ])
        self.add_intent("motivate_me", [
            "motivate me", "give me motivation"
        ])
        self.add_intent("compliment_me", [
            "compliment me", "say something nice"
        ])
        self.add_intent("tell_fact", [
            "tell me a fact", "interesting fact"
        ])
        self.add_intent("sing_song", [
            "sing me a song", "sing a song"
        ])
        self.add_intent("dance", [
            "can you dance", "do a dance"
        ])
        self.add_intent("tell_story", [
            "tell me a story", "read a story"
        ])

        # --- General Knowledge ---
        self.add_intent("get_time", [
            "what time is it", "tell me the time"
        ])
        self.add_intent("get_date", [
            "what is the date", "what's today's date"
        ])
        self.add_intent("show_weather", [
            "what's the weather", "show me the forecast", "how is the weather in"
        ])

        logger.info(f"Setup {len(self.intents)} default intents")