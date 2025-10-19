"""
Chat Router - STEP 3: INTENT DETECTION & QUERY PARSING ✓
Smart chatbot that understands user intent and extracts filters

Features Built:
- Step 1: Basic chat ✓
- Step 2: Conversation memory ✓
- Step 3: Intent detection & query parsing ✓ (YOU ARE HERE)
- Step 4: Recommendation engine integration (NEXT)
- Step 5: Advanced features
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import logging
import re
from groq import Groq
from sentence_transformers import SentenceTransformer


# Import config
from config import settings
from core.recommendation_engine import RecommendationEngine, SearchFilters

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

# Initialize Groq client
groq_client = Groq(api_key=settings.GROQ_API_KEY)


# Initialize text embedding model for search
_text_model = None

def get_text_model():
    """Get or create text embedding model."""
    global _text_model
    if _text_model is None:
        logger.info("Loading text embedding model...")
        _text_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _text_model

# Session storage for conversation memory
chat_sessions: Dict[str, List[Dict]] = {}
MAX_HISTORY = 10  # Keep last 10 messages per session

# Initialize recommendation engine (lazy load)
_recommendation_engine = None

def get_engine():
    """Get or create recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        logger.info("Loading recommendation engine...")
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine


# ========== Pydantic Models ==========

class QueryFilters(BaseModel):
    """Extracted filters from user query."""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    categories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    materials: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Single chat message in the conversation."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request from frontend."""
    message: str = Field(..., description="User's message")
    session_id: str = Field(default="default", description="Conversation session ID")


async def search_products_with_engine(query_text: str, filters: Dict, top_k: int = 8) -> List[Dict]:
    """
    Search products using recommendation engine.
    
    Args:
        query_text: The user's search query text
        filters: Extracted filters from intent detection
        top_k: Number of products to return
        
    Returns:
        List of product dictionaries
    """
    try:
        engine = get_engine()
        
        # Convert filters dict to SearchFilters object
        search_filters = SearchFilters(
            min_price=filters.get('min_price'),
            max_price=filters.get('max_price'),
            categories=filters.get('categories'),
            brands=filters.get('brands'),
            materials=filters.get('materials'),
            colors=filters.get('colors')
        )
        
        # Generate embedding from user's query text
        text_model = get_text_model()
        logger.info(f"Generating embedding for query: {query_text}")
        query_embedding = text_model.encode(query_text).tolist()
        
        # Search with filters
        results = engine.text_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=search_filters
        )
        
        # Convert to dict format
        products = [result.to_dict() for result in results]
        
        logger.info(f"Found {len(products)} products for query: '{query_text}'")
        return products
        
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        return []

class ChatResponse(BaseModel):
    """Response to frontend."""
    message: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Session ID")
    intent: str = Field(default="chat", description="Detected intent")
    filters: Optional[QueryFilters] = Field(None, description="Extracted filters")
    products: Optional[List[Dict]] = Field(None, description="Product recommendations")

    


# ========== Enhanced System Prompt ==========

SYSTEM_PROMPT = """You are an expert furniture shopping assistant with deep product knowledge.

**Your Capabilities:**
- Help users find furniture based on their needs
- Answer questions about materials, dimensions, care
- Provide style and design advice
- Compare products when asked
- Remember conversation context

**Response Style:**
- Friendly and enthusiastic but professional
- Concise (100-150 words max unless detail requested)
- Ask clarifying questions when needed
- Use furniture terminology appropriately

**Important:**
When users search for furniture, acknowledge their request naturally and let the system handle showing products."""


INTENT_DETECTION_PROMPT = """Analyze this user message and classify the intent.

User message: "{message}"

Classify as ONE of these intents:
1. "search" - User wants to find/browse furniture (e.g., "show me sofas", "I need a chair")
2. "compare" - User wants to compare specific products (e.g., "compare these two", "what's the difference")
3. "recommend" - User asks for recommendations based on preferences (e.g., "what do you recommend", "help me choose")
4. "product_question" - Asking about specific product details (dimensions, material, etc.)
5. "general_chat" - General conversation, greetings, or off-topic

Also extract any filters mentioned:
- Price range (e.g., "under $500", "$100-300")
- Categories (e.g., "sofa", "chair", "table", "bed")
- Colors (e.g., "blue", "gray", "white")
- Materials (e.g., "leather", "wood", "metal")
- Brands (e.g., "IKEA", "Ashley")

Respond in this EXACT JSON format:
{{
    "intent": "search|compare|recommend|product_question|general_chat",
    "confidence": 0.0-1.0,
    "filters": {{
    "min_price": null or number,
    "max_price": null or number,
    "categories": [] or ["category1", "category2"],
    "colors": [] or ["color1"],
    "materials": [] or ["material1"],
    "brands": [] or ["brand1"]
    }},
    "search_query": "cleaned up search query" or null
}}

Only respond with valid JSON, nothing else."""


# ========== Intent Detection ==========

def detect_intent(message: str) -> Dict:
    """
    Detect user intent and extract filters using LLM.
    
    Args:
        message: User's message
        
    Returns:
        Dict with intent, confidence, filters, and search_query
    """
    try:
        # Call Groq for intent detection
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intent classification expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": INTENT_DETECTION_PROMPT.format(message=message)
                }
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=300
        )
        
        # Parse JSON response
        import json
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = re.sub(r'```json?\n?|\n?```', '', result_text).strip()
        
        result = json.loads(result_text)
        
        logger.info(f"Intent detected: {result['intent']} (confidence: {result.get('confidence', 0)})")
        
        return result
        
    except Exception as e:
        logger.error(f"Intent detection failed: {e}, defaulting to general_chat")
        # Fallback to simple pattern matching
        return fallback_intent_detection(message)


def fallback_intent_detection(message: str) -> Dict:
    """
    Simple rule-based fallback if LLM intent detection fails.
    
    Args:
        message: User's message
        
    Returns:
        Dict with intent and basic filters
    """
    message_lower = message.lower()
    
    # Search patterns
    search_keywords = ['show', 'find', 'search', 'looking for', 'need', 'want', 'browse']
    compare_keywords = ['compare', 'difference', 'versus', 'vs', 'better']
    recommend_keywords = ['recommend', 'suggest', 'advice', 'help me choose', 'which one']
    
    # Detect intent
    if any(kw in message_lower for kw in compare_keywords):
        intent = "compare"
    elif any(kw in message_lower for kw in recommend_keywords):
        intent = "recommend"
    elif any(kw in message_lower for kw in search_keywords):
        intent = "search"
    else:
        intent = "general_chat"
    
    # Extract price range
    min_price, max_price = extract_price_range(message)
    
    # Extract categories (simple matching)
    categories = []
    category_keywords = ['sofa', 'chair', 'table', 'bed', 'desk', 'cabinet', 'shelf', 'dresser']
    for cat in category_keywords:
        if cat in message_lower:
            categories.append(cat)
    
    return {
        "intent": intent,
        "confidence": 0.6,
        "filters": {
            "min_price": min_price,
            "max_price": max_price,
            "categories": categories if categories else None,
            "colors": None,
            "materials": None,
            "brands": None
        },
        "search_query": message
    }


def extract_price_range(text: str) -> tuple:
    """
    Extract min and max price from text.
    
    Args:
        text: User message
        
    Returns:
        Tuple of (min_price, max_price)
    """
    min_price = None
    max_price = None
    
    # Pattern: "under $500", "below $300"
    under_match = re.search(r'(?:under|below|less than|max)\s*\$?(\d+)', text, re.I)
    if under_match:
        max_price = float(under_match.group(1))
    
    # Pattern: "over $100", "above $200", "at least $150"
    over_match = re.search(r'(?:over|above|more than|at least|min)\s*\$?(\d+)', text, re.I)
    if over_match:
        min_price = float(over_match.group(1))
    
    # Pattern: "$100-$500", "$100 to $500"
    range_match = re.search(r'\$?(\d+)\s*(?:-|to)\s*\$?(\d+)', text, re.I)
    if range_match:
        min_price = float(range_match.group(1))
        max_price = float(range_match.group(2))
    
    return min_price, max_price


# ========== MAIN ENDPOINT ==========

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    INTELLIGENT CHAT WITH INTENT DETECTION
    
    - Detects what user wants (search, compare, recommend, chat)
    - Extracts filters from natural language
    - Routes to appropriate handler
    - Remembers conversation context
    """
    try:
        logger.info(f"Chat request (session: {request.session_id}): {request.message[:50]}...")
        
        # Step 1: Detect Intent & Extract Filters
        intent_result = detect_intent(request.message)
        intent = intent_result.get("intent", "general_chat")
        filters = intent_result.get("filters", {})
        
        logger.info(f"Intent: {intent}, Filters: {filters}")
        
        # Step 2: Get or create session history
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []
            logger.info(f"Created new session: {request.session_id}")
        
        session_history = chat_sessions[request.session_id]
        
        # Step 3: Build messages with history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history
        for msg in session_history[-MAX_HISTORY:]:
            messages.append(msg)
        
        # Add current user message
        current_message = {"role": "user", "content": request.message}
        messages.append(current_message)
        
        # Step 4: Generate response based on intent
        if intent in ["search", "recommend"]:
            # Search products and generate response
            response_text, products = await generate_search_response(request.message, filters, messages)
            
        elif intent == "compare":
            # For compare, acknowledge
            response_text = await generate_compare_response(request.message, messages)
            products = None  # Will be populated in Feature 4
            
        else:
            # General chat
            response_text = await generate_general_response(messages)
            products = None
        
        # Step 5: Save to session history
        assistant_msg_dict = {"role": "assistant", "content": response_text}
        session_history.append(current_message)
        session_history.append(assistant_msg_dict)
        
        # Trim history if too long
        if len(session_history) > MAX_HISTORY * 2:
            chat_sessions[request.session_id] = session_history[-(MAX_HISTORY * 2):]
        
        logger.info(f"Response generated (intent: {intent})")
        
        return ChatResponse(
            message=response_text,
            session_id=request.session_id,
            intent=intent,
            filters=QueryFilters(**filters) if filters else None,
            products=products
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


async def generate_search_response(message: str, filters: Dict, messages: List[Dict]) -> tuple:
    """
    Generate response for search/recommend intent.
    Returns (response_text, products_list)
    """
    # Search products using the actual user message as query
    products = await search_products_with_engine(message, filters, top_k=8)
    
    # Generate natural language response
    if products:
        system_msg = f"User is searching for furniture. We found {len(products)} products matching their query: '{message}'. Acknowledge their search briefly and mention you're showing them the best matches. Be enthusiastic but concise (2-3 sentences max)."
        messages[0]["content"] = system_msg
    else:
        system_msg = f"User searched for '{message}' but we found no matching products. Apologize and suggest they try different search terms or adjust filters. Be helpful and brief."
        messages[0]["content"] = system_msg
    
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    
    response_text = completion.choices[0].message.content
    
    return response_text, products


async def generate_compare_response(message: str, messages: List[Dict]) -> str:
    """Generate response for compare intent."""
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=400
    )
    return completion.choices[0].message.content


async def generate_general_response(messages: List[Dict]) -> str:
    """Generate response for general chat."""
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=400
    )
    return completion.choices[0].message.content


# ========== Session Management ==========

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        logger.info(f"Cleared session: {session_id}")
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get conversation history for a session."""
    if session_id in chat_sessions:
        return {
            "session_id": session_id,
            "message_count": len(chat_sessions[session_id]),
            "history": chat_sessions[session_id]
        }
    return {"message": "Session not found", "history": []}


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "active_sessions": len(chat_sessions),
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(history)
            }
            for sid, history in chat_sessions.items()
        ]
    }


# ========== Health Check ==========

@router.get("/health")
async def chat_health():
    """Check if chat service is working."""
    try:
        # Quick test with Groq
        test = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        
        return {
            "status": "healthy",
            "groq_connected": True,
            "model": "llama-3.3-70b-versatile",
            "active_sessions": len(chat_sessions)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }