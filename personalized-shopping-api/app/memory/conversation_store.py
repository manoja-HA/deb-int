"""
Long-term conversation memory storage
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConversationStore:
    """Store and retrieve conversation history"""

    def __init__(self, storage_dir: str = "./data/conversations"):
        """
        Initialize conversation store

        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_conversation(
        self,
        session_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save a conversation turn

        Args:
            session_id: Session ID
            query: User query
            response: System response
            metadata: Optional metadata (state, metrics, etc.)
        """
        conversation_file = self.storage_dir / f"{session_id}.json"

        # Load existing conversation or create new
        if conversation_file.exists():
            with open(conversation_file, 'r') as f:
                conversation = json.load(f)
        else:
            conversation = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "turns": []
            }

        # Add new turn
        turn = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }

        conversation["turns"].append(turn)
        conversation["updated_at"] = datetime.now().isoformat()

        # Save
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f, indent=2)

        logger.debug(f"Saved conversation turn for session {session_id}")

    def get_conversation(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a conversation

        Args:
            session_id: Session ID

        Returns:
            Conversation dictionary or None
        """
        conversation_file = self.storage_dir / f"{session_id}.json"

        if not conversation_file.exists():
            return None

        with open(conversation_file, 'r') as f:
            return json.load(f)

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversations

        Args:
            limit: Maximum number to return

        Returns:
            List of conversation summaries
        """
        conversations = []

        for conv_file in sorted(
            self.storage_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]:
            try:
                with open(conv_file, 'r') as f:
                    conv = json.load(f)
                    conversations.append({
                        "session_id": conv["session_id"],
                        "created_at": conv["created_at"],
                        "turn_count": len(conv["turns"])
                    })
            except Exception as e:
                logger.warning(f"Failed to load conversation {conv_file}: {e}")

        return conversations
