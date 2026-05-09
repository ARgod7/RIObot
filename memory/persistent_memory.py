"""
Persistent User Profiles (Mock)
Tracks user metadata across sessions for adaptive therapy
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import os


@dataclass
class UserProfile:
    """User stored data across sessions"""
    user_id: str
    name: str
    total_sessions: int = 0
    last_session: str = ""
    trust_level: float = 0.5  # 0-1, grows with positive interactions
    likeness_score: float = 0.6  # 0-1, how much they enjoy talking

    # Learned interventions (what worked in past)
    interventions_effective: Dict[str, int] = field(default_factory=lambda: {
        "validation": 0,
        "reframe": 0,
        "deepening": 0,
        "grounding": 0,
    })

    grief_triggers: List[str] = field(default_factory=list)  # Words that upset them
    joy_triggers: List[str] = field(default_factory=list)  # Topics they enjoy

    favorite_games: List[str] = field(default_factory=list)  # Games they prefer
    conversation_style: str = "warm"  # how to address them

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'UserProfile':
        return UserProfile(**data)


class PersistentMemory:
    """Mock persistent storage for user profiles"""

    def __init__(self, filepath: str = "memory/mock_data/user_profiles.json"):
        self.filepath = filepath
        self.profiles: Dict[str, UserProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        """Load profiles from JSON"""
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for user_id, profile_data in data.get("users", {}).items():
                        self.profiles[user_id] = UserProfile.from_dict(profile_data)
                print(f"✓ Loaded {len(self.profiles)} user profiles")
            else:
                print("⚠️ No existing profiles found, starting fresh")
        except Exception as e:
            print(f"⚠️ Failed to load profiles: {e}")

    def get_user(self, user_id: str = "default_user") -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(
                user_id=user_id,
                name="User",
                last_session=datetime.now().isoformat()
            )
        return self.profiles[user_id]

    def update_intervention_effectiveness(
        self,
        user_id: str,
        intervention_type: str,
        was_effective: bool
    ):
        """Learn which interventions work for this user"""
        profile = self.get_user(user_id)
        if intervention_type in profile.interventions_effective:
            if was_effective:
                profile.interventions_effective[intervention_type] += 1

    def add_grief_trigger(self, user_id: str, trigger: str):
        """Learn what topics upset them"""
        profile = self.get_user(user_id)
        if trigger not in profile.grief_triggers:
            profile.grief_triggers.append(trigger)

    def add_joy_trigger(self, user_id: str, trigger: str):
        """Learn what topics they enjoy"""
        profile = self.get_user(user_id)
        if trigger not in profile.joy_triggers:
            profile.joy_triggers.append(trigger)

    def get_best_intervention(self, user_id: str) -> str:
        """Which intervention worked best for them?"""
        profile = self.get_user(user_id)
        best = max(profile.interventions_effective, key=profile.interventions_effective.get)
        return best if profile.interventions_effective[best] > 0 else "validation"

    def update_user_metrics(
        self,
        user_id: str,
        session_positive: bool,
        conversation_quality: float = 0.7
    ):
        """Update trust & likeness after session"""
        profile = self.get_user(user_id)
        profile.total_sessions += 1
        profile.last_session = datetime.now().isoformat()

        # Trust grows with positive sessions
        if session_positive:
            profile.trust_level = min(1.0, profile.trust_level + 0.05)
        else:
            profile.trust_level = max(0.2, profile.trust_level - 0.02)

        # Likeness grows with good conversations
        profile.likeness_score = min(1.0, conversation_quality + 0.05)

    def save(self):
        """Persist profiles to disk"""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            data = {
                "users": {uid: p.to_dict() for uid, p in self.profiles.items()}
            }
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Profiles saved to {self.filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save profiles: {e}")


# Global instance
_persistent_memory = None

def get_persistent_memory(filepath: str = "memory/mock_data/user_profiles.json") -> PersistentMemory:
    """Get or create persistent memory"""
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemory(filepath)
    return _persistent_memory


def save_all_profiles():
    """Save all user profiles"""
    get_persistent_memory().save()

