"""
Automatic viral hook generator for AI-YouTube-Clipper.
Generates attention-grabbing, context-relevant short-form video hooks
with relevant emojis and optional Ollama intelligence.
"""

import logging
import re
from typing import Dict, List, Optional

from src.moment_detector import CandidateClip, OllamaClient

logger = logging.getLogger("AI-YouTube-Clipper.hook_generator")


# Category-aligned hook templates with emojis
CATEGORY_HOOK_TEMPLATES = {
    "funny": [
        "HE REALLY DID THAT 💀",
        "THIS HAS ME DEAD 😭💀",
        "CANNOT STOP LAUGHING AT THIS 😂",
        "WAIT FOR HIS REACTION 💀",
        "BRO DID NOT HESITATE 🤣"
    ],
    "surprise": [
        "NOBODY EXPECTED THIS 🤯",
        "WAIT FOR WHAT HAPPENS NEXT 😱",
        "THIS IS ACTUALLY UNREAL ⚡",
        "MY JAW LITERALLY DROPPED 🤯",
        "I AM IN COMPLETE SHOCK 😳"
    ],
    "gaming_reaction": [
        "THE MOST CLUTCH PLAY EVER 🔥",
        "HE ACTUALLY PULLED THIS OFF 🎮🔥",
        "GREATEST REACTION OF ALL TIME ⚡",
        "WATCH HIM COOK 💀🔥",
        "THIS IS INSANE 🏆"
    ],
    "argument": [
        "THINGS GOT COMPLETELY OUT OF HAND 🔥",
        "HE CALLED HIM OUT IMMEDIATELY 😳",
        "THE MOST INTENSE MOMENT EVER 💥",
        "NO WAY HE JUST SAID THAT 💀"
    ],
    "strong_opinion": [
        "THE HARSH TRUTH NOBODY ADMITS 🚨",
        "STOP MAKING THIS BIG MISTAKE 🛑",
        "THIS WILL CHANGE EVERYTHING 💡",
        "LISTEN TO THIS VERY CAREFULLY ⚠️"
    ],
    "emotional": [
        "THIS WILL BRING YOU TO TEARS 🥺❤️",
        "THE MOST TOUCHING MOMENT ❤️",
        "SOMETHING YOU WILL NEVER FORGET 🕊️"
    ],
    "question_curiosity": [
        "DID YOU ACTUALLY KNOW THIS? 🤔",
        "THE REAL TRUTH EXPLAINED 🧐",
        "WHY NOBODY UNDERSTANDS THIS 💡"
    ],
    "highlight": [
        "THIS MOMENT WAS UNREAL ⚡",
        "WAIT TILL THE VERY END 🎬",
        "WATCH WHAT HAPPENS HERE 👀"
    ]
}


class HookGenerator:
    """Generates punchy, relevant hooks for video clips."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama = ollama_client or OllamaClient()

    def generate_hook(self, clip: CandidateClip) -> str:
        """
        Generate a hook that matches the content of the clip.
        Uses Ollama if enabled/reachable; otherwise picks the most fitting template.
        """
        category = clip.category
        transcript = clip.transcript_text

        # If clip already has a specific spoken hook question or statement, leverage it
        if clip.hook and len(clip.hook.split()) <= 8:
            clean_hook = clip.hook.strip(" .?!").upper()
            if any(w in clean_hook for w in ["DID YOU KNOW", "WHAT IF", "HERE IS WHY", "WATCH WHAT"]):
                return f"{clean_hook} 🤯"

        templates = CATEGORY_HOOK_TEMPLATES.get(category, CATEGORY_HOOK_TEMPLATES["highlight"])

        # Pick template stably based on clip_id hash so it stays consistent
        idx = hash(clip.clip_id) % len(templates)
        return templates[idx]
