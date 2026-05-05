from __future__ import annotations

from typing import Dict

# Therapy-friendly phonological process descriptions used by the API.
PROCESS_LIBRARY: Dict[str, Dict[str, str]] = {
    "gliding": {
        "label": "Gliding",
        "description": "Liquids like /r/ or /l/ are produced as glides (often /w/ or /y/).",
        "therapy_focus": "Work on lip rounding vs. tongue shaping for /r,l/; start with syllables and minimal pairs.",
    },
    "stopping": {
        "label": "Stopping",
        "description": "Fricatives/affricates are produced as stops (airflow is stopped instead of continuous).",
        "therapy_focus": "Cue continuous airflow (“long sound”) for fricatives; use visual airflow cues (tissue, hand).",
    },
    "fronting": {
        "label": "Fronting",
        "description": "Back sounds (velars /k,g/) are produced as front sounds (often /t,d/).",
        "therapy_focus": "Cue tongue-back lift (\"back sound\"); try /k/ in isolation then CV (ku, ka).",
    },
    "backing": {
        "label": "Backing",
        "description": "Front sounds (often alveolars /t,d/) are produced as back sounds (often /k,g/).",
        "therapy_focus": "Cue tongue-tip placement behind teeth; contrast /t/ vs /k/ with tactile cues.",
    },
    "voicing": {
        "label": "Voicing",
        "description": "Voiceless sounds are produced as voiced sounds (e.g., /t/→/d/).",
        "therapy_focus": "Use hand-on-throat cue to feel voice on/off; practice minimal pairs.",
    },
    "devoicing": {
        "label": "Devoicing",
        "description": "Voiced sounds are produced as voiceless sounds (e.g., /b/→/p/).",
        "therapy_focus": "Cue vibration for voiced targets; elongate voiced sounds where possible (e.g., /z/).",
    },
    "deaffrication": {
        "label": "Deaffrication",
        "description": "Affricates (like /ch/) are produced as fricatives or stops.",
        "therapy_focus": "Shape the stop+fricative sequence; start with \"t\" + \"sh\" blended to /ch/.",
    },
    "palatalization": {
        "label": "Palatalization",
        "description": "Alveolars (like /s/) shift toward palatal/postalveolar (like /sh/).",
        "therapy_focus": "Cue tongue groove and \"smile\" lips for /s/; contrast /s/ vs /sh/ in minimal pairs.",
    },
    "depalatalization": {
        "label": "Depalatalization",
        "description": "Postalveolars (like /sh/) shift toward alveolars (like /s/).",
        "therapy_focus": "Cue tongue retraction and rounding for /sh/; use \"quiet\" /sh/ vs \"snake\" /s/ contrast.",
    },
}

