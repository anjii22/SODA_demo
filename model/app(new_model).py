"""
SODA speech therapy API — **phoneme-level ASR** backend.

Decodes audio with a **phoneme CTC** model (IPA → ARPAbet), optional **Wav2Vec2**
word CTC, and **Groq Whisper Large** (``whisper-large-v3`` by default) for a third
word hypothesis. Transcripts are converted to phonemes with ``g2p_en``; streams are
**aligned and fused** (CTC+w2v, then Whisper G2P merged with the same NW rules; disable
with ``GROQ_ASR_ENABLED=false`` if you do not want external API calls).
Optional **Groq Llama** consumes one bundled pipeline JSON (all ASR streams + automatic metrics together) and returns strict JSON blended into **combined_*** fields alongside the automatic scores (adjunct only; core SODA keys unchanged).
For local testing this file may set the key directly (see ``_GROQ_API_KEY_LOCAL_TEST``); use your backed-up copy for anything shared.

Based on `app(highadvanced)whisper.py`, adapted for phoneme-first decoding.
"""

import json
import logging
import math
import os
import sys
import re
import subprocess
import tempfile
import uuid
from io import BytesIO
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import groupby, zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_DIR = Path(__file__).resolve().parent

from phoneme_data import ARPA_VOWELS, IPA_TO_ARPA, PHONEME_FEATURES, TONGUE_FEEDBACK, VOWELS
from groq_prompts import GROQ_COMBINED_JUDGE_SYSTEM
from therapy_data import PROCESS_LIBRARY

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from flask import Flask, jsonify, request
from g2p_en import G2p
from transformers import AutoModelForCTC, AutoProcessor

try:
    import noisereduce as nr

    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    nr = None  # type: ignore[assignment]
    _NOISEREDUCE_AVAILABLE = False

try:
    from groq import Groq as GroqClient

    _GROQ_SDK_AVAILABLE = True
except ImportError:
    GroqClient = None  # type: ignore[assignment]
    _GROQ_SDK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not _GROQ_SDK_AVAILABLE:
    logger.warning(
        "Groq SDK not importable — run: `%s -m pip install groq` (use the SAME interpreter as this app)",
        sys.executable,
    )


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    s = v.strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


# CONFIG — English phoneme CTC (IPA labels; same tokenizer family as the old base checkpoint).
# Default: XLS-R 300M English phoneme CTC (~0.09 PER on the author's eval — much stronger than wav2vec2-base).
# Override with `ASR_MODEL_PHONEME`. Legacy base: `bobboyms/wav2vec2-base-en-phoneme-ctc-41h`
ASR_MODEL_PHONEME = os.getenv(
    "ASR_MODEL_PHONEME",
    "bobboyms/wav2vec2-xls-r-300m-en-phoneme-ctc-41h",
)
# Light noise suppression before ASR (non-stationary reduction when `noisereduce` is installed).
# Set `AUDIO_DENOISE=false` to disable. Install optional: `pip install noisereduce`
AUDIO_DENOISE = _env_bool("AUDIO_DENOISE", True)
# GPU only: set `ASR_FP16=true` for lower VRAM / faster inference (slightly less stable than fp32).
ASR_FP16 = _env_bool("ASR_FP16", False)
# Optional second pass: English word CTC (960h) → G2P phonemes, fused with phoneme CTC.
# Set `ASR_WORD_ENABLED=false` to save VRAM / startup time.
ASR_WORD_ENABLED = _env_bool("ASR_WORD_ENABLED", True)
ASR_MODEL_WORD = os.getenv(
    "ASR_MODEL_WORD",
    # Wav2Vec2 Large (word-level); override with `ASR_MODEL_WORD` if needed.
    "facebook/wav2vec2-large-960h",
)
# Groq Whisper Large → first word → g2p_en phonemes, chained into fusion after CTC+w2v (default on if SDK+key).
_GROQ_API_KEY_LOCAL_TEST = "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"
os.environ["GROQ_API_KEY"] = _GROQ_API_KEY_LOCAL_TEST.strip()

GROQ_ASR_ENABLED = _env_bool("GROQ_ASR_ENABLED", True)
# Groq hosts Whisper Large as `whisper-large-v3`; turbo is a faster fallback.
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
GROQ_WHISPER_FALLBACK_MODEL = os.getenv(
    "GROQ_WHISPER_FALLBACK_MODEL",
    "whisper-large-v3-turbo",
).strip()
# Groq chat (Llama): given expected word + dictionary phones + pipeline-identified phones, judges closeness
# ( adjunct only — same API key; pipeline scoring unchanged ).
GROQ_PHONEME_JUDGE_ENABLED = _env_bool("GROQ_PHONEME_JUDGE_ENABLED", True)
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
# Comma-separated fallbacks if the primary chat model rejects the request (deprecations, limits).
GROQ_CHAT_MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv("GROQ_CHAT_MODEL_FALLBACKS", "llama-3.1-8b-instant").split(",")
    if m.strip()
]
# When Groq returns a calibrated JSON, blend with automatic severities (higher = weight on API).
_w_api = os.getenv("GROQ_COMBINED_WEIGHT_API", "0.55").strip()
try:
    GROQ_COMBINED_WEIGHT_API = max(0.0, min(1.0, float(_w_api)))
except ValueError:
    GROQ_COMBINED_WEIGHT_API = 0.55
GROQ_COMBINED_WEIGHT_LLM = round(1.0 - GROQ_COMBINED_WEIGHT_API, 4)

BASE_DIR = _PROJECT_DIR
TEMP_DIR = Path(tempfile.gettempdir())
MAX_AUDIO_DURATION_SEC = 6
SAMPLE_RATE = 16000
MAX_WORD_LENGTH = 30

# Compare phonemes after mapping ARPAbet symbols to perceptual equivalence classes
# (dictionary / g2p notation vs ASR often differ in spelling for similar phones).
# strict = exact token match; lenient = collapse common confusable vowels (and a few cons).
_raw_phon_eq = os.getenv("PHONEME_EQUIVALENCE_MODE", "lenient").strip().lower()
PHONEME_EQUIVALENCE_MODE: str = (
    "strict" if _raw_phon_eq in {"strict", "exact", "0", "false", "off"} else "lenient"
)

_PHONE_EQUIVALENCE_GROUPS: Tuple[frozenset[str], ...] = (
    # Schwa / low vowels: g2p often uses AH/AX while ASR uses AW for similar quality.
    frozenset({"ah", "ax", "aw"}),
    frozenset({"er", "axr"}),
    frozenset({"ih", "iy"}),
    frozenset({"eh", "ey"}),
    frozenset({"aa", "ao"}),
    frozenset({"uh", "uw"}),
    frozenset({"t", "dx"}),
)


def _build_phoneme_canonical_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for g in _PHONE_EQUIVALENCE_GROUPS:
        leader = sorted(g, key=lambda s: (len(s), s))[0]
        for p in g:
            m[p.lower()] = leader.lower()
    return m


_PHON_CANONICAL: Dict[str, str] = _build_phoneme_canonical_map()


def canonical_phoneme(symbol: str) -> str:
    """Map one ARPAbet-style symbol to comparison token (identity in strict mode)."""
    x = symbol.strip().lower()
    if PHONEME_EQUIVALENCE_MODE == "strict" or not x:
        return x
    return _PHON_CANONICAL.get(x, x)


def canonical_phoneme_seq(phones: List[str]) -> List[str]:
    return [canonical_phoneme(p) for p in phones]


def phonemes_symbol_match(expected: str, produced: str) -> bool:
    return canonical_phoneme(expected) == canonical_phoneme(produced)


def phoneme_sequences_equivalent(seq_a: List[str], seq_b: List[str]) -> bool:
    return canonical_phoneme_seq(seq_a) == canonical_phoneme_seq(seq_b)


def phoneme_sequence_severity(expected_ph: List[str], predicted_ph: List[str]) -> float:
    """Sequence similarity with optional equivalence collapsing (therapy-friendly)."""
    return calculate_severity(
        canonical_phoneme_seq(expected_ph),
        canonical_phoneme_seq(predicted_ph),
    )


# LOAD MODELS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

_asr_dtype = torch.float32
if device.type == "cuda" and ASR_FP16:
    _asr_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

phoneme_processor = AutoProcessor.from_pretrained(ASR_MODEL_PHONEME)
_load_kw: Dict[str, Any] = {"low_cpu_mem_usage": True}
if _asr_dtype != torch.float32:
    _load_kw["torch_dtype"] = _asr_dtype
phoneme_model = AutoModelForCTC.from_pretrained(ASR_MODEL_PHONEME, **_load_kw).to(device)
phoneme_model.eval()

logger.info(
    "Phoneme ASR: model=%s dtype=%s | audio denoise=%s (noisereduce=%s)",
    ASR_MODEL_PHONEME,
    str(_asr_dtype),
    AUDIO_DENOISE,
    _NOISEREDUCE_AVAILABLE,
)

word_processor = None
word_model = None
if ASR_WORD_ENABLED:
    try:
        word_processor = AutoProcessor.from_pretrained(ASR_MODEL_WORD)
        _wkw: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if _asr_dtype != torch.float32:
            _wkw["torch_dtype"] = _asr_dtype
        word_model = AutoModelForCTC.from_pretrained(ASR_MODEL_WORD, **_wkw).to(device)
        word_model.eval()
        logger.info("Word ASR (fusion): model=%s", ASR_MODEL_WORD)
    except Exception as e:
        logger.warning("Word ASR disabled (load failed): %s", e)
        word_processor = None
        word_model = None


def word_asr_ready() -> bool:
    return word_processor is not None and word_model is not None


def groq_api_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def groq_asr_ready() -> bool:
    return bool(_GROQ_SDK_AVAILABLE and GroqClient is not None and groq_api_configured() and GROQ_ASR_ENABLED)


def groq_phoneme_judge_ready() -> bool:
    """Same `GROQ_API_KEY` as Whisper; runs even if `GROQ_ASR_ENABLED` is false."""
    return bool(
        _GROQ_SDK_AVAILABLE
        and GroqClient is not None
        and groq_api_configured()
        and GROQ_PHONEME_JUDGE_ENABLED
    )


def groq_env_diagnostics() -> Dict[str, Any]:
    """Why Groq may be off (no raw key)."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return {
        "groq_python_package_installed": _GROQ_SDK_AVAILABLE,
        "python_executable": sys.executable,
        "GROQ_API_key_present": bool(key),
        "GROQ_API_key_length_chars": len(key),
        "GROQ_ASR_ENABLED": GROQ_ASR_ENABLED,
        "GROQ_PHONEME_JUDGE_ENABLED": GROQ_PHONEME_JUDGE_ENABLED,
        "groq_local_embedded_key_used": bool(_GROQ_API_KEY_LOCAL_TEST.strip()),
        "groq_asr_ready": groq_asr_ready(),
        "groq_phoneme_judge_ready": groq_phoneme_judge_ready(),
    }


if GROQ_ASR_ENABLED:
    if _GROQ_SDK_AVAILABLE and groq_api_configured():
        logger.info("Groq ASR (fusion): model=%s", GROQ_WHISPER_MODEL)
    elif _GROQ_SDK_AVAILABLE and not groq_api_configured():
        logger.info("Groq ASR inactive: empty GROQ_API_KEY (fill _GROQ_API_KEY_LOCAL_TEST in this file)")
    # missing SDK: already logged once at import with sys.executable hint

if groq_phoneme_judge_ready():
    logger.info("Groq phoneme LLM judge: chat model=%s", GROQ_CHAT_MODEL)

g2p = G2p()

def _strip_ipa_stress(token: str) -> str:
    t = token.strip()
    while len(t) >= 1 and t[0] in "\u02c8\u02cc":
        t = t[1:]
    return t.strip()


def ipa_token_to_arpa_phones(token: str) -> List[str]:
    """Map one IPA tokenizer piece (possibly stressed) to ARPAbet-style phones."""
    base = _strip_ipa_stress(token)
    if not base or base == "|":
        return []
    return list(IPA_TO_ARPA.get(base, []))


def ctc_ids_to_heard_phonemes(predicted_ids: List[int], processor: Any) -> List[str]:
    """Collapse CTC duplicates and map tokenizer IDs → ARPAbet-style phoneme list."""
    collapsed = [k for k, _ in groupby(predicted_ids)]
    skip: Set[int] = set(processor.tokenizer.all_special_ids)
    pid = getattr(processor.tokenizer, "pad_token_id", None)
    if pid is not None:
        skip.add(int(pid))
    uidx = getattr(processor.tokenizer, "unk_token_id", None)
    if uidx is not None:
        skip.add(int(uidx))
    wdid = getattr(processor.tokenizer, "word_delimiter_token_id", None)
    if wdid is not None:
        skip.add(int(wdid))

    out: List[str] = []
    for tid in collapsed:
        if tid in skip:
            continue
        tok = processor.tokenizer.convert_ids_to_tokens(int(tid))
        out.extend(ipa_token_to_arpa_phones(str(tok)))
    return out


def extract_phonemes_from_bank(bank: Dict[str, Any]) -> List[str]:
    phonemes: List[str] = []
    if "consonants" in bank:
        for category in bank["consonants"].values():
            if isinstance(category, dict):
                phonemes.extend(category.keys())
    if "vowels" in bank:
        for category in bank["vowels"].values():
            if isinstance(category, dict):
                phonemes.extend(category.keys())
    if "diphthongs" in bank and isinstance(bank["diphthongs"], dict):
        phonemes.extend(bank["diphthongs"].keys())
    return phonemes


try:
    with open(BASE_DIR / "exercise_bank.json", "r", encoding="utf-8") as f:
        exercise_bank = json.load(f)
    if isinstance(exercise_bank, dict) and "metadata" in exercise_bank:
        PHONEMES = extract_phonemes_from_bank(exercise_bank)
        logger.info(f"Loaded {len(PHONEMES)} phonemes from enhanced exercise bank")
    else:
        PHONEMES = list(exercise_bank.keys()) if isinstance(exercise_bank, dict) else []
        logger.info(f"Loaded {len(PHONEMES)} phonemes from exercise bank (legacy format)")
except FileNotFoundError:
    logger.warning("exercise_bank.json not found, using empty phoneme list")
    PHONEMES = []
except Exception as e:
    logger.error(f"Error loading exercise bank: {e}")
    PHONEMES = []


def calculate_severity(a: Any, b: Any) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1 - SequenceMatcher(None, a, b).ratio()


def split_into_syllable_like_chunks(word: str) -> List[str]:
    """
    Very simple orthographic syllable approximation for therapy use.
    Example: "swabbit" -> ["swa", "bbi", "t"]
    Heuristic:
    - Treat vowels (including y) as syllable nuclei.
    - Build syllables as (onset consonants) + (vowel nucleus).
    - For consonant clusters between vowels, bias toward the next syllable onset
      (max-onset style), which helps cases like: swabbit -> swa + bbi + t.
    - Final consonant-only remainder becomes its own chunk (collapsing doubled
      letters like "tt" -> "t" for display).
    """
    w = word.lower()
    if not w:
        return []

    # Split into runs of vowels vs consonants
    runs: List[Tuple[str, str]] = []
    cur = w[0]
    cur_is_v = cur in VOWELS
    buf = cur
    for ch in w[1:]:
        is_v = ch in VOWELS
        if is_v == cur_is_v:
            buf += ch
        else:
            runs.append(("v" if cur_is_v else "c", buf))
            buf = ch
            cur_is_v = is_v
    runs.append(("v" if cur_is_v else "c", buf))

    syllables: List[str] = []
    i = 0
    while i < len(runs):
        onset = ""
        nucleus = ""

        if runs[i][0] == "c":
            onset = runs[i][1]
            i += 1

        if i < len(runs) and runs[i][0] == "v":
            nucleus = runs[i][1][:1]  # single vowel nucleus for display
            i += 1
        else:
            # No vowel left; remaining consonants are a final chunk
            final = onset
            if final:
                # collapse doubles for readability: "tt" -> "t"
                if len(final) >= 2 and all(ch == final[0] for ch in final):
                    final = final[0]
                syllables.append(final)
            break

        # Intervocalic consonant cluster (coda/onset). Prefer next onset.
        inter = ""
        if i < len(runs) and runs[i][0] == "c":
            inter = runs[i][1]
            i += 1

        # If there's another vowel ahead, move the consonant cluster to the next onset.
        if i < len(runs) and runs[i][0] == "v":
            syllables.append(onset + nucleus)
            # push back consonants as a new consonant run for next onset
            if inter:
                runs.insert(i, ("c", inter))
        else:
            # End of word; keep consonants as a final chunk
            syllables.append(onset + nucleus)
            if inter:
                final = inter
                if len(final) >= 2 and all(ch == final[0] for ch in final):
                    final = final[0]
                syllables.append(final)
            break

    return [s for s in syllables if s]


def build_syllable_analysis(
    expected_syllables: List[str], predicted_syllables: List[str]
) -> List[Dict[str, Any]]:
    """
    Compare expected vs predicted syllable-by-syllable (sound chunks).
    Returns list of { position, expected_syllable, predicted_syllable, match }.
    """
    analysis: List[Dict[str, Any]] = []
    for i, (exp_syl, pred_syl) in enumerate(
        zip_longest(expected_syllables, predicted_syllables, fillvalue="")
    ):
        analysis.append(
            {
                "position": i,
                "expected_syllable": exp_syl or "",
                "predicted_syllable": pred_syl or "",
                "match": (exp_syl or "") == (pred_syl or ""),
            }
        )
    return analysis


def phoneme_index_to_syllable_index(
    word: str, syllables: List[str], phoneme_index: int, total_phonemes: int
) -> int:
    """
    Map phoneme index to syllable index. Distributes phonemes evenly across
    syllable positions so each error can be tagged with expected_syllable.
    """
    if not syllables or total_phonemes <= 0:
        return 0
    n = len(syllables)
    # Even distribution: syllable k gets phonemes in [k*step, (k+1)*step)
    step = max(1, total_phonemes // n)
    k = min(phoneme_index // step, n - 1)
    return k


def annotate_errors_with_syllables(
    expected: str,
    predicted: str,
    expected_syllables: List[str],
    predicted_syllables: List[str],
    expected_ph: List[str],
    articulation_errors: List[Dict[str, Any]],
    taxonomy: List[Dict[str, Any]],
    tongue_analysis: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Add expected_syllable / produceded_syllable / syllable_position to each error."""
    n_exp_ph = len(expected_ph)

    def syl_for_ph_idx(ph_idx: int) -> int:
        return phoneme_index_to_syllable_index(
            expected, expected_syllables, ph_idx, n_exp_ph
        )

    def produced_syl(k: int) -> str:
        return predicted_syllables[k] if 0 <= k < len(predicted_syllables) else ""

    out_art: List[Dict[str, Any]] = []
    for err in articulation_errors:
        e = err.copy()
        # Ensure `details` always exists
        if not isinstance(e.get("details"), dict):
            e["details"] = {}
        idx = err.get("expected_index")
        if idx is None:
            exp_p = err.get("expected")
            if exp_p and expected_ph:
                try:
                    idx = next(i for i, p in enumerate(expected_ph) if p == exp_p)
                except StopIteration:
                    idx = 0
        if idx is not None and 0 <= idx < n_exp_ph:
            k = syl_for_ph_idx(idx)
            e["syllable_position"] = k
            e["expected_syllable"] = expected_syllables[k] if k < len(expected_syllables) else ""
            e["produceded_syllable"] = produced_syl(k)
        else:
            e.setdefault("expected_syllable", "")
            e.setdefault("produceded_syllable", "")
        out_art.append(e)

    out_tax: List[Dict[str, Any]] = []
    for t in taxonomy:
        e = t.copy()
        if not isinstance(e.get("details"), dict):
            e["details"] = {}
        idx = t.get("expected_index")
        if idx is not None and 0 <= idx < n_exp_ph:
            k = syl_for_ph_idx(idx)
            e["syllable_position"] = k
            e["expected_syllable"] = expected_syllables[k] if k < len(expected_syllables) else ""
            e["produceded_syllable"] = produced_syl(k)
        else:
            e.setdefault("expected_syllable", "")
            e.setdefault("produceded_syllable", "")
        out_tax.append(e)

    out_tongue: List[Dict[str, Any]] = []
    for h in tongue_analysis:
        e = h.copy()
        if not isinstance(e.get("details"), dict):
            e["details"] = {}
        # tongue_analysis doesn't have index; match by expected_phoneme position in expected_ph
        exp_ph = e.get("expected_phoneme")
        if exp_ph and expected_ph:
            try:
                idx = expected_ph.index(exp_ph)
            except ValueError:
                idx = 0
            k = syl_for_ph_idx(idx)
            e["syllable_position"] = k
            e["expected_syllable"] = expected_syllables[k] if k < len(expected_syllables) else ""
            e["produceded_syllable"] = produced_syl(k)
        else:
            e.setdefault("expected_syllable", "")
            e.setdefault("produceded_syllable", "")
        out_tongue.append(e)

    return out_art, out_tax, out_tongue


def detect_soda(expected: str, predicted: str) -> str:
    if expected == predicted:
        return "Correct"
    matcher = SequenceMatcher(None, expected, predicted)
    ops = [tag for tag, *_ in matcher.get_opcodes() if tag != "equal"]
    if not ops:
        return "Correct"
    if all(op == "replace" for op in ops):
        return "Substitution"
    if all(op == "delete" for op in ops):
        return "Omission"
    if all(op == "insert" for op in ops):
        return "Addition"
    return "Mixed"


def detect_soda_phoneme_lists(expected_ph: List[str], predicted_ph: List[str]) -> str:
    """SODA category from phoneme-token alignment (Substitution / Omission / Addition / Mixed)."""
    ec, pc = canonical_phoneme_seq(expected_ph), canonical_phoneme_seq(predicted_ph)
    if ec == pc:
        return "Correct"
    matcher = SequenceMatcher(None, ec, pc)
    ops = [tag for tag, *_ in matcher.get_opcodes() if tag != "equal"]
    if not ops:
        return "Correct"
    if all(op == "replace" for op in ops):
        return "Substitution"
    if all(op == "delete" for op in ops):
        return "Omission"
    if all(op == "insert" for op in ops):
        return "Addition"
    return "Mixed"


def first_mismatched_expected_phone(
    expected_ph: List[str], predicted_ph: List[str]
) -> Optional[str]:
    matcher = SequenceMatcher(
        None,
        canonical_phoneme_seq(expected_ph),
        canonical_phoneme_seq(predicted_ph),
    )
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete") and i2 > i1:
            return expected_ph[i1]
    return None


def phoneme_chunks_from_arpa(phones: List[str]) -> List[str]:
    """Rough syllable-like chunks from ARPAbet phones (vowel nucleus)."""
    if not phones:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    for p in phones:
        cur.append(p)
        if p in ARPA_VOWELS:
            chunks.append("-".join(cur))
            cur = []
    if cur:
        tail = "-".join(cur)
        if chunks:
            chunks[-1] = f"{chunks[-1]}-{tail}"
        else:
            chunks.append(tail)
    return chunks or ["-".join(phones)]


def phoneme_position(word: str, phoneme: str) -> Optional[str]:
    idx = word.find(phoneme)
    if idx == -1:
        return None
    if idx == 0:
        return "initial"
    if idx + len(phoneme) == len(word):
        return "final"
    return "medial"


def get_mismatched_phoneme(expected: str, predicted: str) -> Optional[str]:
    matcher = SequenceMatcher(None, expected, predicted)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("replace", "delete", "insert"):
            segment = expected[i1:i2]
            for p in sorted(PHONEMES, key=len, reverse=True):
                if segment.startswith(p):
                    return p
    return None


def build_cv(word: str, base: str) -> str:
    idx = word.find(base)
    if idx != -1 and idx + len(base) < len(word):
        nxt = word[idx + len(base)]
        if nxt in VOWELS:
            return base + nxt
    return base


def therapy_level(sev: float) -> str:
    if sev >= 0.75:
        return "high"
    if sev >= 0.4:
        return "medium"
    return "low"


def clean_phonemes(lst: List[Any]) -> List[str]:
    """Normalize ARPAbet tokens from g2p_en (strip stress digits, lowercase)."""
    out: List[str] = []
    for p in lst:
        if str(p).isspace():
            continue
        s = "".join(c for c in str(p).lower() if c.isalpha())
        if s:
            out.append(s)
    return out


@lru_cache(maxsize=2000)
def g2p_cached(word: str) -> List[str]:
    return clean_phonemes(g2p(word))


def word_to_phonemes(word: str) -> List[str]:
    return clean_phonemes(g2p(word))


def nw_align_backtrace(
    a: List[str], b: List[str]
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Global alignment (Levenshtein) backtrace: each column is (token_a | None, token_b | None).
    Prefers diagonal matches when costs tie so merge stays stable.
    """
    na, nb = len(a), len(b)
    if na == 0:
        return [(None, b[j]) for j in range(nb)]
    if nb == 0:
        return [(a[i], None) for i in range(na)]

    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        dp[i][0] = i
    for j in range(1, nb + 1):
        dp[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = na, nb
    cols_rev: List[Tuple[Optional[str], Optional[str]]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            diag = dp[i - 1][j - 1] + cost
            if dp[i][j] == diag:
                cols_rev.append((a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        up_cost = dp[i - 1][j] + 1 if i > 0 else 10**9
        left_cost = dp[i][j - 1] + 1 if j > 0 else 10**9
        if i > 0 and dp[i][j] == up_cost and up_cost <= left_cost:
            cols_rev.append((a[i - 1], None))
            i -= 1
        elif j > 0:
            cols_rev.append((None, b[j - 1]))
            j -= 1
        elif i > 0:
            cols_rev.append((a[i - 1], None))
            i -= 1
        else:
            break
    cols_rev.reverse()
    return cols_rev


def merge_alignment_column(c: Optional[str], l: Optional[str]) -> str:
    """Resolve one alignment column from (phoneme_ctc, word_g2p_phoneme)."""
    if c and not l:
        return c
    if l and not c:
        return l
    if not c and not l:
        return ""
    assert c is not None and l is not None
    if c == l:
        return c
    if c in ARPA_VOWELS and l in ARPA_VOWELS:
        return l
    if c not in ARPA_VOWELS and l not in ARPA_VOWELS:
        return c
    if c not in ARPA_VOWELS:
        return c
    return l


def fuse_phoneme_ctc_and_word(ctc_ph: List[str], asr_word: str) -> List[str]:
    """
    Merge acoustic phoneme CTC with phonemes from word-level ASR + G2P.

    Streams are globally aligned; on conflict: vowel–vowel → word/G2P;
    consonant–consonant → CTC; mixed → keep consonant stream when c is cons else l.
    """
    w = "".join(ch for ch in asr_word.lower().strip() if ch.isalpha())
    if not w:
        return list(ctc_ph)
    lex_ph = word_to_phonemes(w)
    if not lex_ph:
        return list(ctc_ph)
    if not ctc_ph:
        return list(lex_ph)
    cols = nw_align_backtrace(ctc_ph, lex_ph)
    merged: List[str] = []
    for aa, bb in cols:
        tok = merge_alignment_column(aa, bb)
        if tok:
            merged.append(tok)
    return merged


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def summarize_audio_research_features(audio: torch.Tensor) -> Dict[str, float]:
    a = audio.detach().float().flatten()
    n = int(a.numel())
    if n <= 0:
        return {
            "duration_sec": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "zcr": 0.0,
            "rms_db": float("-inf"),
        }
    duration = n / float(SAMPLE_RATE)
    rms = torch.sqrt(torch.mean(a**2) + 1e-12).item()
    peak = torch.max(torch.abs(a)).item()
    if n > 1:
        zcr = torch.mean((a[:-1] * a[1:] < 0).float()).item()
    else:
        zcr = 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    return {
        "duration_sec": round(duration, 3),
        "rms": round(float(rms), 6),
        "peak": round(float(peak), 6),
        "zcr": round(float(zcr), 6),
        "rms_db": round(float(rms_db), 3),
    }


def detect_phonological_processes(
    expected_ph: List[str], predicted_ph: List[str]
) -> List[Dict[str, Any]]:
    """
    Heuristic process labels from expected vs. ASR-derived phoneme sequences.
    Intended for research exploration (not ground-truth).
    """
    processes: List[Dict[str, Any]] = []

    if phoneme_sequences_equivalent(expected_ph, predicted_ph):
        return processes

    # Deletions
    if len(predicted_ph) < len(expected_ph) and expected_ph:
        if predicted_ph == expected_ph[: len(predicted_ph)]:
            last = expected_ph[-1]
            if last in PHONEME_FEATURES:
                processes.append(
                    {
                        "process": "final_consonant_deletion",
                        "evidence": {"expected_tail": last, "predicted_tail": None},
                    }
                )

    # Cluster reduction (very rough): consecutive consonants in expected + overall deletion.
    exp_cons_idx = [
        i for i, p in enumerate(expected_ph) if p in PHONEME_FEATURES  # consonant set here
    ]
    for i in range(len(exp_cons_idx) - 1):
        a_i = exp_cons_idx[i]
        b_i = exp_cons_idx[i + 1]
        if b_i == a_i + 1 and len(predicted_ph) < len(expected_ph):
            processes.append(
                {
                    "process": "cluster_reduction",
                    "evidence": {
                        "expected_cluster": expected_ph[a_i : b_i + 1],
                        "predicted_len": len(predicted_ph),
                        "expected_len": len(expected_ph),
                    },
                }
            )
            break

    # Substitution-style processes via feature diffs.
    for idx, (e, p) in enumerate(zip_longest(expected_ph, predicted_ph, fillvalue=None)):
        if e is None or p is None or phonemes_symbol_match(str(e), str(p)):
            continue
        if e not in PHONEME_FEATURES or p not in PHONEME_FEATURES:
            continue
        ef = PHONEME_FEATURES[e]
        pf = PHONEME_FEATURES[p]

        def _add(proc: str) -> None:
            lib = PROCESS_LIBRARY.get(proc, {})
            processes.append(
                {
                    "process": proc,
                    "label": lib.get("label", proc.replace("_", " ").title()),
                    "description": lib.get("description"),
                    "therapy_focus": lib.get("therapy_focus"),
                    "evidence": {"index": idx, "expected": e, "produced": p},
                }
            )

        if e in {"r", "l"} and p in {"w", "y"}:
            _add("gliding")

        if ef.get("manner") in {"fricative", "affricate"} and pf.get("manner") == "stop":
            _add("stopping")

        if ef.get("place") == "velar" and pf.get("place") == "alveolar":
            _add("fronting")

        if ef.get("place") == "alveolar" and pf.get("place") == "velar":
            _add("backing")

        if ef.get("voiced") is True and pf.get("voiced") is False:
            _add("devoicing")
        if ef.get("voiced") is False and pf.get("voiced") is True:
            _add("voicing")

        # Common additional patterns
        if e == "ch" and p in {"sh", "t", "s"}:
            _add("deaffrication")
        if e == "s" and p == "sh":
            _add("palatalization")
        if e == "sh" and p == "s":
            _add("depalatalization")

    # De-duplicate
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for pr in processes:
        key = (str(pr.get("process")), json.dumps(pr.get("evidence", {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pr)
    return deduped


def build_tongue_hypotheses(
    expected_ph: List[str],
    predicted_ph: List[str],
    asr_confidence: float,
) -> List[Dict[str, Any]]:
    hypotheses: List[Dict[str, Any]] = []
    base_conf = max(0.0, min(float(asr_confidence), 1.0))
    for idx, (e, p) in enumerate(zip_longest(expected_ph, predicted_ph, fillvalue=None)):
        if e is None or p is None or phonemes_symbol_match(str(e), str(p)):
            continue
        if e not in PHONEME_FEATURES or p not in PHONEME_FEATURES:
            continue
        ef = PHONEME_FEATURES[e]
        pf = PHONEME_FEATURES[p]
        e_pos = ef.get("tongue")
        p_pos = pf.get("tongue")
        tip = TONGUE_FEEDBACK.get(e_pos)
        confidence = base_conf
        if ef.get("place") != pf.get("place"):
            confidence = min(1.0, confidence + 0.1)
        hypotheses.append(
            {
                "index": idx,
                "target_phoneme": e,
                "produced_phoneme": p,
                "target_place": ef.get("place"),
                "produced_place": pf.get("place"),
                "target_manner": ef.get("manner"),
                "produced_manner": pf.get("manner"),
                "target_voiced": ef.get("voiced"),
                "produced_voiced": pf.get("voiced"),
                "target_tongue_position": e_pos,
                "produced_tongue_position": p_pos,
                "suggested_tip": tip,
                "confidence": round(confidence, 3),
            }
        )
    return hypotheses


def format_therapist_recommendations(
    *,
    processes: List[Dict[str, Any]],
    tongue_hypotheses: List[Dict[str, Any]],
    articulation: Optional[List[Dict[str, Any]]] = None,
    max_items: int = 6,
) -> List[str]:
    recs: List[str] = []
    for pr in processes:
        label = pr.get("label") or str(pr.get("process") or "").replace("_", " ").title()
        focus = pr.get("therapy_focus")
        if label and focus:
            recs.append(f"{label}: {focus}")
        elif label:
            recs.append(f"Pattern to watch: {label}.")
    if articulation:
        for err in articulation:
            e = err.get("expected")
            p = err.get("produced")
            details = err.get("details") or {}
            if e and p and isinstance(details, dict):
                summary = ", ".join(str(v) for v in details.values() if v)
                if summary:
                    recs.append(f"Articulation cue for /{e}/ vs /{p}/: {summary}.")
    for h in tongue_hypotheses:
        target = h.get("target_phoneme")
        tip = h.get("suggested_tip")
        if target and tip:
            recs.append(f"Target /{target}/: {tip}")
    # De-duplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for r in recs:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
        if len(out) >= max_items:
            break
    return out


def simplify_tongue_hypotheses(
    tongue_hypotheses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    simplified: List[Dict[str, Any]] = []
    for h in tongue_hypotheses:
        simplified.append(
            {
                "target_phoneme": h.get("target_phoneme"),
                "produced_phoneme": h.get("produced_phoneme"),
                "target_tongue_position": h.get("target_tongue_position"),
                # "suggested_tip": h.get("suggested_tip"),
                "confidence": h.get("confidence"),
            }
        )
    return simplified


def _parse_bool_arg(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _analyze_error_response(
    message: str,
    *,
    details: Optional[str] = None,
    status_code: int = 400,
    include_meta: bool = False,
) -> Tuple[Any, int]:
    payload: Dict[str, Any] = {"error": {"message": message}}
    if details is not None:
        payload["error"]["details"] = details
    if include_meta:
        payload["meta"] = {}
    return jsonify(payload), status_code


def _minimal_taxonomy_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in rows:
        out.append(
            {
                "category": t.get("category"),
                "expected": t.get("expected"),
                "produced": t.get("produced"),
            }
        )
    return out


def _minimal_articulation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed_detail = {"manner_error", "place_error", "voicing_error"}
    for a in rows:
        raw_details = (a.get("details") or {}) if isinstance(a.get("details"), dict) else {}
        slim_details = {
            k: raw_details[k] for k in allowed_detail if k in raw_details
        }
        out.append(
            {
                "expected": a.get("expected"),
                "produced": a.get("produced"),
                "details": slim_details,
            }
        )
    return out


def wrong_lexical_word(expected: str, predicted_word: str) -> bool:
    """True when word-level ASR transcript exists and differs from the target word (case-insensitive)."""
    pw = (predicted_word or "").strip().lower()
    if not pw:
        return False
    return pw != (expected or "").strip().lower()


def _minimal_tongue_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = (
        "expected_phoneme",
        "produced_phoneme",
        "expected_tongue_position",
        "produced_tongue_position",
    )
    out: List[Dict[str, Any]] = []
    for t in rows:
        if isinstance(t, dict):
            out.append({k: t.get(k) for k in keys})
    return out


def build_minimal_analyze_payload(
    *,
    expected: str,
    basic_results: Dict[str, Any],
    sev_acoustic: float,
    merged_combo: Dict[str, Any],
    taxonomy: List[Dict[str, Any]],
    articulation: List[Dict[str, Any]],
    tongue: List[Dict[str, Any]],
    expected_ph: List[str],
    predicted_ph: List[str],
    predicted_word: str,
    whisper_word: str = "",
    whisper_phonemes_g2p: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Thin client-facing JSON for GET/POST `/analyze`: severities + SODA + articulation/tongue + word."""
    ww = wrong_lexical_word(expected, predicted_word)
    # Phoneme-level correctness from lenient alignment; force false if word ASR says a different word.
    is_correct_out = bool(basic_results["is_correct"]) and not ww
    wg = list(whisper_phonemes_g2p or [])

    return {
        "is_correct": is_correct_out,
        "wrong_word": ww,
        # Overall severity (0..1, higher = worse) computed by `composite_severity`.
        "severity": round(float(sev_acoustic), 3),
        "severity_phoneme": basic_results["severity_phoneme"],
        "severity_text": basic_results["severity_text"],
        "severity_phoneme_strict": basic_results["severity_phoneme_strict"],
        "acoustic_severity": round(float(sev_acoustic), 3),
        "pronunciation_quality_combined": merged_combo["pronunciation_quality_combined"],
        "severity_phoneme_combined": merged_combo["severity_phoneme_combined"],
        "severity_phoneme_strict_combined": merged_combo["severity_phoneme_strict_combined"],
        "severity_text_combined": merged_combo["severity_text_combined"],
        "soda_errors": {
            "base_soda_error": soda_error_type(expected_ph, predicted_ph),
            "primary_error_type": basic_results["error_type"],
            "taxonomy": _minimal_taxonomy_rows(taxonomy),
        },
        "articulation_errors": _minimal_articulation_rows(articulation),
        "tongue_position_analysis": _minimal_tongue_rows(tongue),
        "predicted_word": predicted_word or "",
        "whisper_word": (whisper_word or "").strip(),
        "whisper_phonemes_g2p": wg,
    }


def articulation_analysis(expected_ph: List[str], predicted_ph: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for e, p in zip_longest(expected_ph, predicted_ph, fillvalue=None):
        if e is None or p is None or phonemes_symbol_match(str(e), str(p)):
            continue
        if e not in PHONEME_FEATURES or p not in PHONEME_FEATURES:
            continue
        ef = PHONEME_FEATURES[e]
        pf = PHONEME_FEATURES[p]
        errors: Dict[str, str] = {}
        if ef["place"] != pf["place"]:
            errors["place_error"] = f"{ef['place']} → {pf['place']}"
        if ef["manner"] != pf["manner"]:
            errors["manner_error"] = f"{ef['manner']} → {pf['manner']}"
        if ef.get("voiced") != pf.get("voiced"):
            errors["voicing_error"] = (
                f"{'voiced' if ef.get('voiced') else 'voiceless'} → "
                f"{'voiced' if pf.get('voiced') else 'voiceless'}"
            )
        if errors:
            results.append({"expected": e, "produced": p, "details": errors})
    return results


def tongue_feedback(expected_ph: List[str], predicted_ph: List[str]) -> List[Dict[str, Any]]:
    feedback: List[Dict[str, Any]] = []
    for e, p in zip_longest(expected_ph, predicted_ph, fillvalue=None):
        if e is None or p is None or phonemes_symbol_match(str(e), str(p)):
            continue
        if e not in PHONEME_FEATURES or p not in PHONEME_FEATURES:
            continue
        e_pos = PHONEME_FEATURES[e]["tongue"]
        p_pos = PHONEME_FEATURES[p]["tongue"]
        tip = TONGUE_FEEDBACK.get(e_pos)
        feedback.append(
            {
                "expected_phoneme": e,
                "produced_phoneme": p,
                "expected_tongue_position": e_pos,
                "produced_tongue_position": p_pos,
                # "therapy_tip": tip,
            }
        )
    return feedback


def soda_error_type(expected_ph: List[str], predicted_ph: List[str]) -> str:
    if phoneme_sequences_equivalent(expected_ph, predicted_ph):
        return "correct"
    if len(predicted_ph) < len(expected_ph):
        return "omission"
    if len(predicted_ph) > len(expected_ph):
        return "addition"
    return "substitution"


def error_taxonomy_details(e: str, p: str) -> Dict[str, Optional[str]]:
    """
    Returns therapist-friendly taxonomy info for a single expected→produced pair.
    """
    category = "substitution"
    if e in {"r", "l"} and p in {"w", "y"}:
        category = "gliding"
    elif e in {"s", "z", "sh", "zh", "ch", "jh", "f", "v", "th", "dh"} and p in {"t", "d", "k", "g"}:
        category = "stopping"
    elif e in {"k", "g", "ng"} and p in {"t", "d", "n"}:
        category = "fronting"
    elif e in {"t", "d", "n"} and p in {"k", "g", "ng"}:
        category = "backing"
    elif e in {"p", "t", "k", "s", "f", "th", "sh", "ch"} and p in {"b", "d", "g", "z", "v", "dh", "zh", "jh"}:
        category = "voicing"
    elif e in {"b", "d", "g", "z", "v", "dh", "zh", "jh"} and p in {"p", "t", "k", "s", "f", "th", "sh", "ch"}:
        category = "devoicing"
    elif e == "ch" and p in {"sh", "s", "t"}:
        category = "deaffrication"
    elif e == "s" and p == "sh":
        category = "palatalization"
    elif e == "sh" and p == "s":
        category = "depalatalization"

    lib = PROCESS_LIBRARY.get(category, {})
    return {
        "category": category,
        # "label": lib.get("label"),
        # "description": lib.get("description"),
        # "therapy_focus": lib.get("therapy_focus"),
    }


def build_error_taxonomy_from_alignment(
    expected_ph: List[str], predicted_ph: List[str]
) -> List[Dict[str, Any]]:
    """
    Build taxonomy using aligned phoneme sequences (not simple positional zip),
    so categories like gliding/stopping/fronting show up more reliably.
    """
    taxonomy: List[Dict[str, Any]] = []
    sm = SequenceMatcher(
        None,
        canonical_phoneme_seq(expected_ph),
        canonical_phoneme_seq(predicted_ph),
    )
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        exp_seg = expected_ph[i1:i2]
        pred_seg = predicted_ph[j1:j2]

        if tag == "replace":
            for off, (e, p) in enumerate(zip_longest(exp_seg, pred_seg, fillvalue=None)):
                if e is None or p is None:
                    continue
                if phonemes_symbol_match(str(e), str(p)):
                    continue
                item: Dict[str, Any] = {
                    "expected": e,
                    "produced": p,
                    # "expected_index": i1 + off,
                    # "produced_index": j1 + off,
                }
                item.update(error_taxonomy_details(e, p))
                taxonomy.append(item)
        elif tag == "delete":
            for off, e in enumerate(exp_seg):
                taxonomy.append(
                    {
                        "expected": e,
                        "produced": None,
                        "expected_index": i1 + off,
                        "produced_index": None,
                        "category": "omission",
                        "label": "Omission",
                        # "description": "A target sound may be omitted.",
                        # "therapy_focus": "Try slowing the word and adding the missing sound with clear models and tactile/visual cues.",
                    }
                )
        elif tag == "insert":
            for off, p in enumerate(pred_seg):
                taxonomy.append(
                    {
                        "expected": None,
                        "produced": p,
                        "expected_index": None,
                        "produced_index": j1 + off,
                        "category": "addition",
                        "label": "Addition",
                        "description": "An extra sound may be inserted.",
                        "therapy_focus": "Work on smooth transitions between sounds and pacing; practice the target word slowly then increase rate.",
                    }
                )
    return taxonomy


def error_taxonomy(e: str, p: str) -> str:
    return str(error_taxonomy_details(e, p).get("category") or "substitution")


def acoustic_confidence(audio: torch.Tensor) -> float:
    if audio.numel() == 0:
        return 0.0
    energy = torch.mean(audio**2).item()
    variance = torch.var(audio).item()
    return min(energy + variance, 1.0)


def get_asr_confidence(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1).values
    return torch.mean(max_probs).item()


def composite_severity(
    expected: str,
    predicted: str,
    expected_ph: List[str],
    predicted_ph: List[str],
    audio: torch.Tensor,
    asr_confidence: float,
) -> float:
    text_score = 1 - calculate_severity(expected, predicted)
    phoneme_score = 1 - phoneme_sequence_severity(expected_ph, predicted_ph)
    acoustic_score = acoustic_confidence(audio)
    score = (
        0.35 * text_score
        + 0.25 * phoneme_score
        + 0.20 * acoustic_score
        + 0.20 * asr_confidence
    )
    return 1 - score


def composite_severity_app_py(
    expected: str,
    predicted: str,
    expected_ph: List[str],
    predicted_ph: List[str],
    audio: torch.Tensor,
) -> float:
    """
    Composite severity matching `app.py`'s advanced endpoint.
    (Different weighting vs. the optimized pipeline.)
    """
    text_score = 1 - calculate_severity(expected, predicted)
    phoneme_score = 1 - phoneme_sequence_severity(expected_ph, predicted_ph)
    acoustic_score = acoustic_confidence(audio)
    score = 0.4 * text_score + 0.3 * phoneme_score + 0.3 * acoustic_score
    return 1 - score


def convert_to_wav_16k_mono(input_path: Path, output_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "wav",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("Audio conversion timed out")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Audio conversion failed: {e}")


def denoise_audio_for_pipeline(audio: Any, sr: int) -> np.ndarray:
    """
    Reduce background noise before phoneme ASR / acoustic features.

    Uses ``noisereduce`` (non-stationary profile) when installed; otherwise a
    mild high-pass to remove rumble and DC-ish bias.
    """
    if not AUDIO_DENOISE:
        return audio
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    if _NOISEREDUCE_AVAILABLE and nr is not None:
        try:
            return np.asarray(
                nr.reduce_noise(y=x, sr=sr, stationary=False, prop_decrease=0.75),
                dtype=np.float32,
            )
        except Exception as e:
            logger.warning("noisereduce failed (%s); using high-pass fallback", e)
    w = torch.from_numpy(x.copy()).unsqueeze(0)
    w = torchaudio.functional.highpass_biquad(w, sr, cutoff_freq=80.0)
    return w.squeeze().numpy().astype(np.float32)


def load_audio_tensor(wav_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = resampler(wav)
    x = wav.squeeze()
    den = denoise_audio_for_pipeline(x.numpy(), SAMPLE_RATE)
    return torch.from_numpy(den)


def analyze_asr_phoneme_ctc(wav_path: Path) -> Tuple[List[str], float]:
    """
    Phoneme recognizer (CTC): returns heard phones in ARPAbet-style tokens + confidence.
    """
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = resampler(wav)
    max_samples = SAMPLE_RATE * MAX_AUDIO_DURATION_SEC
    wav = wav[:, :max_samples]
    audio = denoise_audio_for_pipeline(wav.squeeze().numpy(), SAMPLE_RATE)

    inputs = phoneme_processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device=device, dtype=_asr_dtype)
    with torch.no_grad():
        logits = phoneme_model(input_values).logits
    confidence = get_asr_confidence(logits)
    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    heard_ph = ctc_ids_to_heard_phonemes(predicted_ids, phoneme_processor)
    return heard_ph, confidence


def analyze_asr_word_ctc(wav_path: Path) -> Tuple[str, float]:
    """Word-level Wav2Vec2 CTC: orthographic transcript + mean frame confidence."""
    if not word_asr_ready():
        return "", 0.0
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    max_samples = SAMPLE_RATE * MAX_AUDIO_DURATION_SEC
    wav = wav[:, :max_samples]
    audio = denoise_audio_for_pipeline(wav.squeeze().numpy(), SAMPLE_RATE)
    inputs = word_processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device=device, dtype=_asr_dtype)
    with torch.no_grad():
        logits = word_model(input_values).logits
    confidence = get_asr_confidence(logits)
    pred_ids = torch.argmax(logits, dim=-1)
    text = word_processor.batch_decode(pred_ids)[0].lower().strip()
    text = "".join(c for c in text if c.isalpha() or c.isspace()).strip()
    if text:
        text = text.split()[0]
    return text, confidence


def _groq_transcription_to_dict(transcription: Any) -> Dict[str, Any]:
    if isinstance(transcription, str):
        return {"text": transcription}
    if hasattr(transcription, "model_dump"):
        return transcription.model_dump()
    if hasattr(transcription, "dict"):
        return transcription.dict()
    if isinstance(transcription, dict):
        return transcription
    return {"text": getattr(transcription, "text", str(transcription))}


def _groq_format_api_error(exc: BaseException) -> str:
    chunks: List[str] = [f"{type(exc).__name__}: {exc}"]
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code is not None:
            chunks.append(f"http_status={code}")
        txt = getattr(resp, "text", "") or ""
        if txt.strip():
            chunks.append(f"body={txt.strip()[:1500]}")
    body_attr = getattr(exc, "body", None)
    if body_attr is not None and str(body_attr).strip():
        chunks.append(f"exc.body={str(body_attr)[:800]}")
    return " | ".join(chunks)


def _groq_whisper_models_to_try() -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for m in (
        GROQ_WHISPER_MODEL,
        GROQ_WHISPER_FALLBACK_MODEL if GROQ_WHISPER_FALLBACK_MODEL != GROQ_WHISPER_MODEL else "",
    ):
        mt = (m or "").strip()
        if mt and mt not in seen:
            seen.add(mt)
            out.append(mt)
    return out if out else [GROQ_WHISPER_MODEL]


def groq_confidence_from_payload(payload: Dict[str, Any]) -> float:
    """Map verbose_json cues to [0.05, 1.0] when possible."""
    words = payload.get("words")
    if isinstance(words, list):
        probs: List[float] = []
        for w in words:
            if isinstance(w, dict) and w.get("probability") is not None:
                try:
                    probs.append(float(w["probability"]))
                except (TypeError, ValueError):
                    continue
        if probs:
            return float(max(0.05, min(1.0, sum(probs) / len(probs))))
    segs = payload.get("segments")
    if isinstance(segs, list):
        logps: List[float] = []
        for s in segs:
            if isinstance(s, dict) and s.get("avg_logprob") is not None:
                try:
                    logps.append(float(s["avg_logprob"]))
                except (TypeError, ValueError):
                    continue
        if logps:
            x = sum(logps) / len(logps)
            return float(max(0.05, min(1.0, 1.0 + x)))
    return 0.88


def groq_transcribe_verbose_file(audio_path: Path) -> Dict[str, Any]:
    """
    Robust Groq Whisper call: multipart upload uses (filename, BytesIO) — some stacks
    choke on unnamed file handles. Retries downgrade timestamps / verbosity / Whisper model.
    """
    if not groq_asr_ready() or GroqClient is None:
        raise RuntimeError("Groq ASR not configured")
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is empty")
    path = audio_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Groq Whisper: empty file {path}")

    fname = path.name if path.name else "audio.wav"

    transcription_attempts: List[Tuple[str, Dict[str, Any]]] = []
    for model_name in _groq_whisper_models_to_try():
        transcription_attempts.extend(
            [
                (model_name, {"response_format": "verbose_json", "language": "en", "timestamp_granularities": ["word", "segment"]}),
                (model_name, {"response_format": "verbose_json", "language": "en"}),
                (model_name, {"response_format": "verbose_json"}),
                (model_name, {"response_format": "json", "language": "en"}),
                (model_name, {"response_format": "json"}),
                (model_name, {"response_format": "text", "language": "en"}),
            ]
        )

    client = GroqClient(api_key=api_key)
    last_exc: Optional[BaseException] = None
    for mid, extras in transcription_attempts:
        try:
            bio = BytesIO(raw)
            file_arg = (fname, bio)
            transcription = client.audio.transcriptions.create(
                file=file_arg,
                model=mid,
                temperature=0.0,
                **extras,
            )
            return _groq_transcription_to_dict(transcription)
        except Exception as e:
            last_exc = e
            logger.warning(
                "Groq Whisper failed model=%s params=%s | %s",
                mid,
                extras,
                _groq_format_api_error(e),
            )
            continue

    last_msg = _groq_format_api_error(last_exc) if last_exc is not None else "unknown_error"
    raise RuntimeError(
        f"Groq Whisper transcription failed after {len(transcription_attempts)} attempts; last={last_msg}"
    ) from last_exc


def analyze_asr_groq_word(wav_path: Path) -> Tuple[str, float]:
    """
    Groq Whisper → graphemic text (first alphabetic word) + rough confidence.

    Phoneme path uses g2p_en on that word elsewhere (same as other word ASRs).
    """
    if not groq_asr_ready():
        return "", 0.0
    payload = groq_transcribe_verbose_file(wav_path)
    text = (payload.get("text") or "").strip().lower()
    text = "".join(c for c in text if c.isalpha() or c.isspace()).strip()
    if text:
        text = text.split()[0]
    conf = groq_confidence_from_payload(payload)
    return text, conf



def _groq_parse_json_message_content(content: str) -> Dict[str, Any]:
    t = (content or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
        t = t.strip()
    return json.loads(t)


def _clamp_unit(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def _validate_llm_phoneme_list(xs: Any, fallback: List[str]) -> List[str]:
    if not isinstance(xs, list):
        return list(fallback)
    out = clean_phonemes(xs)
    return out if out else list(fallback)


def _validate_llm_syllable_analysis(
    xs: Any, fallback: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not isinstance(xs, list):
        return list(fallback)
    if len(xs) == 0:
        return list(fallback)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(xs):
        if not isinstance(row, dict):
            return list(fallback)
        if not all(k in row for k in ("expected_syllable", "match", "position", "predicted_syllable")):
            return list(fallback)
        try:
            pos = int(row["position"])
        except (TypeError, ValueError):
            pos = i
        out.append(
            {
                "expected_syllable": str(row.get("expected_syllable", "")),
                "match": bool(row.get("match", False)),
                "position": pos,
                "predicted_syllable": str(row.get("predicted_syllable", "")),
            }
        )
    return out if out else list(fallback)


def _validate_llm_tongue_analysis(
    xs: Any, fallback: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not isinstance(xs, list):
        return list(fallback)
    if len(xs) == 0:
        return list(fallback)
    out: List[Dict[str, Any]] = []
    for row in xs:
        if not isinstance(row, dict):
            return list(fallback)
        need = (
            "expected_phoneme",
            "produced_phoneme",
            "expected_tongue_position",
            "produced_tongue_position",
        )
        if not all(k in row for k in need):
            return list(fallback)
        out.append(
            {
                "expected_phoneme": str(row["expected_phoneme"]),
                "produced_phoneme": str(row["produced_phoneme"]),
                "expected_tongue_position": str(row["expected_tongue_position"]),
                "produced_tongue_position": str(row["produced_tongue_position"]),
            }
        )
    return out


def merge_pipeline_and_llm_assessment(
    *,
    api_severity_phoneme: float,
    api_severity_strict: float,
    api_severity_text: float,
    heard_word_g2p: List[str],
    heard_fused: List[str],
    syllable_pipeline: List[Dict[str, Any]],
    tongue_pipeline: List[Dict[str, Any]],
    llm_json: Optional[Dict[str, Any]],
    groq_ok: bool,
    w_api: float = GROQ_COMBINED_WEIGHT_API,
    w_llm: float = GROQ_COMBINED_WEIGHT_LLM,
) -> Dict[str, Any]:
    """
    Blend automatic metrics with Llama JSON. When Groq is unavailable, echoes pipeline-only values.
    """
    w_tot = max(1e-9, w_api + w_llm)
    wa, wl = w_api / w_tot, w_llm / w_tot

    pq_api = max(0.0, min(1.0, 1.0 - float(api_severity_phoneme)))

    base: Dict[str, Any] = {
        "pronunciation_quality_combined": round(pq_api, 3),
        "severity_phoneme_combined": round(float(api_severity_phoneme), 2),
        "severity_phoneme_strict_combined": round(float(api_severity_strict), 2),
        "severity_text_combined": round(float(api_severity_text), 2),
        "therapy_level_combined": therapy_level(float(api_severity_phoneme)),
        "heard_phonemes_word_g2p_combined": list(heard_word_g2p),
        "heard_phonemes_fused_combined": list(heard_fused),
        "syllable_analysis_combined": list(syllable_pipeline),
        "tongue_position_analysis_combined": list(tongue_pipeline),
        "merge_weights": {"api": round(wa, 3), "llm": round(wl, 3)},
    }

    if not groq_ok or not llm_json:
        return base

    llm_sp = _clamp_unit(llm_json.get("severity_phoneme"), float(api_severity_phoneme))
    llm_st = _clamp_unit(llm_json.get("severity_phoneme_strict"), float(api_severity_strict))
    llm_tx = _clamp_unit(llm_json.get("severity_text"), float(api_severity_text))
    llm_pq = _clamp_unit(llm_json.get("pronunciation_quality"), pq_api)

    sev_c = wa * float(api_severity_phoneme) + wl * llm_sp
    st_c = wa * float(api_severity_strict) + wl * llm_st
    tx_c = wa * float(api_severity_text) + wl * llm_tx
    pq_c = wa * pq_api + wl * llm_pq

    base.update(
        {
            "pronunciation_quality_combined": round(pq_c, 3),
            "severity_phoneme_combined": round(sev_c, 2),
            "severity_phoneme_strict_combined": round(st_c, 2),
            "severity_text_combined": round(tx_c, 2),
            "therapy_level_combined": therapy_level(sev_c),
            "heard_phonemes_word_g2p_combined": _validate_llm_phoneme_list(
                llm_json.get("heard_phonemes_word_g2p"), heard_word_g2p
            ),
            "heard_phonemes_fused_combined": _validate_llm_phoneme_list(
                llm_json.get("heard_phonemes_fused"), heard_fused
            ),
            "syllable_analysis_combined": _validate_llm_syllable_analysis(
                llm_json.get("syllable_analysis"), syllable_pipeline
            ),
            "tongue_position_analysis_combined": _validate_llm_tongue_analysis(
                llm_json.get("tongue_position_analysis"), tongue_pipeline
            ),
        }
    )
    return base


def groq_llm_combined_pronunciation_assessment(
    pipeline_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Groq Llama: one bundled assessment JSON in, one strict JSON out; merged later with pipeline scores.
    """
    if not groq_phoneme_judge_ready() or GroqClient is None:
        return {"ok": False, "error": "Groq phoneme judge not configured"}

    user_msg = json.dumps(
        {"pipeline_automated_assessment": pipeline_snapshot},
        ensure_ascii=False,
    )

    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        return {"ok": False, "error": "GROQ_API_KEY is empty"}

    models_to_try: List[str] = []
    seen_chat: Set[str] = set()
    for m in [GROQ_CHAT_MODEL, *GROQ_CHAT_MODEL_FALLBACKS]:
        mt = m.strip()
        if mt and mt not in seen_chat:
            seen_chat.add(mt)
            models_to_try.append(mt)

    client = GroqClient(api_key=api_key)
    last_problem = ""
    choice_text = ""

    for chat_model in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": _GROQ_COMBINED_JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": user_msg
                        + "\n\nReturn only the JSON object specified in the system message.",
                    },
                ],
                temperature=0.15,
                max_tokens=1800,
            )
            if not getattr(completion, "choices", None):
                last_problem = f"model={chat_model}: empty Groq chat choices"
                continue
            choice_text = getattr(completion.choices[0].message, "content", None) or ""
            parsed = _groq_parse_json_message_content(choice_text)
            return {
                "ok": True,
                "groq_chat_model": chat_model,
                "llm_assessment": parsed,
                "raw_reply_excerpt": (
                    choice_text[:2500] + ("…" if len(choice_text) > 2500 else "")
                ),
            }
        except json.JSONDecodeError as e:
            last_problem = f"model={chat_model}: JSON parse ({e})"
            logger.warning(
                "%s | excerpt=%s",
                last_problem,
                (choice_text or "")[:500],
            )
            continue
        except Exception as e:
            last_problem = _groq_format_api_error(e)
            logger.warning("Groq combined assessment model=%s: %s", chat_model, last_problem)
            continue

    return {
        "ok": False,
        "groq_chat_model": GROQ_CHAT_MODEL,
        "error": last_problem or "all Groq chat models failed",
        "raw_reply_excerpt": (choice_text[:1500] if choice_text else ""),
    }


def validate_single_word(text: str) -> Tuple[bool, Optional[str]]:
    text = text.strip()
    if not text:
        return False, "Input cannot be empty"
    if " " in text:
        return (
            False,
            "Input must be a single word or sound, not a sentence. Please provide only one word.",
        )
    sentence_punctuation = {".", ",", "!", "?", ";", ":", "-", "—", "–", "'", '"'}
    if any(char in sentence_punctuation for char in text):
        return False, "Input must be a single word or sound. Remove punctuation marks and hyphens."
    if len(text) > MAX_WORD_LENGTH:
        return (
            False,
            f"Input too long. Maximum length is {MAX_WORD_LENGTH} characters for a single word/sound.",
        )
    if not text.isalpha():
        return (
            False,
            "Input must contain only alphabetic characters (letters only, no numbers, hyphens, or special characters).",
        )
    if not any(c.isalpha() for c in text):
        return False, "Input must contain at least one letter."
    return True, None


def perform_basic_analysis_phoneme(expected: str, heard_ph: List[str]) -> Dict[str, Any]:
    """SODA / severity from expected-text phonemes vs phoneme-ASR output (no lexical prediction)."""
    expected_ph = word_to_phonemes(expected)
    predicted_ph = heard_ph
    ec = canonical_phoneme_seq(expected_ph)
    pc = canonical_phoneme_seq(predicted_ph)
    sev_text = calculate_severity(" ".join(ec), " ".join(pc))
    sev_ph = phoneme_sequence_severity(expected_ph, predicted_ph)
    sev_ph_strict = calculate_severity(expected_ph, predicted_ph)
    error_type = detect_soda_phoneme_lists(expected_ph, predicted_ph)
    is_correct = error_type == "Correct"
    base_phoneme = None
    target_phoneme = None
    phoneme_pos = None
    if not is_correct:
        base_phoneme = first_mismatched_expected_phone(expected_ph, predicted_ph)
        if base_phoneme:
            target_phoneme = build_cv(expected, base_phoneme)
            phoneme_pos = phoneme_position(expected, base_phoneme)
    return {
        "is_correct": is_correct,
        "error_type": error_type,
        "severity_text": round(sev_text, 2),
        "severity_phoneme": round(sev_ph, 2),
        "therapy_level": therapy_level(sev_ph),
        "base_phoneme": base_phoneme,
        "target_phoneme": target_phoneme,
        "phoneme_position": phoneme_pos,
        "expected_phonemes": expected_ph,
        "predicted_phonemes": predicted_ph,
        "severity_phoneme_strict": round(float(sev_ph_strict), 2),
        "phoneme_equivalence_mode": PHONEME_EQUIVALENCE_MODE,
    }


app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze():
    job_id = str(uuid.uuid4())
    raw_path = TEMP_DIR / f"{job_id}.input"
    wav_path = TEMP_DIR / f"{job_id}.wav"
    # Optional `include_meta=true` adds a small `meta` object to success and error payloads.
    include_meta = _parse_bool_arg(request.args.get("include_meta"), False)

    try:
        audio = request.files.get("audio")
        expected = request.form.get("expected_text")
        if not audio or not expected:
            return _analyze_error_response(
                "audio and expected_text required",
                include_meta=include_meta,
            )

        expected = expected.strip().lower()

        is_valid, error_msg = validate_single_word(expected)
        if not is_valid:
            return _analyze_error_response(
                "Invalid input format",
                details=error_msg,
                include_meta=include_meta,
            )

        audio.save(str(raw_path))
        try:
            convert_to_wav_16k_mono(raw_path, wav_path)
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return _analyze_error_response(
                "Audio conversion failed",
                details=str(e),
                include_meta=include_meta,
            )

        try:
            heard_ctc, conf_ctc = analyze_asr_phoneme_ctc(wav_path)
        except Exception as e:
            logger.error(f"ASR failed: {e}")
            return _analyze_error_response(
                "Speech recognition failed",
                details=str(e),
                status_code=500,
                include_meta=include_meta,
            )

        heard_from_word: List[str] = []
        predicted_word = ""
        conf_word = 0.0
        if word_asr_ready():
            try:
                predicted_word, conf_word = analyze_asr_word_ctc(wav_path)
                heard_from_word = word_to_phonemes(predicted_word) if predicted_word else []
            except Exception as e:
                logger.warning("Word ASR inference failed: %s", e)

        predicted_groq = ""
        conf_groq = 0.0
        heard_from_groq: List[str] = []
        if groq_asr_ready():
            try:
                predicted_groq, conf_groq = analyze_asr_groq_word(wav_path)
                heard_from_groq = word_to_phonemes(predicted_groq) if predicted_groq else []
            except Exception as e:
                logger.warning("Groq ASR failed: %s", _groq_format_api_error(e))

        if not heard_ctc and not heard_from_word and not heard_from_groq:
            return _analyze_error_response(
                "No speech detected",
                include_meta=include_meta,
            )

        if heard_ctc and heard_from_word:
            heard_ph = fuse_phoneme_ctc_and_word(heard_ctc, predicted_word)
        elif heard_from_word:
            heard_ph = heard_from_word
        else:
            heard_ph = list(heard_ctc)

        if predicted_groq:
            if heard_ph:
                heard_ph = fuse_phoneme_ctc_and_word(heard_ph, predicted_groq)
            else:
                heard_ph = list(heard_from_groq)

        fusion_confs = [c for c in (conf_ctc, conf_word, conf_groq) if float(c) > 0]
        asr_confidence = (
            sum(fusion_confs) / len(fusion_confs) if fusion_confs else conf_ctc
        )

        predicted = " ".join(heard_ph)

        expected_syllables = split_into_syllable_like_chunks(expected)
        predicted_syllables = phoneme_chunks_from_arpa(heard_ph)
        syllable_analysis = build_syllable_analysis(expected_syllables, predicted_syllables)

        basic_results = perform_basic_analysis_phoneme(expected, heard_ph)
        wav = load_audio_tensor(wav_path)
        expected_ph = basic_results["expected_phonemes"]
        predicted_ph = basic_results["predicted_phonemes"]
        articulation = articulation_analysis(expected_ph, predicted_ph)
        taxonomy = build_error_taxonomy_from_alignment(expected_ph, predicted_ph)
        tongue = tongue_feedback(expected_ph, predicted_ph)

        # Annotate phoneme-level errors with syllable (sound) position
        articulation, taxonomy, tongue = annotate_errors_with_syllables(
            expected,
            predicted,
            expected_syllables,
            predicted_syllables,
            expected_ph,
            articulation,
            taxonomy,
            tongue,
        )
        sev_acoustic = composite_severity(
            expected,
            predicted,
            expected_ph,
            predicted_ph,
            wav,
            asr_confidence,
        )

        lexical_mismatch = wrong_lexical_word(expected, predicted_word)
        aggregate_is_correct = bool(basic_results["is_correct"]) and not lexical_mismatch

        pipeline_snapshot: Dict[str, Any] = {
            "expected_word": expected,
            "reference_dictionary_phonemes": expected_ph,
            "automatic_multistream_recognition": {
                "phoneme_ctc": heard_ctc,
                "word_transcript": predicted_word or "",
                "word_g2p_phonemes": heard_from_word,
                "groq_word_transcript": predicted_groq or "",
                "groq_g2p_phonemes": heard_from_groq,
                "fused_phonemes": heard_ph,
                "asr_confidence_phoneme_ctc": round(conf_ctc, 3),
                "asr_confidence_word": round(conf_word, 3) if word_asr_ready() else None,
                "asr_confidence_groq": round(conf_groq, 3) if groq_asr_ready() else None,
            },
            "automatic_metrics": {
                "severity_phoneme": basic_results["severity_phoneme"],
                "severity_phoneme_strict": basic_results["severity_phoneme_strict"],
                "severity_text": basic_results["severity_text"],
                "acoustic_severity": round(sev_acoustic, 3),
                "is_correct": aggregate_is_correct,
                "asr_confidence": round(asr_confidence, 3),
                "predicted_surface": predicted,
                "error_type": basic_results["error_type"],
                "base_phoneme": basic_results["base_phoneme"],
                "target_phoneme": basic_results["target_phoneme"],
            },
            "syllable_analysis_automatic": syllable_analysis,
            "tongue_position_analysis_automatic": tongue,
            "error_taxonomy_automatic": taxonomy,
        }

        groq_similarity_blob: Dict[str, Any]
        if groq_phoneme_judge_ready():
            try:
                groq_similarity_blob = groq_llm_combined_pronunciation_assessment(pipeline_snapshot)
            except Exception as e:
                logger.warning("Groq combined pronunciation judge failed: %s", e)
                groq_similarity_blob = {"ok": False, "groq_chat_model": GROQ_CHAT_MODEL, "error": str(e)}
        else:
            groq_similarity_blob = {"ok": False, "skipped": True}

        llm_assessment_premerge = groq_similarity_blob.get("llm_assessment")
        llm_dict = llm_assessment_premerge if isinstance(llm_assessment_premerge, dict) else None
        merged_combo = merge_pipeline_and_llm_assessment(
            api_severity_phoneme=float(basic_results["severity_phoneme"]),
            api_severity_strict=float(basic_results["severity_phoneme_strict"]),
            api_severity_text=float(basic_results["severity_text"]),
            heard_word_g2p=heard_from_word,
            heard_fused=heard_ph,
            syllable_pipeline=syllable_analysis,
            tongue_pipeline=tongue,
            llm_json=llm_dict if groq_similarity_blob.get("ok") else None,
            groq_ok=bool(groq_similarity_blob.get("ok")),
        )

        slim = build_minimal_analyze_payload(
            expected=expected,
            basic_results=basic_results,
            sev_acoustic=sev_acoustic,
            merged_combo=merged_combo,
            taxonomy=taxonomy,
            articulation=articulation,
            tongue=tongue,
            expected_ph=expected_ph,
            predicted_ph=predicted_ph,
            predicted_word=predicted_word,
            whisper_word=predicted_groq or "",
            whisper_phonemes_g2p=heard_from_groq,
        )
        if include_meta:
            out = dict(slim)
            out["meta"] = {
                "model": ASR_MODEL_PHONEME,
                "device": str(device),
                "asr_confidence": round(float(asr_confidence), 3),
                "groq_llm_blend_api_weight": GROQ_COMBINED_WEIGHT_API,
                "groq_whisper_large_model": GROQ_WHISPER_MODEL if groq_asr_ready() else None,
                "groq_whisper_used": bool(groq_asr_ready() and (predicted_groq or "").strip()),
            }
            return jsonify(out)
        return jsonify(slim)
    except Exception as e:
        logger.exception("Unexpected error in analyze endpoint")
        return _analyze_error_response(
            "Internal server error",
            details=str(e),
            status_code=500,
            include_meta=include_meta,
        )
    finally:
        for p in (raw_path, wav_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {p}: {e}")


@app.route("/health", methods=["GET"])
def health():
    bits = ["phoneme_ctc"]
    if word_asr_ready():
        bits.append("wav2vec2")
    if groq_asr_ready():
        bits.append("groq_whisper")
    return jsonify(
        {
            "status": "healthy",
            "device": str(device),
            "phoneme_asr_model": ASR_MODEL_PHONEME,
            "asr_mode": "+".join(bits),
            "word_asr_ready": word_asr_ready(),
            "groq_llm_judge_ready": groq_phoneme_judge_ready(),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
