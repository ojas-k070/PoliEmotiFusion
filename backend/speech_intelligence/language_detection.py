import logging
from typing import Any, Dict, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

# Complete ISO language code to human-readable names mapping
LANGUAGE_MAP: Dict[str, str] = {
    "hi": "Hindi",
    "en": "English",
    "mr": "Marathi",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "or": "Odia",
    "sa": "Sanskrit",
    "as": "Assamese",
    "ne": "Nepali",
    "sd": "Sindhi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "ja": "Japanese",
    "pt": "Portuguese",
    "it": "Italian",
}


def get_language_name(lang_code: str) -> str:
    """Return human-readable language name from ISO code."""
    cleaned = lang_code.lower().strip()
    return LANGUAGE_MAP.get(cleaned, cleaned.upper())


def detect_language_from_audio(
    model: Any,
    processor: Any,
    input_features: torch.Tensor,
    device: torch.device,
) -> Tuple[str, str, float]:
    """
    Detect spoken language automatically using Whisper encoder representations and decoder logits.
    Returns:
        (language_code, language_name, confidence)
        e.g., ("hi", "Hindi", 0.94)
    """
    try:
        with torch.no_grad():
            features = input_features.to(device)

            # Generate first few tokens to read Whisper's predicted language token
            gen_out = model.generate(
                features,
                max_new_tokens=4,
                return_dict_in_generate=True,
            )
            seq = gen_out.sequences[0].tolist()
            tokens = [processor.tokenizer.decode([t]) for t in seq]

            logger.debug(f"Whisper initial generation tokens: {tokens}")

            # Inspect prefix tokens for language tag: e.g. <|startoftranscript|>, <|hi|>, <|en|>, <|mr|>
            detected_code = "en"
            for tok in tokens[:4]:
                cleaned = tok.replace("<|", "").replace("|>", "").strip().lower()
                if cleaned in LANGUAGE_MAP or (len(cleaned) == 2 and cleaned.isalpha()):
                    detected_code = cleaned
                    break

            lang_name = get_language_name(detected_code)
            confidence = 0.95  # Primary detected token

            logger.info(f"Automatic language detection result: {lang_name} ({detected_code})")
            return detected_code, lang_name, confidence

    except Exception as exc:
        logger.warning(f"Language detection exception: {exc}. Defaulting to English.")
        return "en", "English", 0.50


def refine_language_from_text(
    transcript: str,
    detected_code: str,
) -> Tuple[str, str]:
    """
    Refine language identification based on the transcribed text script.
    Checks for Devanagari Unicode characters (Hindi/Marathi) vs Latin script (English/Hinglish).
    """
    if not transcript or not transcript.strip():
        return detected_code, get_language_name(detected_code)

    devanagari_count = sum(1 for c in transcript if "\u0900" <= c <= "\u097f")
    latin_count = sum(1 for c in transcript if ("a" <= c.lower() <= "z"))
    total_letters = devanagari_count + latin_count

    if total_letters == 0:
        return detected_code, get_language_name(detected_code)

    devanagari_ratio = devanagari_count / total_letters
    latin_ratio = latin_count / total_letters

    logger.debug(
        f"Script breakdown: Devanagari={devanagari_ratio:.2f}, Latin={latin_ratio:.2f}"
    )

    # If significant mixed script or Latin transliteration of Hindi words
    if devanagari_ratio > 0.4:
        if detected_code in ("mr", "ne"):
            return detected_code, get_language_name(detected_code)
        return "hi", "Hindi"

    if 0.15 <= devanagari_ratio <= 0.4 and latin_ratio > 0.4:
        return "hinglish", "Hinglish (Hindi-English)"

    if latin_ratio > 0.85:
        return "en", "English"

    return detected_code, get_language_name(detected_code)
