from __future__ import annotations

from typing import Optional

from modules.utils.text import repair_mojibake_text


WHISPER_LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

WHISPER_LANGUAGE_ALIASES = {
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

TO_LANGUAGE_CODE = {
    name: code for code, name in WHISPER_LANGUAGES.items()
}
TO_LANGUAGE_CODE.update(WHISPER_LANGUAGE_ALIASES)


def normalize_lang_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    if hasattr(value, "unwrap"):
        value = value.unwrap()

    if not isinstance(value, str):
        return value

    normalized = repair_mojibake_text(value).strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered in WHISPER_LANGUAGES:
        return lowered
    if lowered in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[lowered]
    if lowered in WHISPER_LANGUAGES.values():
        return TO_LANGUAGE_CODE[lowered]
    return lowered


def normalize_lang_choice(value: Optional[str], automatic_detection_label: str) -> str:
    normalized = normalize_lang_value(value)
    if normalized is None:
        raw_value = value.unwrap() if hasattr(value, "unwrap") else value
        if isinstance(raw_value, str) and raw_value.strip():
            raw_normalized = repair_mojibake_text(raw_value).strip()
            if raw_normalized.casefold() == automatic_detection_label.casefold():
                return automatic_detection_label
        return "english"

    if normalized in WHISPER_LANGUAGES:
        return WHISPER_LANGUAGES[normalized]
    if normalized in WHISPER_LANGUAGES.values():
        return normalized
    return "english"
