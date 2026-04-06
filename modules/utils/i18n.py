from gradio.context import LocalContext
from gradio_i18n import Translate, gettext as gettext
from gradio_i18n.i18n import I18nString, TranslateContext


def _get_request():
    try:
        return LocalContext.request.get(None)
    except LookupError:
        return None


def _patch_gradio_i18n():
    if getattr(I18nString, "_whisper_webui_request_patch", False):
        return

    def _safe_new(cls, value):
        request = _get_request()
        if request is None:
            return str.__new__(cls, value)

        lang = TranslateContext.get_current_language(request)
        translated = TranslateContext.dictionary.get(lang, {}).get(value, value)
        return str.__new__(cls, translated)

    def _safe_str(self):
        request = _get_request()
        if request is None:
            return self

        lang = TranslateContext.get_current_language(request)
        result = TranslateContext.dictionary.get(lang, {}).get(self, str.__str__(self))

        for value in self.radd_values:
            result = str(value) + result

        for value in self.add_values:
            result = result + str(value)

        while len(result) >= 2 and result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        return result

    I18nString.__new__ = _safe_new
    I18nString.__str__ = _safe_str
    I18nString._whisper_webui_request_patch = True


_patch_gradio_i18n()

_ = gettext

__all__ = ["Translate", "_"]
