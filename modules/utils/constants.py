from modules.utils.i18n import _

class _StaticI18nString(str):
    def unwrap(self):
        return str(self)


try:
    AUTOMATIC_DETECTION = _("Automatic Detection")
except LookupError:
    AUTOMATIC_DETECTION = _StaticI18nString("Automatic Detection")

GRADIO_NONE_STR = ""
GRADIO_NONE_NUMBER_MAX = 9999
GRADIO_NONE_NUMBER_MIN = 0
