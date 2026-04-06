from modules.utils.i18n import _


def test_gettext_does_not_require_request_context():
    value = _("Language")

    assert str(value) == "Language"
    assert value.unwrap() == "Language"
