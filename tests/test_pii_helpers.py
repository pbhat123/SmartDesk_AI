"""The phone-number check ignores random hex ids that happen to contain 98765 but still catches the number."""
import uuid

from tests.pii_helpers import has_phone


def test_the_check_catches_the_phone_number_in_every_written_form():
    for form in ("+91 98765 43210", "9876543210", "98765-43210", "+91-98765-43210", "call 98765.43210 now"):
        assert has_phone(form), form


def test_a_random_id_containing_98765_is_not_a_phone_number():
    assert not has_phone('{"id":"2f1c9d0e-5a98-4b76-a5-97ca-ec98765d0b97","start_time":"2026-10-18"}')
    assert not has_phone(f"request_id=d354a991fc4756a27b3b30d6b51dd3 {uuid.UUID(int=0x98765 << 64)}")


def test_the_redacted_form_is_clean():
    assert not has_phone("call me on [PHONE] or mail j***@example.com")


def test_the_number_cannot_match_inside_any_hex_token():
    # the digits 9876543210 are valid hex, so the worst case is an id that contains them; a token boundary rules it out
    for hex_id in ("ab9876543210cd", "9876543210ab", "ab9876543210", "f" * 8 + "98765-43210" + "e" * 4,
                   "x" * 7 + "-" + "1234-5678-9abc-def098765432109"):
        assert not has_phone(hex_id), hex_id
    assert not has_phone("98765 432101"), "a longer digit run is not the phone number"
    assert has_phone('"phone": "98765 43210"')
