"""Tests for the ``mediaref`` CLI patch operations.

The pure-text patch helpers are exercised directly. Filesystem-mutating
subcommands aren't tested end-to-end because we don't want CI runs to
mutate the installed datasets package; their behavior is a thin wrapper
around the helpers tested here plus standard read_text / write_text.
"""

from __future__ import annotations

import pytest

from mediaref._cli import _BEGIN, _END, apply_patch, is_enabled, main, remove_patch


class TestPatchOperations:
    def test_apply_to_empty_string(self):
        out = apply_patch("")
        assert _BEGIN in out
        assert _END in out
        assert is_enabled(out)

    def test_apply_to_existing_content(self):
        original = "# existing module\n_FEATURE_TYPES = {}\n"
        out = apply_patch(original)
        assert original in out
        assert is_enabled(out)

    def test_apply_idempotent(self):
        text = "module body\n"
        once = apply_patch(text)
        twice = apply_patch(once)
        thrice = apply_patch(twice)
        assert once == twice == thrice

    def test_remove_no_op_when_not_present(self):
        text = "module body without the patch"
        assert remove_patch(text) == text

    def test_remove_after_apply_recovers_original(self):
        original = "module body\n"
        applied = apply_patch(original)
        removed = remove_patch(applied)
        assert removed == original
        assert not is_enabled(removed)

    def test_apply_remove_round_trip_repeated(self):
        text = "module body\n"
        for _ in range(5):
            text = remove_patch(apply_patch(text))
        assert text == "module body\n"

    def test_remove_strips_surrounding_blank_lines(self):
        # Patch is appended with leading/trailing newlines; removing it
        # shouldn't leave a trail of empty lines behind.
        original = "module body\n"
        applied = apply_patch(original) + "\n\n"  # simulate trailing whitespace
        removed = remove_patch(applied)
        assert "\n\n\n" not in removed

    def test_is_enabled_negative_cases(self):
        assert not is_enabled("")
        assert not is_enabled("# completely unrelated")
        assert not is_enabled("import datasets")

    def test_patch_text_imports_mediaref_hf(self):
        # Make sure the patched text actually does what we claim — imports
        # mediaref.hf and assigns the feature into _FEATURE_TYPES.
        out = apply_patch("")
        assert "from mediaref.hf import MediaRefFeature" in out
        assert '_FEATURE_TYPES["MediaRef"] = _mediaref_MediaRefFeature' in out
        # ImportError fallback stays silent (no log spam if mediaref[hf] not installed).
        assert "except ImportError" in out


class TestCLIArgumentParsing:
    def test_no_subcommand_errors(self):
        # argparse exits with non-zero when subcommand is missing/required.
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code != 0

    def test_unknown_subcommand_errors(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["bogus"])
        assert excinfo.value.code != 0
