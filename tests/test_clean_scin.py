"""Unit tests for skin_type_classifier.data.clean_scin pure functions."""

import pytest

from skin_type_classifier.data.clean_scin import extract_image_filename, parse_fst_label


class TestExtractImageFilename:
    """Tests for extract_image_filename()."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("csv_path", "expected"),
        [
            ("dataset/images/-123.png", "-123.png"),
            ("dataset/images/some_file.jpg", "some_file.jpg"),
            ("images/file.png", "file.png"),
            ("file.png", "file.png"),
            ("a/b/c/d/deep_nested.png", "deep_nested.png"),
        ],
    )
    def test_valid_paths(self, csv_path: str, expected: str) -> None:
        """Extracts the filename from various path depths."""
        assert extract_image_filename(csv_path) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "csv_path",
        [
            float("nan"),
            "",
            "   ",
            None,
        ],
    )
    def test_invalid_paths_return_none(self, csv_path: object) -> None:
        """Returns None for NaN, empty, whitespace-only, and None inputs."""
        assert extract_image_filename(csv_path) is None


class TestParseFstLabel:
    """Tests for parse_fst_label()."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("FST1", 1),
            ("FST2", 2),
            ("FST3", 3),
            ("FST4", 4),
            ("FST5", 5),
            ("FST6", 6),
        ],
    )
    def test_valid_labels(self, label: str, expected: int) -> None:
        """Parses all six valid Fitzpatrick skin type labels."""
        assert parse_fst_label(label) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "label",
        [
            float("nan"),
            "",
            "FST",
            "FST0x",
            "FOOBAR",
            "2",
            None,
        ],
    )
    def test_invalid_labels_return_none(self, label: object) -> None:
        """Returns None for NaN, empty, prefix-only, non-numeric suffix, no prefix, and None."""
        assert parse_fst_label(label) is None

    @pytest.mark.unit
    def test_returns_int_type(self) -> None:
        """Ensures the return type is int, not float."""
        result = parse_fst_label("FST3")
        assert isinstance(result, int)
