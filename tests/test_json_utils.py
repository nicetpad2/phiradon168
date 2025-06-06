import os
import sys
import json
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.json_utils import load_json_with_comments


def test_load_json_with_comments_basic(tmp_path):
    path = tmp_path / "test.json"
    path.write_text("""\
# comment line
{
    // inline comment should be ignored
    \"a\": 1,
    \"b\": 2
}
// trailing comment
""", encoding="utf-8")
    data = load_json_with_comments(str(path))
    assert data == {"a": 1, "b": 2}


def test_load_json_with_comments_invalid(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{\n  \"a\": 1,\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json_with_comments(str(path))


def test_load_json_with_string_hash(tmp_path):
    path = tmp_path / "hash.json"
    path.write_text("""{
    \"#foo\": \"bar\",
    \"url\": \"http://example.com\"
}
""", encoding="utf-8")
    data = load_json_with_comments(str(path))
    assert data == {"#foo": "bar", "url": "http://example.com"}



