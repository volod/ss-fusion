"""Unit tests for pipeline.label_vocab."""


from selfsuvis.pipeline.vision.labels import DEFAULT_LABELS, load_labels

# --- DEFAULT_LABELS ---

def test_default_labels_is_nonempty_list():
    """DEFAULT_LABELS is a non-empty list of strings."""
    assert isinstance(DEFAULT_LABELS, list)
    assert len(DEFAULT_LABELS) > 0


def test_default_labels_all_strings():
    """Every entry in DEFAULT_LABELS is a non-empty string."""
    assert all(isinstance(lbl, str) and lbl for lbl in DEFAULT_LABELS)


def test_default_labels_no_duplicates():
    """DEFAULT_LABELS contains no duplicate entries."""
    assert len(DEFAULT_LABELS) == len(set(DEFAULT_LABELS))


# --- load_labels: fallback cases ---

def test_load_labels_none_returns_defaults():
    """load_labels(None) returns DEFAULT_LABELS."""
    assert load_labels(None) is DEFAULT_LABELS


def test_load_labels_nonexistent_path_returns_defaults(tmp_path):
    """load_labels with a path that doesn't exist returns DEFAULT_LABELS."""
    result = load_labels(str(tmp_path / "no_such_file.txt"))
    assert result is DEFAULT_LABELS


def test_load_labels_empty_file_returns_defaults(tmp_path):
    """load_labels with an empty file falls back to DEFAULT_LABELS."""
    f = tmp_path / "empty.txt"
    f.write_text("")
    result = load_labels(str(f))
    assert result is DEFAULT_LABELS


def test_load_labels_all_comments_returns_defaults(tmp_path):
    """load_labels with a file of only comments falls back to DEFAULT_LABELS."""
    f = tmp_path / "comments.txt"
    f.write_text("# this is a comment\n# another comment\n")
    result = load_labels(str(f))
    assert result is DEFAULT_LABELS


# --- load_labels: file reading ---

def test_load_labels_from_file_happy_path(tmp_path):
    """load_labels reads labels from a file, one per line."""
    f = tmp_path / "labels.txt"
    f.write_text("dog\ncat\nbird\n")
    result = load_labels(str(f))
    assert result == ["dog", "cat", "bird"]


def test_load_labels_strips_whitespace(tmp_path):
    """load_labels strips leading/trailing whitespace from each line."""
    f = tmp_path / "labels.txt"
    f.write_text("  dog  \n  cat\nbird  \n")
    result = load_labels(str(f))
    assert result == ["dog", "cat", "bird"]


def test_load_labels_skips_blank_lines(tmp_path):
    """load_labels ignores blank lines."""
    f = tmp_path / "labels.txt"
    f.write_text("dog\n\ncat\n\nbird\n")
    result = load_labels(str(f))
    assert result == ["dog", "cat", "bird"]


def test_load_labels_skips_comment_lines(tmp_path):
    """load_labels ignores lines starting with '#'."""
    f = tmp_path / "labels.txt"
    f.write_text("# header\ndog\n# ignore this\ncat\n")
    result = load_labels(str(f))
    assert result == ["dog", "cat"]


def test_load_labels_mixed_content(tmp_path):
    """load_labels handles comments, blank lines, and real labels together."""
    f = tmp_path / "labels.txt"
    f.write_text(
        "# animals\ndog\ncat\n\n# vehicles\ncar\ntruck\n"
    )
    result = load_labels(str(f))
    assert result == ["dog", "cat", "car", "truck"]
