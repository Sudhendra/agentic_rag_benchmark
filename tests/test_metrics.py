"""Unit tests for evaluation metrics."""

from src.evaluation.metrics import (
    exact_match,
    f1_score,
    joint_metrics,
    normalize_answer,
    supporting_fact_metrics,
)


class TestNormalizeAnswer:
    """Tests for answer normalization."""

    def test_lowercase(self):
        """Test that normalization lowercases text."""
        assert normalize_answer("HELLO") == "hello"
        assert normalize_answer("HeLLo WoRLd") == "hello world"

    def test_remove_articles(self):
        """Test that normalization removes articles."""
        assert normalize_answer("the quick brown fox") == "quick brown fox"
        assert normalize_answer("a cat and an apple") == "cat and apple"
        assert normalize_answer("The Eiffel Tower") == "eiffel tower"

    def test_remove_punctuation(self):
        """Test that normalization removes punctuation."""
        assert normalize_answer("hello, world!") == "hello world"
        assert normalize_answer("what's up?") == "whats up"
        assert normalize_answer("test...test") == "testtest"

    def test_whitespace_normalization(self):
        """Test that normalization fixes whitespace."""
        assert normalize_answer("hello   world") == "hello world"
        assert normalize_answer("  hello  ") == "hello"
        assert normalize_answer("cat\t\ndog") == "cat dog"

    def test_combined_normalization(self):
        """Test combined normalization effects."""
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
        assert normalize_answer("  A  big   CAT!  ") == "big cat"

    def test_empty_string(self):
        """Test normalization of empty string."""
        assert normalize_answer("") == ""

    def test_only_articles(self):
        """Test normalization when input is only articles."""
        assert normalize_answer("the a an") == ""


class TestExactMatch:
    """Tests for exact match metric."""

    def test_identical_strings(self):
        """Test exact match with identical strings."""
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        """Test that exact match is case insensitive."""
        assert exact_match("PARIS", "paris") == 1.0
        assert exact_match("Paris", "PARIS") == 1.0

    def test_with_articles(self):
        """Test that articles are ignored."""
        assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0
        assert exact_match("a cat", "the cat") == 1.0

    def test_with_punctuation(self):
        """Test that punctuation is ignored."""
        assert exact_match("Hello, World!", "hello world") == 1.0

    def test_different_strings(self):
        """Test exact match with different strings."""
        assert exact_match("Paris", "London") == 0.0
        assert exact_match("yes", "no") == 0.0

    def test_empty_strings(self):
        """Test exact match with empty strings."""
        assert exact_match("", "") == 1.0

    def test_one_empty_string(self):
        """Test exact match when one string is empty."""
        assert exact_match("hello", "") == 0.0
        assert exact_match("", "hello") == 0.0


class TestF1Score:
    """Tests for F1 score metric."""

    def test_identical_strings(self):
        """Test F1 with identical strings."""
        assert f1_score("quick brown fox", "quick brown fox") == 1.0

    def test_partial_overlap(self):
        """Test F1 with partial overlap."""
        score = f1_score("quick brown fox", "quick red fox")
        # Common: "quick", "fox" (2 tokens)
        # Pred: 3 tokens, Gold: 3 tokens
        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        assert abs(score - (2 / 3)) < 0.01

    def test_no_overlap(self):
        """Test F1 with no overlap."""
        assert f1_score("hello", "world") == 0.0
        assert f1_score("cat dog", "fish bird") == 0.0

    def test_empty_strings(self):
        """Test F1 with empty strings."""
        assert f1_score("", "") == 1.0

    def test_one_empty_string(self):
        """Test F1 when one string is empty."""
        assert f1_score("hello world", "") == 0.0
        assert f1_score("", "hello world") == 0.0

    def test_subset_match(self):
        """Test F1 when one is subset of other."""
        # Pred: "quick brown", Gold: "quick brown fox"
        # Common: 2, Precision = 2/2 = 1, Recall = 2/3
        # F1 = 2 * 1 * (2/3) / (1 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8
        score = f1_score("quick brown", "quick brown fox")
        assert abs(score - 0.8) < 0.01

    def test_superset_match(self):
        """Test F1 when prediction is superset of gold."""
        # Pred: "quick brown fox jumps", Gold: "quick brown fox"
        # Common: 3, Precision = 3/4, Recall = 3/3 = 1
        # F1 = 2 * (3/4) * 1 / (3/4 + 1) = 1.5 / 1.75 = 6/7
        score = f1_score("quick brown fox jumps", "quick brown fox")
        assert abs(score - (6 / 7)) < 0.01

    def test_repeated_words(self):
        """Test F1 with repeated words."""
        # Both have "the" twice
        score = f1_score("the the", "the the")
        assert score == 1.0


class TestSupportingFactMetrics:
    """Tests for supporting fact metrics."""

    def test_exact_match(self):
        """Test when predicted and gold facts match exactly."""
        pred = [("Doc1", 0), ("Doc2", 1)]
        gold = [("Doc1", 0), ("Doc2", 1)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 1.0
        assert f1 == 1.0

    def test_partial_match(self):
        """Test when there's partial overlap."""
        pred = [("Doc1", 0), ("Doc3", 2)]
        gold = [("Doc1", 0), ("Doc2", 1)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 0.0
        # Common: 1, Precision = 1/2, Recall = 1/2, F1 = 0.5
        assert abs(f1 - 0.5) < 0.01

    def test_no_match(self):
        """Test when there's no overlap."""
        pred = [("Doc3", 2)]
        gold = [("Doc1", 0)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 0.0
        assert f1 == 0.0

    def test_empty_predictions(self):
        """Test with empty predictions."""
        pred = []
        gold = [("Doc1", 0)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 0.0
        assert f1 == 0.0

    def test_empty_gold(self):
        """Test with empty gold facts."""
        pred = [("Doc1", 0)]
        gold = []
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 0.0
        assert f1 == 0.0

    def test_both_empty(self):
        """Test with both empty."""
        pred = []
        gold = []
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 1.0
        assert f1 == 1.0

    def test_order_independent(self):
        """Test that order doesn't matter."""
        pred = [("Doc2", 1), ("Doc1", 0)]
        gold = [("Doc1", 0), ("Doc2", 1)]
        em, f1 = supporting_fact_metrics(pred, gold)
        assert em == 1.0
        assert f1 == 1.0


class TestJointMetrics:
    """Tests for joint answer + supporting fact metrics."""

    def test_both_perfect(self):
        """Test when both answer and SP are perfect."""
        joint_em, joint_f1 = joint_metrics(1.0, 1.0, 1.0, 1.0)
        assert joint_em == 1.0
        assert joint_f1 == 1.0

    def test_answer_wrong(self):
        """Test when answer is wrong but SP is correct."""
        joint_em, joint_f1 = joint_metrics(0.0, 0.5, 1.0, 1.0)
        assert joint_em == 0.0
        assert joint_f1 == 0.5

    def test_sp_wrong(self):
        """Test when SP is wrong but answer is correct."""
        joint_em, joint_f1 = joint_metrics(1.0, 1.0, 0.0, 0.5)
        assert joint_em == 0.0
        assert joint_f1 == 0.5

    def test_both_partial(self):
        """Test when both are partial."""
        joint_em, joint_f1 = joint_metrics(0.5, 0.8, 0.5, 0.6)
        assert joint_em == 0.25  # 0.5 * 0.5
        assert abs(joint_f1 - 0.48) < 0.01  # 0.8 * 0.6

    def test_none_sp_metrics(self):
        """Test when SP metrics are None."""
        joint_em, joint_f1 = joint_metrics(1.0, 0.9, None, None)
        assert joint_em is None
        assert joint_f1 is None

    def test_partial_none(self):
        """Test when only one SP metric is None."""
        joint_em, joint_f1 = joint_metrics(1.0, 0.9, 1.0, None)
        assert joint_em is None
        assert joint_f1 is None
