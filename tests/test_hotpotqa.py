"""Unit tests for HotpotQA data loader."""

from unittest.mock import patch

import pytest

from src.core.types import QuestionType
from src.data.hotpotqa import HotpotQALoader, load_hotpotqa


class MockDataset:
    """Mock HuggingFace dataset for testing."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def select(self, indices):
        return MockDataset([self.data[i] for i in indices])


@pytest.fixture
def sample_hotpotqa_item():
    """Create a sample HotpotQA item."""
    return {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "What government position was held by the woman who portrayed Nora combating in Backcraft?",
        "answer": "Senator",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {
            "title": ["Backcraft", "Rebecca Romijn"],
            "sent_id": [0, 1],
        },
        "context": {
            "title": ["Backcraft", "Rebecca Romijn", "Some Other Doc"],
            "sentences": [
                [
                    "Backcraft is a 1991 film.",
                    "Nora is a character in the film.",
                    "She is portrayed by Rebecca Romijn.",
                ],
                ["Rebecca Romijn is an actress.", "She served as a Senator."],
                ["This is an unrelated document.", "It has multiple sentences."],
            ],
        },
    }


@pytest.fixture
def mock_dataset(sample_hotpotqa_item):
    """Create a mock dataset with one item."""
    return MockDataset([sample_hotpotqa_item])


class TestHotpotQALoaderQuestions:
    """Tests for question parsing."""

    def test_parses_question_id(self, mock_dataset):
        """Test that question ID is parsed correctly."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            assert questions[0].id == "5a8b57f25542995d1e6f1371"

    def test_parses_question_text(self, mock_dataset):
        """Test that question text is parsed correctly."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            assert "government position" in questions[0].text

    def test_parses_gold_answer(self, mock_dataset):
        """Test that gold answer is parsed correctly."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            assert questions[0].gold_answer == "Senator"

    def test_parses_bridge_type(self, mock_dataset):
        """Test that bridge question type is parsed."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            assert questions[0].type == QuestionType.BRIDGE

    def test_parses_comparison_type(self, sample_hotpotqa_item):
        """Test that comparison question type is parsed."""
        sample_hotpotqa_item["type"] = "comparison"
        mock_ds = MockDataset([sample_hotpotqa_item])

        with patch("src.data.hotpotqa.load_dataset", return_value=mock_ds):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            assert questions[0].type == QuestionType.COMPARISON

    def test_parses_supporting_facts(self, mock_dataset):
        """Test that supporting facts are parsed correctly."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            sf = questions[0].supporting_facts
            assert sf is not None
            assert ("Backcraft", 0) in sf
            assert ("Rebecca Romijn", 1) in sf

    def test_metadata_includes_level(self, mock_dataset):
        """Test that metadata includes difficulty level."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            assert questions[0].metadata.get("level") == "hard"


class TestHotpotQALoaderCorpus:
    """Tests for corpus/document parsing."""

    def test_parses_documents(self, mock_dataset):
        """Test that documents are parsed from context."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            _, corpus = loader.load()

            # Should have 3 documents
            assert len(corpus) == 3

    def test_document_has_title(self, mock_dataset):
        """Test that documents have titles."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            _, corpus = loader.load()

            titles = [doc.title for doc in corpus]
            assert "Backcraft" in titles
            assert "Rebecca Romijn" in titles

    def test_document_has_text(self, mock_dataset):
        """Test that documents have concatenated text."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            _, corpus = loader.load()

            backcraft_doc = next(doc for doc in corpus if doc.title == "Backcraft")
            assert "Backcraft is a 1991 film." in backcraft_doc.text
            assert "Nora is a character" in backcraft_doc.text

    def test_document_has_sentences(self, mock_dataset):
        """Test that documents have sentence list."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            _, corpus = loader.load()

            backcraft_doc = next(doc for doc in corpus if doc.title == "Backcraft")
            assert len(backcraft_doc.sentences) == 3

    def test_document_id_format(self, mock_dataset):
        """Test that document IDs include question ID and title."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            loader = HotpotQALoader()
            _, corpus = loader.load()

            doc_ids = [doc.id for doc in corpus]
            assert any("5a8b57f25542995d1e6f1371_Backcraft" in id for id in doc_ids)


class TestHotpotQALoaderSettings:
    """Tests for loader settings."""

    def test_default_setting_is_distractor(self):
        """Test that default setting is distractor."""
        loader = HotpotQALoader()
        assert loader.setting == "distractor"

    def test_default_split_is_validation(self):
        """Test that default split is validation."""
        loader = HotpotQALoader()
        assert loader.split == "validation"

    def test_passes_setting_to_load_dataset(self, mock_dataset):
        """Test that setting is passed to load_dataset."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset) as mock_load:
            loader = HotpotQALoader(setting="distractor", split="train")
            loader.load()

            mock_load.assert_called_with("hotpot_qa", "distractor", split="train")

    def test_subset_size_limits_data(self, sample_hotpotqa_item):
        """Test that subset_size limits the data."""
        # Create dataset with multiple items
        items = [dict(sample_hotpotqa_item, id=f"q{i}") for i in range(10)]
        mock_ds = MockDataset(items)

        with patch("src.data.hotpotqa.load_dataset", return_value=mock_ds):
            loader = HotpotQALoader(subset_size=3)
            questions, _ = loader.load()

            assert len(questions) == 3


class TestHotpotQALoaderEdgeCases:
    """Tests for edge cases in data loading."""

    def test_handles_missing_supporting_facts(self, sample_hotpotqa_item):
        """Test handling of missing supporting facts."""
        del sample_hotpotqa_item["supporting_facts"]
        mock_ds = MockDataset([sample_hotpotqa_item])

        with patch("src.data.hotpotqa.load_dataset", return_value=mock_ds):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            assert questions[0].supporting_facts is None

    def test_handles_unknown_question_type(self, sample_hotpotqa_item):
        """Test handling of unknown question type."""
        sample_hotpotqa_item["type"] = "unknown_type"
        mock_ds = MockDataset([sample_hotpotqa_item])

        with patch("src.data.hotpotqa.load_dataset", return_value=mock_ds):
            loader = HotpotQALoader()
            questions, _ = loader.load()

            # Should default to BRIDGE
            assert questions[0].type == QuestionType.BRIDGE

    def test_handles_empty_context(self, sample_hotpotqa_item):
        """Test handling of empty context."""
        sample_hotpotqa_item["context"] = {"title": [], "sentences": []}
        mock_ds = MockDataset([sample_hotpotqa_item])

        with patch("src.data.hotpotqa.load_dataset", return_value=mock_ds):
            loader = HotpotQALoader()
            questions, corpus = loader.load()

            assert len(questions) == 1
            assert len(corpus) == 0


class TestLoadHotpotqaConvenience:
    """Tests for the convenience function."""

    def test_load_hotpotqa_returns_tuple(self, mock_dataset):
        """Test that load_hotpotqa returns (questions, corpus) tuple."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset):
            result = load_hotpotqa()

            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_load_hotpotqa_passes_parameters(self, mock_dataset):
        """Test that load_hotpotqa passes parameters to loader."""
        with patch("src.data.hotpotqa.load_dataset", return_value=mock_dataset) as mock_load:
            load_hotpotqa(setting="fullwiki", split="train", subset_size=50)

            mock_load.assert_called_with("hotpot_qa", "fullwiki", split="train")
