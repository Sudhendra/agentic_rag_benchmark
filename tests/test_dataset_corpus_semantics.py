from unittest.mock import patch

from src.data.hotpotqa import HotpotQALoader
from src.data.musique import MuSiQueLoader
from src.data.wiki2hop import Wiki2HopLoader


class MockDataset:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def select(self, indices):
        return MockDataset([self.data[i] for i in indices])


def test_hotpotqa_loader_keeps_question_scoped_candidate_corpus() -> None:
    dataset = MockDataset(
        [
            {
                "id": "q1",
                "question": "Question A?",
                "answer": "A",
                "type": "bridge",
                "context": {
                    "title": ["Shared Title", "Alpha Only"],
                    "sentences": [["Alpha shared"], ["Alpha only"]],
                },
            },
            {
                "id": "q2",
                "question": "Question B?",
                "answer": "B",
                "type": "bridge",
                "context": {
                    "title": ["Shared Title", "Beta Only"],
                    "sentences": [["Beta shared"], ["Beta only"]],
                },
            },
        ]
    )

    with patch("src.data.hotpotqa.load_dataset", return_value=dataset):
        questions, corpus = HotpotQALoader().load()

    assert len(corpus) == 4
    assert questions[0].candidate_corpus is not None
    assert [doc.id for doc in questions[0].candidate_corpus] == [
        "q1_Shared Title",
        "q1_Alpha Only",
    ]
    assert questions[1].candidate_corpus is not None
    assert [doc.id for doc in questions[1].candidate_corpus] == [
        "q2_Shared Title",
        "q2_Beta Only",
    ]


def test_musique_loader_keeps_question_scoped_candidate_corpus() -> None:
    dataset = MockDataset(
        [
            {
                "id": "q1",
                "question": "Question A?",
                "answer": "A",
                "question_decomposition": ["step 1", "step 2"],
                "paragraphs": [
                    {"title": "Shared Title", "paragraph_text": "Alpha shared", "idx": 0},
                    {"title": "Alpha Only", "paragraph_text": "Alpha only", "idx": 1},
                ],
            },
            {
                "id": "q2",
                "question": "Question B?",
                "answer": "B",
                "question_decomposition": ["step 1", "step 2"],
                "paragraphs": [
                    {"title": "Shared Title", "paragraph_text": "Beta shared", "idx": 0},
                    {"title": "Beta Only", "paragraph_text": "Beta only", "idx": 1},
                ],
            },
        ]
    )

    with patch("src.data.musique.load_dataset", return_value=dataset):
        questions, corpus = MuSiQueLoader().load()

    assert len(corpus) == 4
    assert questions[0].candidate_corpus is not None
    assert [doc.id for doc in questions[0].candidate_corpus] == [
        "q1_Shared Title_0",
        "q1_Alpha Only_1",
    ]
    assert questions[1].candidate_corpus is not None
    assert [doc.id for doc in questions[1].candidate_corpus] == [
        "q2_Shared Title_0",
        "q2_Beta Only_1",
    ]


def test_wiki2hop_loader_keeps_question_scoped_candidate_corpus() -> None:
    dataset = MockDataset(
        [
            {
                "id": "q1",
                "question": "Question A?",
                "answer": "A",
                "type": "bridging",
                "context": {
                    "title": ["Shared Title", "Alpha Only"],
                    "sentences": [["Alpha shared"], ["Alpha only"]],
                },
            },
            {
                "id": "q2",
                "question": "Question B?",
                "answer": "B",
                "type": "bridging",
                "context": {
                    "title": ["Shared Title", "Beta Only"],
                    "sentences": [["Beta shared"], ["Beta only"]],
                },
            },
        ]
    )

    with patch("src.data.wiki2hop.load_dataset", return_value=dataset):
        questions, corpus = Wiki2HopLoader().load()

    assert len(corpus) == 4
    assert questions[0].candidate_corpus is not None
    assert [doc.id for doc in questions[0].candidate_corpus] == [
        "q1_Shared Title",
        "q1_Alpha Only",
    ]
    assert questions[1].candidate_corpus is not None
    assert [doc.id for doc in questions[1].candidate_corpus] == [
        "q2_Shared Title",
        "q2_Beta Only",
    ]
