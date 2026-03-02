"""MuSiQue dataset loader."""

import os
from pathlib import Path

from datasets import DownloadConfig, load_dataset

from ..core.types import Document, Question, QuestionType


class MuSiQueLoader:
    """Loader for the MuSiQue (Multi-Step Inference) multi-hop QA dataset.

    MuSiQue requires multi-step reasoning over multiple paragraphs.
    Each question has a decomposition into sub-questions.
    """

    def __init__(
        self,
        split: str = "validation",
        subset_size: int | None = None,
        local_files_only: bool | None = None,
        cache_dir: str | None = None,
    ):
        """Initialize the loader.

        Args:
            split: Dataset split ('train' or 'validation')
            subset_size: Optional limit on number of questions
            local_files_only: Whether to force local cached dataset usage only
            cache_dir: Optional HuggingFace datasets cache directory
        """
        self.split = split
        self.subset_size = subset_size
        if local_files_only is None:
            env_value = os.getenv("MUSIQUE_LOCAL_ONLY", "1").strip().lower()
            self.local_files_only = env_value in {"1", "true", "yes", "on"}
        else:
            self.local_files_only = local_files_only

        if cache_dir is None:
            cache_dir = os.getenv(
                "MUSIQUE_CACHE_DIR",
                str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface" / "datasets"),
            )
        self.cache_dir = cache_dir

    def _parse_question_type(self, question_decomposition: list) -> QuestionType:
        """Parse MuSiQue question type based on decomposition.

        Args:
            question_decomposition: List of sub-questions

        Returns:
            QuestionType enum value (COMPOSITIONAL for multi-hop)
        """
        num_subqs = len(question_decomposition) if question_decomposition else 2
        if num_subqs >= 3:
            return QuestionType.COMPOSITIONAL
        return QuestionType.BRIDGE

    def load(self) -> tuple[list[Question], list[Document]]:
        """Load the MuSiQue dataset.

        Returns:
            Tuple of (questions, corpus)
            - questions: List of Question objects
            - corpus: List of all Document objects from paragraphs
        """
        download_config = DownloadConfig(local_files_only=self.local_files_only)
        dataset = load_dataset(
            "dgslibisey/MuSiQue",
            "default",
            split=self.split,
            cache_dir=self.cache_dir,
            download_config=download_config,
        )

        if self.subset_size:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))

        questions = []
        corpus_dict: dict[str, Document] = {}

        for item in dataset:
            q_id = item["id"]

            question_decomposition = item.get("question_decomposition", [])
            supporting_facts = None
            if "paragraphs" in item:
                paragraphs = item["paragraphs"]
                supporting_facts = []
                for para in paragraphs:
                    if para.get("is_supporting", False):
                        supporting_facts.append((para.get("title", ""), para.get("idx", 0)))

            q = Question(
                id=q_id,
                text=item["question"],
                type=self._parse_question_type(question_decomposition),
                gold_answer=item["answer"],
                supporting_facts=supporting_facts if supporting_facts else None,
                metadata={
                    "question_decomposition": question_decomposition,
                    "answer_aliases": item.get("answer_aliases", []),
                },
            )
            questions.append(q)

            if "paragraphs" in item:
                paragraphs = item["paragraphs"]
                for para in paragraphs:
                    title = para.get("title", "")
                    para_text = para.get("paragraph_text", "")
                    idx = para.get("idx", 0)

                    if not title:
                        title = f"doc_{q_id}_{idx}"

                    doc_id = f"{q_id}_{title}_{idx}"

                    if doc_id not in corpus_dict:
                        sentences = [s.strip() for s in para_text.split(". ") if s.strip()]
                        corpus_dict[doc_id] = Document(
                            id=doc_id,
                            title=title,
                            text=para_text,
                            sentences=sentences,
                        )

        return questions, list(corpus_dict.values())


def load_musique(
    split: str = "validation",
    subset_size: int | None = None,
    local_files_only: bool | None = None,
    cache_dir: str | None = None,
) -> tuple[list[Question], list[Document]]:
    """Convenience function to load MuSiQue.

    Args:
        split: Dataset split
        subset_size: Optional limit
        local_files_only: Whether to force local cached dataset usage only
        cache_dir: Optional HuggingFace datasets cache directory

    Returns:
        Tuple of (questions, corpus)
    """
    loader = MuSiQueLoader(
        split=split,
        subset_size=subset_size,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    return loader.load()
