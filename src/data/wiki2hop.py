"""2WikiMultiHopQA dataset loader."""

import os
from pathlib import Path

from datasets import DownloadConfig, load_dataset

from ..core.types import Document, Question, QuestionType


class Wiki2HopLoader:
    """Loader for the 2WikiMultiHopQA multi-hop QA dataset.

    2WikiMultiHopQA requires multi-step reasoning with explicit evidence chains.
    Questions are categorized into 4 types: comparison, inference, bridging, transfer.
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
            split: Dataset split ('train', 'validation', or 'test')
            subset_size: Optional limit on number of questions
            local_files_only: Whether to force local cached dataset usage only
            cache_dir: Optional HuggingFace datasets cache directory
        """
        self.split = split
        self.subset_size = subset_size
        if local_files_only is None:
            env_value = os.getenv("WIKI2HOP_LOCAL_ONLY", "1").strip().lower()
            self.local_files_only = env_value in {"1", "true", "yes", "on"}
        else:
            self.local_files_only = local_files_only

        if cache_dir is None:
            cache_dir = os.getenv(
                "WIKI2HOP_CACHE_DIR",
                str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface" / "datasets"),
            )
        self.cache_dir = cache_dir

    def _parse_question_type(self, type_str: str) -> QuestionType:
        """Parse 2WikiMultiHopQA question type to enum.

        Args:
            type_str: 'comparison', 'inference', 'bridging', or 'transfer'

        Returns:
            QuestionType enum value
        """
        type_map = {
            "comparison": QuestionType.COMPARISON,
            "bridging": QuestionType.BRIDGE,
            "inference": QuestionType.COMPOSITIONAL,
            "transfer": QuestionType.COMPOSITIONAL,
        }
        return type_map.get(type_str.lower(), QuestionType.BRIDGE)

    def load(self) -> tuple[list[Question], list[Document]]:
        """Load the 2WikiMultiHopQA dataset.

        Returns:
            Tuple of (questions, corpus)
            - questions: List of Question objects
            - corpus: List of all Document objects from contexts
        """
        download_config = DownloadConfig(local_files_only=self.local_files_only)
        dataset = load_dataset(
            "xanhho/2WikiMultihopQA",
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

            supporting_facts = None
            if "supporting_facts" in item and item["supporting_facts"]:
                sf = item["supporting_facts"]
                if isinstance(sf, dict) and "title" in sf and "sent_id" in sf:
                    supporting_facts = list(zip(sf["title"], sf["sent_id"]))

            q = Question(
                id=q_id,
                text=item["question"],
                type=self._parse_question_type(item.get("type", "bridging")),
                gold_answer=item["answer"],
                supporting_facts=supporting_facts,
                metadata={
                    "evidences": item.get("evidences", []),
                },
            )
            questions.append(q)

            context = item.get("context", {})
            if isinstance(context, dict):
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])

                for title, sentences in zip(titles, sentences_list):
                    doc_id = f"{q_id}_{title}"

                    if doc_id not in corpus_dict:
                        text = " ".join(sentences) if isinstance(sentences, list) else sentences

                        corpus_dict[doc_id] = Document(
                            id=doc_id,
                            title=title,
                            text=text,
                            sentences=sentences if isinstance(sentences, list) else [sentences],
                        )

        return questions, list(corpus_dict.values())


def load_2wiki(
    split: str = "validation",
    subset_size: int | None = None,
    local_files_only: bool | None = None,
    cache_dir: str | None = None,
) -> tuple[list[Question], list[Document]]:
    """Convenience function to load 2WikiMultiHopQA.

    Args:
        split: Dataset split
        subset_size: Optional limit
        local_files_only: Whether to force local cached dataset usage only
        cache_dir: Optional HuggingFace datasets cache directory

    Returns:
        Tuple of (questions, corpus)
    """
    loader = Wiki2HopLoader(
        split=split,
        subset_size=subset_size,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    return loader.load()
