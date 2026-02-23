"""HotpotQA dataset loader."""

import os
from pathlib import Path

from datasets import DownloadConfig, load_dataset

from ..core.types import Document, Question, QuestionType


class HotpotQALoader:
    """Loader for the HotpotQA multi-hop QA dataset.

    Supports two settings:
    - distractor: 10 paragraphs provided (2 gold + 8 distractors)
    - fullwiki: Open retrieval from Wikipedia (not supported yet)
    """

    def __init__(
        self,
        setting: str = "distractor",
        split: str = "validation",
        subset_size: int | None = None,
        local_files_only: bool | None = None,
        cache_dir: str | None = None,
    ):
        """Initialize the loader.

        Args:
            setting: 'distractor' or 'fullwiki'
            split: Dataset split ('train' or 'validation')
            subset_size: Optional limit on number of questions
            local_files_only: Whether to force local cached dataset usage only
            cache_dir: Optional HuggingFace datasets cache directory
        """
        self.setting = setting
        self.split = split
        self.subset_size = subset_size
        if local_files_only is None:
            env_value = os.getenv("HOTPOTQA_LOCAL_ONLY", "1").strip().lower()
            self.local_files_only = env_value in {"1", "true", "yes", "on"}
        else:
            self.local_files_only = local_files_only

        if cache_dir is None:
            cache_dir = os.getenv(
                "HOTPOTQA_CACHE_DIR",
                str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface" / "datasets"),
            )
        self.cache_dir = cache_dir

    def _parse_question_type(self, type_str: str) -> QuestionType:
        """Parse HotpotQA question type to enum.

        Args:
            type_str: 'bridge' or 'comparison'

        Returns:
            QuestionType enum value
        """
        type_map = {
            "bridge": QuestionType.BRIDGE,
            "comparison": QuestionType.COMPARISON,
        }
        return type_map.get(type_str.lower(), QuestionType.BRIDGE)

    def load(self) -> tuple[list[Question], list[Document]]:
        """Load the HotpotQA dataset.

        Returns:
            Tuple of (questions, corpus)
            - questions: List of Question objects
            - corpus: List of all Document objects from contexts
        """
        # Load from local HuggingFace cache by default to avoid network dependency.
        download_config = DownloadConfig(local_files_only=self.local_files_only)
        dataset = load_dataset(
            "hotpot_qa",
            self.setting,
            split=self.split,
            cache_dir=self.cache_dir,
            download_config=download_config,
        )

        # Apply subset if specified
        if self.subset_size:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))

        questions = []
        corpus_dict: dict[str, Document] = {}

        for item in dataset:
            # Parse supporting facts
            supporting_facts = None
            if "supporting_facts" in item and item["supporting_facts"]:
                sf = item["supporting_facts"]
                if isinstance(sf, dict) and "title" in sf and "sent_id" in sf:
                    supporting_facts = list(zip(sf["title"], sf["sent_id"]))

            # Create Question
            q = Question(
                id=item["id"],
                text=item["question"],
                type=self._parse_question_type(item.get("type", "bridge")),
                gold_answer=item["answer"],
                supporting_facts=supporting_facts,
                metadata={
                    "level": item.get("level", ""),
                },
            )
            questions.append(q)

            # Parse context to build corpus
            context = item.get("context", {})
            if isinstance(context, dict):
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])

                for title, sentences in zip(titles, sentences_list):
                    doc_id = f"{item['id']}_{title}"

                    if doc_id not in corpus_dict:
                        # Join sentences to form document text
                        text = " ".join(sentences) if isinstance(sentences, list) else sentences

                        corpus_dict[doc_id] = Document(
                            id=doc_id,
                            title=title,
                            text=text,
                            sentences=sentences if isinstance(sentences, list) else [sentences],
                        )

        return questions, list(corpus_dict.values())


def load_hotpotqa(
    setting: str = "distractor",
    split: str = "validation",
    subset_size: int | None = None,
    local_files_only: bool | None = None,
    cache_dir: str | None = None,
) -> tuple[list[Question], list[Document]]:
    """Convenience function to load HotpotQA.

    Args:
        setting: 'distractor' or 'fullwiki'
        split: Dataset split
        subset_size: Optional limit
        local_files_only: Whether to force local cached dataset usage only
        cache_dir: Optional HuggingFace datasets cache directory

    Returns:
        Tuple of (questions, corpus)
    """
    loader = HotpotQALoader(
        setting=setting,
        split=split,
        subset_size=subset_size,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    return loader.load()
