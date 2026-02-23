"""Tests for Planner RAG prompt templates."""

from pathlib import Path


def test_planner_prompt_files_exist() -> None:
    assert Path("prompts/planner_action.txt").exists()
    assert Path("prompts/planner_solve.txt").exists()
    assert Path("prompts/planner_synthesize.txt").exists()
    assert Path("prompts/planner_bridge_refine.txt").exists()


def test_planner_action_prompt_mentions_actions() -> None:
    prompt = Path("prompts/planner_action.txt").read_text()
    for action in ["TRAVERSE", "SELECT", "ROLLOUT", "BACKTRACK", "STOP"]:
        assert action in prompt
