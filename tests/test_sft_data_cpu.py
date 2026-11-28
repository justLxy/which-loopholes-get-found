import json

from src.train.sft import load_dataset


def test_sft_dataset_keeps_prompt_completion_pairs_without_training(tmp_path):
    records = [
        {"prompt": [{"role": "user", "content": "2+2?"}], "completion": [{"role": "assistant", "content": "4"}],
         "uid": "a"},
        {"prompt": [{"role": "user", "content": "3+3?"}], "completion": [{"role": "assistant", "content": "6"}],
         "uid": "b"},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    dataset, original = load_dataset(path)

    assert original == records and len(dataset) == 2
    assert set(dataset.column_names) == {"prompt", "completion"}
    assert dataset[0]["prompt"][0]["content"] == "2+2?"
    assert dataset[1]["completion"][0]["content"] == "6"
