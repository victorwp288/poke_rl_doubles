import json
from pathlib import Path
from typing import Any


def create_tiny_jsonl_dataset(
    tmp_path: Path, num_samples: int = 5
) -> tuple[list[dict[str, Any]], Path]:
    """
    Helper that creates a tiny JSONL dataset for testing.
    """

    samples = []

    # - Feature 0: always binary
    # - Feature 1: non-binary continuous values
    # - Feature 2: another non-binary value
    # - Feature 3: always binary

    for i in range(num_samples):
        observation = [
            float(i % 2),  # Binary: alternates 0, 1, 0, 1, 0
            float(i * 0.5 + 0.1),  # Non-binary: 0.1, 0.6, 1.1, 1.6, 2.1
            float(i + 10.0),  # Non-binary: 10.0, 11.0, 12.0, 13.0, 14.0
            1.0 if i > 2 else 0.0,  # Binary: 0, 0, 0, 1, 1
        ]

        action = [0, 1]

        mask = [[1, 1, 1, 1], [1, 1, 1, 1]]

        sample_dict = {"observation": observation, "action": action, "mask": mask}

        samples.append(sample_dict)

    # Write to JSONL file
    dataset_path = tmp_path / "test_dataset.jsonl"

    with dataset_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples, dataset_path
