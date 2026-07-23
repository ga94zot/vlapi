"""
Convert the modified_libero_rlds dataset (already downloaded on this server) to
LeRobot format.

Forked from convert_libero_data_to_lerobot.py so the original example script is
left untouched. This version pins this server's dataset paths as defaults
(both can still be overridden on the command line) and writes to an explicit
output directory instead of relying on the HF_LEROBOT_HOME env var.

Usage:
uv sync --group rlds
uv run --group rlds examples/libero/convert_libero_data_to_lerobot_local.py
uv run --group rlds examples/libero/convert_libero_data_to_lerobot_local.py \
    --data_dir /other/raw/data --output_dir /other/output/dir

If you want to push your dataset to the Hugging Face Hub, you can add --push_to_hub.

Note: to run the script, you need tensorflow_datasets, which is pulled in by
the `rlds` dependency group above (or install manually with
`uv pip install tensorflow tensorflow_datasets`).
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

# Raw modified_libero_rlds dataset location on this server.
DEFAULT_DATA_DIR = "~/datasets/modified_libero_rlds"
# Where to write the converted LeRobot dataset on this server.
DEFAULT_OUTPUT_DIR = "~/datasets/lerobot/libero"

REPO_NAME = "yuankai/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str = DEFAULT_DATA_DIR, output_dir: str = DEFAULT_OUTPUT_DIR, *, push_to_hub: bool = False):
    data_dir = str(Path(data_dir).expanduser())
    output_path = Path(output_dir).expanduser() / REPO_NAME

    # Clean up any existing dataset in the output directory
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
