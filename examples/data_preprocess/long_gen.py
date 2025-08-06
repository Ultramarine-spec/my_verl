# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/public/jhchen/verl/0-verl-data")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/new-length-half-split-2025-07-15-19:11:18"
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    dataset = datasets.load_from_disk(f"/public/jhchen/huggingface/datasets/{data_source}")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            if "refined_instruction" in example:
                query = example["refined_instruction"]
            elif "instruction" in example:
                query = example["instruction"]
            elif "query_with_length_request" in example:
                query = example["query_with_length_request"]
            elif "query" in example:
                query = example["query"]
            else:
                query = example["conversation"][0]['content']
            language = example['language']
            checklist = example["checklist"]
            if 'generation_length' in example and example['generation_length'] != [-1, -1]:
                length = example['generation_length']
            else:
                length = None

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": query}],
                "reward_model": {"style": "llm"},
                "extra_info": {"split": split, "index": idx, "checklist": checklist, "length": length, 'language': language},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset = train_dataset.select_columns(
        ["data_source", "prompt", "reward_model", "extra_info"]
    )
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset = test_dataset.select_columns(
        ["data_source", "prompt", "reward_model", "extra_info"]
    )

    print(train_dataset)
    print(train_dataset[0])
    # print(train_dataset['generation_length'])
    # import pdb; pdb.set_trace()

    local_dir = f"{args.local_dir}/{data_source}"
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
