# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import List

import torch
from camel.embeddings import OpenAIEmbedding
from camel.types import EmbeddingModelType
from transformers import AutoModel, AutoTokenizer


# Function: Process each batch
@torch.no_grad()
def process_batch(
    model: AutoModel, tokenizer: AutoTokenizer, batch_texts: List[str], device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    return outputs.pooler_output


def generate_post_vector_multi_gpu(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts,
    batch_size,
    gpu_ids=None,
    use_gpu_isolation=True,
):
    """
    Generate post vectors using multiple GPUs by splitting data across devices
    This helps avoid CUDA out of memory errors for large datasets

    Args:
        model: The transformer model (can be None for fresh loading)
        tokenizer: The tokenizer
        texts: List of texts to process
        batch_size: Batch size for processing
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 3]). If None, uses all available GPUs
        use_gpu_isolation: Whether CUDA_VISIBLE_DEVICES isolation was used (affects GPU ID validation)
    """
    import math

    total_gpu_count = torch.cuda.device_count()
    if total_gpu_count <= 1:
        # Fallback to single GPU if only one is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return generate_post_vector(model, tokenizer, texts, batch_size, device=device)

    # Determine which GPUs to use
    if gpu_ids is None:
        gpu_ids = list(range(total_gpu_count))
    else:
        # Validate GPU IDs based on isolation mode
        if use_gpu_isolation:
            # With isolation, validate against visible GPU count (renumbered)
            valid_gpu_ids = []
            for gpu_id in gpu_ids:
                if 0 <= gpu_id < total_gpu_count:
                    valid_gpu_ids.append(gpu_id)
                else:
                    print(
                        f"Warning: GPU {gpu_id} is not available (total visible: {total_gpu_count}), skipping"
                    )
        else:
            # Without isolation, validate against total system GPU count
            import subprocess

            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                system_gpu_count = len(result.stdout.strip().split("\n"))
            except Exception:
                system_gpu_count = total_gpu_count  # Fallback

            valid_gpu_ids = []
            for gpu_id in gpu_ids:
                if 0 <= gpu_id < system_gpu_count:
                    valid_gpu_ids.append(gpu_id)
                else:
                    print(
                        f"Warning: GPU {gpu_id} is not available (total system GPUs: {system_gpu_count}), skipping"
                    )

        gpu_ids = valid_gpu_ids
        if not gpu_ids:
            print("Warning: No valid GPU IDs provided, falling back to single GPU")
            device = torch.device("cuda:0")
            return generate_post_vector(
                model, tokenizer, texts, batch_size, device=device
            )

    isolation_status = (
        "with CUDA_VISIBLE_DEVICES isolation"
        if use_gpu_isolation
        else "without isolation"
    )
    print(f"Using specified GPUs {gpu_ids} for data splitting ({isolation_status})")

    # Split texts across specified GPUs
    gpu_count = len(gpu_ids)
    texts_per_gpu = math.ceil(len(texts) / gpu_count)
    all_outputs = []

    # Create separate model instances ONLY on specified GPUs
    models_per_gpu = {}

    # Clear GPU memory first
    if use_gpu_isolation:
        # In isolation mode, clear all visible GPUs
        for i in range(total_gpu_count):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
    else:
        # In direct mode, clear only specified GPUs
        for gpu_id in gpu_ids:
            try:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: Could not clear GPU {gpu_id}: {e}")

    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * texts_per_gpu
        end_idx = min((i + 1) * texts_per_gpu, len(texts))

        if start_idx >= len(texts):
            break

        gpu_texts = texts[start_idx:end_idx]
        device = torch.device(f"cuda:{gpu_id}")

        print(
            f"Loading model on PyTorch GPU {gpu_id} and processing {len(gpu_texts)} texts"
        )

        # Load model ONLY on this specific GPU
        try:
            # Load a fresh model instance directly on the target GPU
            from transformers import AutoModel

            models_per_gpu[gpu_id] = AutoModel.from_pretrained(
                "Twitter/twhin-bert-base"
            ).to(device)
            print(f"✓ Model loaded successfully on GPU {gpu_id}")
        except Exception as e:
            print(f"✗ Failed to load model on GPU {gpu_id}: {e}")
            continue

        # Process this chunk on the specific GPU
        gpu_outputs = []
        for j in range(0, len(gpu_texts), batch_size):
            batch_texts = gpu_texts[j : j + batch_size]
            batch_outputs = process_batch(
                models_per_gpu[gpu_id], tokenizer, batch_texts, device=device
            )
            gpu_outputs.append(batch_outputs)

        if gpu_outputs:
            gpu_tensor = torch.cat(gpu_outputs, dim=0)
            # Move to CPU to free GPU memory
            all_outputs.append(gpu_tensor.cpu())
            print(f"✓ GPU {gpu_id} processing completed, moved results to CPU")

        # Clear this GPU's cache after processing
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

    # Clear the temporary models to free memory
    for gpu_id in list(models_per_gpu.keys()):
        del models_per_gpu[gpu_id]
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        print(f"✓ Cleared model from GPU {gpu_id}")

    # Concatenate all results
    if all_outputs:
        final_tensor = torch.cat(all_outputs, dim=0)
        print(f"✓ Combined results from {len(gpu_ids)} GPUs into final tensor")
        return final_tensor
    else:
        # Fallback if no outputs generated
        print("Warning: No outputs generated, falling back to single GPU")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return generate_post_vector(model, tokenizer, texts, batch_size, device=device)


def generate_post_vector(
    model: AutoModel, tokenizer: AutoTokenizer, texts, batch_size, device=None
):
    # Loop through all messages
    # If the list of messages is too large, process them in batches.
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_outputs = process_batch(model, tokenizer, batch_texts, device=device)
        all_outputs.append(batch_outputs)
    all_outputs_tensor = torch.cat(all_outputs, dim=0)  # num_posts x dimension
    return all_outputs_tensor.cpu()


def generate_post_vector_openai(texts: List[str], batch_size: int = 100):
    """
    Generate embeddings using OpenAI API

    Args:
        texts: List of texts to process
        batch_size: Size of each batch
    """
    openai_embedding = OpenAIEmbedding(
        model_type=EmbeddingModelType.TEXT_EMBEDDING_3_SMALL
    )

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        cleaned_texts = [
            text.strip() if text and isinstance(text, str) else "empty"
            for text in batch_texts
        ]
        batch_embeddings = openai_embedding.embed_list(objs=cleaned_texts)
        batch_tensor = torch.tensor(batch_embeddings)
        all_embeddings.append(batch_tensor)

    return torch.cat(all_embeddings, dim=0)


if __name__ == "__main__":
    # Input list of strings (assuming there are tens of thousands of messages)
    # Here, the same message is repeated 10000 times as an example
    texts = ["I'm using TwHIN-BERT! #TwHIN-BERT #NLP"] * 10000
    # Define batch size
    batch_size = 100

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base")
    model = AutoModel.from_pretrained("Twitter/twhin-bert-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_outputs_tensor = generate_post_vector(
        model, tokenizer, texts, batch_size, device=device
    )
    print(all_outputs_tensor.shape)
