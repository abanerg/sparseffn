import numpy as np
import torch

def analyze_sign_distribution(embeddings):
    """
    Calculate the average relative distribution of positive vs negative elements in a list of embeddings.

    Args:
        embeddings: List of embedding tensors or numpy arrays

    Returns:
        dict: Contains percentages of positive, negative, and zero values
    """
    positive_percentages = []
    negative_percentages = []
    zero_percentages = []

    for embedding in embeddings:
        # Convert to numpy if it's a torch tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().to(torch.float16).numpy()

        # Count positive, negative, and zero elements
        total_elements = embedding.size
        positive_count = np.sum(embedding > 0)
        negative_count = np.sum(embedding < 0)
        zero_count = np.sum(embedding == 0)

        # Calculate percentages
        positive_percentages.append(positive_count / total_elements * 100)
        negative_percentages.append(negative_count / total_elements * 100)
        zero_percentages.append(zero_count / total_elements * 100)

    # Calculate averages across all embeddings
    avg_positive = np.mean(positive_percentages)
    avg_negative = np.mean(negative_percentages)
    avg_zero = np.mean(zero_percentages)

    return {
        "positive_percentage": avg_positive,
        "negative_percentage": avg_negative,
        "zero_percentage": avg_zero,
        "positive_to_negative_ratio": avg_positive / avg_negative if avg_negative > 0 else float('inf')
    }

def find_percentile_threshold(embeddings, percentile):
    """
    Find the average value at the bottom of a specific percentile across embeddings.

    Args:
        embeddings: List of embedding tensors or numpy arrays
        percentile: Percentile value (0-100)

    Returns:
        dict: Contains threshold values for absolute, positive-only, and negative-only distributions
    """
    abs_thresholds = []
    pos_thresholds = []
    neg_thresholds = []

    for embedding in embeddings:
        # Convert to numpy if it's a torch tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().to(torch.float16).numpy()

        # Absolute value threshold
        abs_threshold = np.percentile(np.abs(embedding), percentile)
        abs_thresholds.append(abs_threshold)

        # Positive values threshold
        positive_values = embedding[embedding > 0]
        if len(positive_values) > 0:
            pos_threshold = np.percentile(positive_values, percentile)
            pos_thresholds.append(pos_threshold)

        # Negative values threshold (absolute values of negatives)
        negative_values = np.abs(embedding[embedding < 0])
        if len(negative_values) > 0:
            neg_threshold = np.percentile(negative_values, percentile)
            neg_thresholds.append(neg_threshold)

    return {
        "absolute_threshold": np.mean(abs_thresholds),
        "positive_threshold": np.mean(pos_thresholds) if pos_thresholds else None,
        "negative_threshold": np.mean(neg_thresholds) if neg_thresholds else None
    }

dirpath = "/mnt/storage/spffn/metrics/inferences/mlp/"
dists = {x: [] for x in range(32)}
at = {x: [] for x in range(32)}
pt = {x: [] for x in range(32)}
nt = {x: [] for x in range(32)}

for i in range(50):
    print(i)
    data = torch.load(dirpath + f"prompt_{i}.pt")
    for l in range(32):
        embeddings = [data[l][j]["act_fn output"] for j in range(len(data[l]))]
        dists[l].append(analyze_sign_distribution(embeddings)["positive_to_negative_ratio"])
        threshold = find_percentile_threshold(embeddings, 55)
        at[l].append(threshold["absolute_threshold"])
        pt[l].append(threshold["positive_threshold"])
        nt[l].append(threshold["negative_threshold"])

for l in range(32):
    print(f"On layer {l}")
    print(f"Average distributions: {np.mean(dists[l])}")
    print(f"Abs threshold: {np.mean(at[l])}")
    print(f"positive threshold: {np.mean(pt[l])}")
    print(f"Negative threshold: {np.mean(nt[l])}")

torch.save(dists, "./dists.pt")
torch.save(at, "./at.pt")
torch.save(pt, "./pt.pt")
torch.save(nt, "./nt.pt")