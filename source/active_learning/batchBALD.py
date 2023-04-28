import torch
from toma import toma
from tqdm.auto import tqdm
import math

from .joint_entropy import *
from .BALD import compute_conditional_entropy


def get_batchbald_batch(log_probs_N_K_C: torch.Tensor,
                        query_batch_size: int,
                        num_samples: int=1000,
                        dtype=None,
                        device=None):
    """
    Select a batch of points to acquire using BatchBALD algorithm.

    batch_joint_entropy is the key of the trick introduced in the paper

    Args:
        log_probs_N_K_C (torch.Tensor): Log probabilities of data points (N) with samples (K) and classes (C). Shape: (N, K, C)
        query_batch_size (int): Number of data points to select in the batch.
        num_samples (int): Number of MC samples used for approximation.
        dtype (Optional[torch.dtype]): Data type for the tensors, defaults to None.
        device (Optional[torch.device]): Device to perform computations on, defaults to None.

    Returns:
        CandidateBatch: A tuple containing the scores and indices of the selected data points.
    """

    # Get the dimensions of log probabilities tensor
    N, K, C = log_probs_N_K_C.shape

    # Ensure query_batch_size doesn't exceed the number of data points
    query_batch_size = min(query_batch_size, N)

    # Initialize lists to store candidate indices and scores
    candidate_indices = []
    candidate_scores = []

    # Compute the conditional entropies for all data points
    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    # Initialize batch_joint_entropy object, which will iteratively updated during the loop
    batch_joint_entropy = DynamicJointEntropy(num_samples, query_batch_size - 1, K, C,
                                              dtype=dtype, device=device)

    # Create an empty tensor to store scores
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    # Iterate through the query batch size
    for i in tqdm(range(query_batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            # If not the first iteration, add the latest index to the batch_joint_entropy object
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index: latest_index + 1])

        # Compute the sum of conditional entropies for already selected indices
        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        # Compute joint entropies for the current batch
        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        # Update scores by subtracting the conditional entropies and shared conditional entropies
        scores_N -= conditional_entropies_N + shared_conditinal_entropies

        # Set scores for already selected indices to -inf
        scores_N[candidate_indices] = -float("inf")

        # Choose the best candidate based on scores_N
        candidate_score, candidate_index = scores_N.max(dim=0)

        # Append the selected candidate's index and score to the lists
        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    # Return the final batch of candidates with their scores and indices
    return candidate_scores, candidate_indices
