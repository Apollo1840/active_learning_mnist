import torch
from toma import toma
from tqdm.auto import tqdm
import math


def compute_conditional_entropy(log_probs_N_K_C):
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)  # using cuda to compute
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C):
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def get_bald_batch(probs_N_K_C, batch_size: int):
    log_probs_N_K_C = torch.from_numpy(probs_N_K_C).log().double()

    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)  # top-K

    return candiate_scores.tolist(), candidate_indices.tolist()
