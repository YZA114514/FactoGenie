import collections
import numpy as np

BETA_START = 0.4
BETA_FRAMES = 100000

Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "done", "new_state"]
)


class ExperienceBuffer:
    def __init__(self, buf_size):
        self.capacity = buf_size
        self.buffer = []
        self.pos = 0
        self.beta = 1.0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}."
            )
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return samples, np.array(indices, dtype=np.int64), weights

    def update_beta(self, *_):
        self.beta = 1.0
        return self.beta

    def update_priorities(self, *_, **__):
        return None


class PrioReplayBuffer:
    def __init__(self, buf_size, prob_alpha=0.6):
        self.capacity = buf_size
        self.prob_alpha = prob_alpha
        self.buffer = []
        self.priorities = np.zeros((buf_size,), dtype=np.float32)
        self.pos = 0
        self.beta = BETA_START

    def __len__(self):
        return len(self.buffer)

    def update_beta(self, idx):
        frac = min(idx / float(BETA_FRAMES), 1.0)
        self.beta = BETA_START + frac * (1.0 - BETA_START)
        return self.beta

    def append(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}."
            )
        prios = self.priorities[: len(self.buffer)]
        probs = prios ** self.prob_alpha
        probs_sum = probs.sum()
        if probs_sum == 0.0:
            probs = np.ones_like(prios) / len(prios)
        else:
            probs /= probs_sum
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = float(np.abs(prio)) + 1e-5