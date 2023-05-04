import torch
import torch.nn as nn
from torch.optim import Optimizer


class RSGD(Optimizer):
    """
    Riemannian stochastic gradient descent (RSGD) optimizer.
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        defaults = dict(lr=lr)
        super(RSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Compute the Riemannian gradient
                rgrad = self._riemannian_gradient(p)

                # Update the model parameters using the Riemannian exponential map
                p.add_(self._riemannian_exponential_map(p, -group['lr'] * rgrad))

    @staticmethod
    def _riemannian_gradient(p):
        grad = p.grad.data
        p_sqnorm = p.data.pow(2).sum(dim=-1, keepdim=True)
        return (1 - p_sqnorm).pow(2) * grad

    @staticmethod
    def _riemannian_exponential_map(p, u):
        p_sqnorm = p.data.pow(2).sum(dim=-1, keepdim=True)
        u_sqnorm = u.pow(2).sum(dim=-1, keepdim=True)
        second_term = ((1 - p_sqnorm).sqrt() * (u_sqnorm / (1 - p_sqnorm)).atanh() * u).clamp(
            max=(1 - p_sqnorm).sqrt() - 1e-8)
        return second_term


class PoincareEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(PoincareEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.embedding.weight.data.uniform_(-0.001, 0.001)

    def forward(self, indices):
        return self.embedding(indices)

    @staticmethod
    def poincare_distance(u, v):
        norm_u = u.norm(dim=-1, keepdim=True)
        norm_v = v.norm(dim=-1, keepdim=True)
        euclidean_squared = (u - v).norm(dim=-1).pow(2)
        return torch.acosh(1 + 2 * euclidean_squared / ((1 - norm_u.pow(2)) * (1 - norm_v.pow(2))))

    @staticmethod
    def poincare_loss(u, v, eps=1e-6):
        sqdist = torch.norm(u - v, dim=-1) ** 2
        denom_u = 1 - u.norm(dim=-1) ** 2
        denom_v = 1 - v.norm(dim=-1) ** 2

        num = sqdist + 2 * eps
        denom = denom_u * denom_v + eps

        acosh_arg = 1 + 2 * num / denom
        acosh_arg = torch.clamp(acosh_arg, min=1.0)  # Ensure the input value to acosh is >= 1

        dist = torch.acosh(acosh_arg)
        return dist.mean()


class EuclideanEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(EuclideanEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, indices):
        return self.embedding(indices)

    @staticmethod
    def euclidean_distance(u, v):
        return torch.norm(u - v, dim=-1)

    @staticmethod
    def euclidean_loss(u, v):
        dist = EuclideanEmbedding.euclidean_distance(u, v)
        return dist.mean()


def train_embeddings(model, optimizer, train_data, loss_function, max_norm, num_epochs=1):
    """
    Trains the given model using the given optimizer and loss function.
    """
    for epoch in range(num_epochs):
        total_loss = 0
        for u_idx, v_idx in train_data:
            optimizer.zero_grad()
            u = model(torch.tensor([u_idx]))
            v = model(torch.tensor([v_idx]))
            loss = loss_function(u, v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}')
