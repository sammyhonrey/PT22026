import torch
import torch.nn as nn


class MovieMindNCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 32,
        mlp_hidden_dims=(128, 64, 32),
        dropout: float = 0.10,
    ):
        super().__init__()
        self.user_gmf = nn.Embedding(num_users, embedding_dim)
        self.movie_gmf = nn.Embedding(num_movies, embedding_dim)
        self.user_mlp = nn.Embedding(num_users, embedding_dim)
        self.movie_mlp = nn.Embedding(num_movies, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in mlp_hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(embedding_dim + input_dim, 1)

    def forward(self, user_indices, movie_indices):
        gmf = self.user_gmf(user_indices) * self.movie_gmf(movie_indices)
        mlp_input = torch.cat([self.user_mlp(user_indices), self.movie_mlp(movie_indices)], dim=1)
        mlp_vector = self.mlp(mlp_input)
        return self.output(torch.cat([gmf, mlp_vector], dim=1)).squeeze(1)
