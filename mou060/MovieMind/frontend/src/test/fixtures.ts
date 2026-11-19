import type { Movie, OffsetPage, Rating, User, WatchlistItem } from "@/types/api";

export const testUser: User = {
  created_at: "2025-01-15T10:00:00Z",
  display_name: "Test Viewer",
  email: "viewer@example.com",
  id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  is_active: true,
  role: "user",
  username: "viewer",
};

export const dune: Movie = {
  average_rating: 8.4,
  genres: ["Science Fiction", "Adventure"],
  id: "11111111-1111-4111-8111-111111111111",
  movielens_id: 1,
  overview: "Paul Atreides travels to the desert planet Arrakis to protect his family and his people.",
  poster_url: "https://images.example.com/dune.jpg",
  ratings_count: 120,
  release_year: 2021,
  title: "Dune",
};

export const batman: Movie = {
  average_rating: 8.2,
  genres: ["Action", "Crime"],
  id: "22222222-2222-4222-8222-222222222222",
  movielens_id: 2,
  overview: "Bruce Wayne becomes Batman and confronts the crime that shaped Gotham.",
  poster_url: null,
  ratings_count: 98,
  release_year: 2005,
  title: "Batman Begins",
};

export const interstellar: Movie = {
  average_rating: 8.7,
  genres: ["Science Fiction", "Drama"],
  id: "33333333-3333-4333-8333-333333333333",
  movielens_id: 3,
  overview: "A team of explorers travels through a wormhole in an attempt to save humanity.",
  poster_url: "https://images.example.com/interstellar.jpg",
  ratings_count: 145,
  release_year: 2014,
  title: "Interstellar",
};

export const unratedMovie: Movie = {
  average_rating: null,
  genres: [],
  id: "44444444-4444-4444-8444-444444444444",
  movielens_id: null,
  overview: null,
  poster_url: null,
  ratings_count: null,
  release_year: null,
  title: "Unreviewed Short Film",
};

export const testMovies = [dune, batman, interstellar, unratedMovie];

export const genres = [
  { id: 1, name: "Action" },
  { id: 2, name: "Drama" },
  { id: 3, name: "Science Fiction" },
];

export function pageOf<T>(items: T[], limit = 20, offset = 0): OffsetPage<T> {
  return { items: items.slice(offset, offset + limit), limit, offset, total: items.length };
}

export function makeRating(movie: Movie, score: number): Rating {
  return {
    created_at: "2025-02-01T12:00:00Z",
    id: `rating-${movie.id}`,
    movie,
    movie_id: movie.id,
    score,
    updated_at: "2025-02-01T12:00:00Z",
  };
}

export function makeWatchlistItem(movie: Movie): WatchlistItem {
  return {
    created_at: "2025-02-01T12:00:00Z",
    id: `watchlist-${movie.id}`,
    movie,
  };
}
