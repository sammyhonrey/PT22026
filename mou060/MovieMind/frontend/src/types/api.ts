/** Types copied from the verified FastAPI Pydantic response schemas. */

export type ApiErrorCode =
  | "conflict"
  | "database_unavailable"
  | "forbidden"
  | "internal_error"
  | "model_unavailable"
  | "not_found"
  | "not_implemented"
  | "rate_limited"
  | "unauthorized"
  | "validation_error";

export type ApiErrorResponse = {
  code: ApiErrorCode;
  detail: string;
};

export type Genre = { id: number; name: string };

export type Movie = {
  average_rating: number | null;
  genres: string[];
  id: string;
  movielens_id: number | null;
  overview: string | null;
  poster_url: string | null;
  ratings_count: number | null;
  release_year: number | null;
  title: string;
};

export type OffsetPage<T> = {
  items: T[];
  limit: number;
  offset: number;
  total: number;
};

export type MovieList = OffsetPage<Movie>;

export type User = {
  created_at: string;
  display_name: string | null;
  email: string;
  id: string;
  is_active: boolean;
  role: string;
  username: string | null;
};

export type TokenResponse = {
  access_token: string;
  csrf_token: string;
  expires_in: number;
  token_type?: string;
};

export type RegisterResponse = TokenResponse & { user: User };
export type LoginResponse = TokenResponse & { user: User };
export type CsrfTokenResponse = { csrf_token: string | null };

export type Rating = {
  created_at: string;
  id: string;
  movie: Movie;
  movie_id: string;
  score: number;
  updated_at: string;
};

export type RatingList = OffsetPage<Rating>;

export type WatchlistItem = { created_at: string; id: string; movie: Movie };
export type Watchlist = OffsetPage<WatchlistItem>;

export type Recommendation = {
  movie: Movie;
  reason: string;
  score: number;
  source: string;
};

export type Recommendations = {
  is_fallback: boolean;
  items: Recommendation[];
  limit: number;
  source: string;
};

export type UserPreference = {
  user_id: string;
  favorite_genres: string[];
  favorite_movie_ids: string[];
  description: string | null;
  created_at: string;
  updated_at: string;
};
