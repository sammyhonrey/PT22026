import posterData from "@/data/movie-posters.json";

type MoviePosterRecord = {
  movieId: number;
  tmdbId: number | null;
  posterUrl: string | null;
};

const moviePosters: MoviePosterRecord[] = posterData.moviePosters;
const posterUrlsByMovieId = new Map<number, string | null>(
  moviePosters.map(({ movieId, posterUrl }) => [
    movieId,
    posterUrl,
  ]),
);

export function getMoviePosterUrl(movieId: number | null): string | null {
  if (movieId === null || !Number.isSafeInteger(movieId)) return null;
  const posterUrl = posterUrlsByMovieId.get(movieId);
  if (!posterUrl) return null;

  try {
    const url = new URL(posterUrl);
    return url.protocol === "https:" &&
      url.hostname === "image.tmdb.org" &&
      url.pathname.startsWith("/t/p/")
      ? posterUrl
      : null;
  } catch {
    return null;
  }
}
