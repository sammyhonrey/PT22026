import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Pencil, UserRound } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { updateCurrentUser } from "@/features/auth/api";
import { useAuth } from "@/features/auth/AuthProvider";
import { isApiClientError } from "@/lib/api/errors";
import { queryKeys } from "@/lib/query/keys";

const usernamePattern = /^[a-zA-Z0-9_.-]+$/;

export function ProfilePage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [displayName, setDisplayName] = useState("");
  const [username, setUsername] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const mutation = useMutation({
    mutationFn: updateCurrentUser,
    onSuccess: (updated) => queryClient.setQueryData(queryKeys.user.me(), updated),
  });

  useEffect(() => {
    setDisplayName(user?.display_name ?? "");
    setUsername(user?.username ?? "");
  }, [user?.display_name, user?.username]);

  if (!user) return <StatusPanel>Loading your profile…</StatusPanel>;
  const initials = (user.display_name ?? user.username ?? user.email).slice(0, 1).toUpperCase();

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSaved(false);
    const normalizedUsername = username.trim();
    if (normalizedUsername && (normalizedUsername.length < 3 || normalizedUsername.length > 50 || !usernamePattern.test(normalizedUsername))) {
      setError("Use a username with 3–50 letters, numbers, dots, underscores, or hyphens.");
      return;
    }
    if (displayName.trim().length > 100) {
      setError("Display name cannot exceed 100 characters.");
      return;
    }
    try {
      await mutation.mutateAsync({ display_name: displayName.trim() || undefined, username: normalizedUsername || undefined });
      setSaved(true);
    } catch (requestError) {
      setError(isApiClientError(requestError) ? requestError.message : "We could not update your profile. Please try again.");
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-7"><header><p className="text-sm font-bold text-[var(--mm-accent)]">Account</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">My Profile</h1><p className="mt-2 text-[var(--mm-text-muted)]">Update the profile details supported by MovieMind.</p></header>
      <div className="grid gap-6 md:grid-cols-[minmax(14rem,.7fr)_minmax(0,1.3fr)]"><section className="rounded-3xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6 shadow-[var(--mm-card-shadow)]"><div className="grid size-16 place-items-center rounded-full bg-linear-to-br from-[var(--mm-primary)] to-[var(--mm-primary-deep)] text-2xl font-black">{initials}</div><h2 className="mt-5 text-xl font-bold">{user.display_name ?? user.username ?? "MovieMind member"}</h2><p className="mt-1 text-sm text-[var(--mm-text-muted)]">{user.email}</p><p className="mt-5 text-xs text-[var(--mm-text-subtle)]">Member since {new Date(user.created_at).toLocaleDateString()}</p></section>
        <form className="rounded-3xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6 shadow-[var(--mm-card-shadow)]" onSubmit={submit}><div className="flex items-center gap-2"><Pencil aria-hidden="true" className="size-5 text-[var(--mm-accent)]" /><h2 className="text-xl font-bold">Edit details</h2></div><div className="mt-6 space-y-5"><label className="block text-sm font-semibold">Display name<Input className="mt-2" maxLength={100} onChange={(event) => setDisplayName(event.target.value)} value={displayName} /></label><label className="block text-sm font-semibold">Username<Input className="mt-2" maxLength={50} onChange={(event) => setUsername(event.target.value)} value={username} /></label><label className="block text-sm font-semibold text-[var(--mm-text-muted)]">Email address<Input className="mt-2" disabled value={user.email} /></label></div>{error && <p className="mm-content-enter mt-5 text-sm text-[var(--mm-danger)]" role="alert">{error}</p>}{saved && <p className="mm-content-enter mt-5 text-sm text-[var(--mm-success)]" role="status">Profile updated.</p>}<Button className="mt-6" isLoading={mutation.isPending} type="submit"><UserRound aria-hidden="true" className="size-4" /> Save changes</Button></form>
      </div></div>
  );
}
