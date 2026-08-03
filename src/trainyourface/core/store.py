"""Enrollment store: persisted embeddings and identity matching.

Three real bugs in the old implementation motivated a rewrite rather than a patch:

1. **Reload-per-frame.** `find_face_match()` opened and JSON-parsed the entire
   descriptor database on *every call*, and it was called once per detected face
   per frame. With 2 faces at 30fps that's 60 full file reads and JSON parses per
   second, and cost grew linearly with enrollment count. Embeddings are now held
   in a contiguous NumPy matrix in memory, and matching is one vectorized matmul.

2. **Unbounded per-name growth.** `train_face_images()` appended a descriptor for
   every detection in every photo, forever. Nothing deduplicated or capped, so
   repeated enrollment inflated the file and slowed every subsequent match.

3. **Silent data loss.** `register_face()` with purpose="initial reg" did
   `self.face_descriptors[face_name] = []`, unconditionally wiping existing
   embeddings for an already-registered name. And because `write_json_*` was
   called from a `threading.Thread` while the main loop also wrote, a concurrent
   write could truncate the file. Writes here are atomic (temp + rename) and
   append-only with respect to existing identities.

Storage format is `.npz`: a single float32 matrix plus a parallel array of names.
JSON was storing 512 floats per embedding as decimal text — roughly 10x the size
of the binary form, parsed on every access.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from platformdirs import user_data_dir

from trainyourface.core.contracts import EMBEDDING_DIM, Identity
from trainyourface.core.embed import l2_normalize

# Default cosine-similarity threshold for accepting a match.
#
# 0.36 is the operating point insightface publishes for ArcFace w600k-family
# models at a ~1e-4 false-match rate on standard benchmarks. It is NOT tuned on
# your enrollments — `tyf calibrate` measures the right value for your own data
# and writes it into the store. Shipping a benchmark-derived default and telling
# users to calibrate is more honest than shipping a number tuned on one face and
# implying it generalizes.
DEFAULT_MATCH_THRESHOLD = 0.36

# Cap on stored embeddings per identity. Past roughly this many, additional
# samples of the same face contribute almost nothing to max-similarity matching
# while costing memory and match time on every frame.
MAX_EMBEDDINGS_PER_IDENTITY = 40

# Near-duplicate cutoff. Two embeddings this similar carry redundant information
# (typically consecutive video frames), so the second is dropped at enrollment.
DEDUP_SIMILARITY = 0.98


def default_store_path() -> Path:
    """Store location. `TYF_DATA_DIR` overrides.

    Deliberately outside the repo: these are biometric templates, i.e. PII. The
    old version wrote them into the source tree and relied on .gitignore to keep
    them out of git — one `git add -f` from being published.
    """
    override = os.environ.get("TYF_DATA_DIR")
    base = Path(override).expanduser() if override else Path(user_data_dir("trainyourface"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "enrollments.npz"


class EnrollmentStore:
    """In-memory embedding index with atomic persistence."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        # (N, 512) float32, L2-normalized. Contiguous so matching is one matmul.
        self._embeddings: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        # Parallel to rows of _embeddings.
        self._names: list[str] = []
        # Per-identity metadata (relationship, enrollment time, sample count).
        self._meta: dict[str, dict] = {}
        self.match_threshold: float = DEFAULT_MATCH_THRESHOLD
        if self.path.exists():
            self.load()

    # ---- persistence -------------------------------------------------------

    def load(self) -> None:
        """Load from disk, tolerating a missing or corrupt file.

        A corrupt store must not be fatal: the tool should still run and let the
        user re-enroll, which is why this warns and starts empty rather than
        raising. But it does NOT silently delete the bad file.

        A *missing* file is not corruption — it's a fresh install. `tyf list` and
        `tyf forget` call load() explicitly, so without this check the first thing
        a new user saw was a warning that their enrollment store could not be read.
        """
        if not self.path.exists():
            return

        try:
            with np.load(self.path, allow_pickle=False) as data:
                emb = data["embeddings"].astype(np.float32)
                names = [str(n) for n in data["names"]]
                meta_raw = str(data["meta"].item()) if "meta" in data else "{}"
                threshold = float(data["threshold"].item()) if "threshold" in data else None
        except (OSError, ValueError, KeyError, EOFError) as exc:
            print(f"warning: could not read enrollment store at {self.path}: {exc}")
            print("starting with an empty store; existing file left untouched")
            return

        if emb.ndim != 2 or (emb.size and emb.shape[1] != EMBEDDING_DIM):
            print(
                f"warning: store has {emb.shape} embeddings, expected (N, {EMBEDDING_DIM}). "
                "This usually means it was built with a different model. Ignoring."
            )
            return
        if len(names) != emb.shape[0]:
            print("warning: store is inconsistent (names/embeddings length mismatch). Ignoring.")
            return

        self._embeddings = emb
        self._names = names
        try:
            self._meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            self._meta = {}
        if threshold is not None:
            self.match_threshold = threshold

    def save(self) -> None:
        """Persist atomically: write a temp file, then rename over the target.

        `Path.replace` is atomic on POSIX and Windows, so a crash or concurrent
        reader either sees the old complete file or the new complete file — never
        a truncated one. The old code's bare `open(path, "w")` from a thread could
        leave a half-written JSON file that failed to parse on next start.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".npz.tmp")
        # Write through an open handle rather than passing the path: given a path
        # not ending in ".npz", np.savez_compressed helpfully appends the
        # extension, so a "foo.npz.tmp" target silently becomes
        # "foo.npz.tmp.npz" and the rename below fails. A file object is taken
        # literally.
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                embeddings=self._embeddings,
                names=np.array(self._names, dtype=object).astype("U"),
                meta=np.array(json.dumps(self._meta)),
                threshold=np.array(self.match_threshold),
            )
        tmp.replace(self.path)

    # ---- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return self._embeddings.shape[0]

    @property
    def is_empty(self) -> bool:
        return self._embeddings.shape[0] == 0

    @property
    def identities(self) -> list[str]:
        return sorted(self._meta.keys())

    def count_for(self, name: str) -> int:
        return sum(1 for n in self._names if n == name)

    def summary(self) -> list[dict]:
        return [
            {
                "name": name,
                "samples": self.count_for(name),
                "relationship": self._meta.get(name, {}).get("relationship"),
                "enrolled_at": self._meta.get(name, {}).get("enrolled_at"),
            }
            for name in self.identities
        ]

    # ---- mutation ----------------------------------------------------------

    def add(
        self,
        name: str,
        embeddings: np.ndarray,
        *,
        relationship: str | None = None,
    ) -> int:
        """Add embeddings for an identity. Returns how many were actually kept.

        Additive by design: enrolling an existing name augments that identity
        rather than replacing it. Near-duplicates and over-cap samples are
        dropped, and the count of kept samples is returned so the CLI can tell
        the user "kept 3 of 10" instead of implying all were useful.
        """
        name = name.strip()
        if not name:
            raise ValueError("identity name cannot be empty")

        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if emb.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM}-D embeddings, got {emb.shape[1]}-D")
        # Reject non-finite rows at the boundary. A NaN embedding stored here makes
        # every subsequent similarity NaN, which fails pydantic's [-1, 1] bound on
        # `Identity.similarity` and crashes the viewer one frame per face later —
        # far from the enrollment that caused it. A NaN can reach here from a
        # failed model load or a zero-variance crop, so it's checked, not assumed.
        if not np.all(np.isfinite(emb)):
            bad = int((~np.isfinite(emb).all(axis=1)).sum())
            raise ValueError(
                f"{bad} of {emb.shape[0]} embeddings contain NaN or inf; refusing to "
                "store them because they would make every later match NaN"
            )
        emb = l2_normalize(emb, axis=1)

        existing_idx = [i for i, n in enumerate(self._names) if n == name]
        existing = self._embeddings[existing_idx] if existing_idx else None

        kept: list[np.ndarray] = []
        for row in emb:
            if self.count_for(name) + len(kept) >= MAX_EMBEDDINGS_PER_IDENTITY:
                break
            # Compare against both what's on disk and what we're about to add,
            # so a batch of near-identical frames doesn't slip through.
            pool = [existing] if existing is not None and len(existing) else []
            if kept:
                pool.append(np.stack(kept))
            if pool:
                sims = np.concatenate([p @ row for p in pool])
                if sims.max() >= DEDUP_SIMILARITY:
                    continue
            kept.append(row)

        if kept:
            new = np.stack(kept)
            self._embeddings = (
                new if self.is_empty else np.vstack([self._embeddings, new])
            ).astype(np.float32)
            self._names.extend([name] * len(kept))

        entry = self._meta.setdefault(
            name,
            {"enrolled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        )
        if relationship is not None:
            entry["relationship"] = relationship
        entry["samples"] = self.count_for(name)

        return len(kept)

    def remove(self, name: str) -> int:
        """Delete an identity and all its embeddings. Returns rows removed.

        Present because biometric data needs a delete path — GDPR-style erasure
        is a real requirement for anything storing face templates, and the old
        version had no way to remove someone short of hand-editing JSON.
        """
        keep = [i for i, n in enumerate(self._names) if n != name]
        removed = len(self._names) - len(keep)
        self._embeddings = (
            self._embeddings[keep] if keep else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        )
        self._names = [self._names[i] for i in keep]
        self._meta.pop(name, None)
        return removed

    # ---- matching ----------------------------------------------------------

    def identify(self, embedding: np.ndarray, threshold: float | None = None) -> Identity:
        """Match one embedding against the store.

        Single vectorized matmul against the whole index — no file I/O, which is
        the fix for the per-frame reload. Uses max similarity per identity rather
        than mean: a person enrolled under varied lighting has some embeddings far
        from any given probe, and averaging those in penalizes exactly the
        well-enrolled identities that should match most reliably.
        """
        thr = self.match_threshold if threshold is None else threshold

        if self.is_empty:
            return Identity(name=None, similarity=0.0, threshold=thr, database_empty=True)

        probe = l2_normalize(np.asarray(embedding, dtype=np.float32).ravel())
        sims = self._embeddings @ probe

        # A non-finite probe (bad frame, failed model run) must be a non-match, not
        # a crash: this runs once per face per frame in the live viewer, and a NaN
        # similarity violates Identity's [-1, 1] bound. Fail closed — unknown.
        if not np.all(np.isfinite(sims)):
            return Identity(name=None, similarity=0.0, threshold=thr, database_empty=False)

        best = int(np.argmax(sims))
        best_sim = float(sims[best])

        if best_sim >= thr:
            return Identity(
                name=self._names[best],
                similarity=best_sim,
                threshold=thr,
                database_empty=False,
            )
        # Explicit non-match: carries the score so the CLI can say "closest was
        # KAITLYN at 0.31, below threshold 0.36" rather than a bare UNKNOWN.
        return Identity(name=None, similarity=best_sim, threshold=thr, database_empty=False)

    def identify_batch(
        self, embeddings: np.ndarray, threshold: float | None = None
    ) -> list[Identity]:
        """Match several probes at once — one matmul for all faces in a frame."""
        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if emb.shape[0] == 0:
            return []
        thr = self.match_threshold if threshold is None else threshold
        if self.is_empty:
            return [
                Identity(name=None, similarity=0.0, threshold=thr, database_empty=True)
            ] * emb.shape[0]

        sims = l2_normalize(emb, axis=1) @ self._embeddings.T
        out: list[Identity] = []
        for row in sims:
            # Same fail-closed rule as `identify`, per probe: one bad face in a
            # frame must not take down the whole frame's recognition.
            if not np.all(np.isfinite(row)):
                out.append(Identity(name=None, similarity=0.0, threshold=thr))
                continue
            best = int(np.argmax(row))
            best_sim = float(row[best])
            out.append(
                Identity(
                    name=self._names[best] if best_sim >= thr else None,
                    similarity=best_sim,
                    threshold=thr,
                    database_empty=False,
                )
            )
        return out

    # ---- calibration -------------------------------------------------------

    def genuine_impostor_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Split all pairwise similarities into genuine (same person) and impostor.

        Feeds `tyf calibrate`, which picks a threshold from *your* enrollments
        instead of inheriting a benchmark constant. Only the upper triangle is
        used, since the similarity matrix is symmetric and the diagonal is
        trivially 1.0.
        """
        n = len(self)
        if n < 2:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)

        sims = self._embeddings @ self._embeddings.T
        names = np.array(self._names)
        same = names[:, None] == names[None, :]
        iu = np.triu_indices(n, k=1)

        vals = sims[iu]
        is_same = same[iu]
        return vals[is_same].astype(np.float32), vals[~is_same].astype(np.float32)
