"""Model weight resolution: cache, download, verify.

The old README asked users to manually download two `.bz2` files from dlib.net,
extract them, and drop them in the right folder. That's a setup step people get
wrong, and it silently half-works: miss one file and you get an exception several
layers into a stack trace about a missing `.dat`.

This module makes weights self-installing and verified. Three properties matter:

1. **Cached outside the repo.** Weights live in a platform cache dir, so they
   survive `git clean`, aren't duplicated per-checkout, and can't be committed.
2. **Checksum-verified.** A truncated download is a real failure mode (the
   insightface issue tracker has multiple "can't download model" threads with
   dead Dropbox/Baidu links). An unverified partial file produces garbage
   embeddings rather than an error, so we hash before trusting.
3. **Offline-friendly.** Nothing here phones home if the file is already cached,
   and `TYF_MODEL_DIR` lets an air-gapped user pre-place weights. The project's
   privacy claim is that it runs fully offline; that has to remain true after the
   one-time fetch.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

# Bump when a model spec changes incompatibly so stale caches are not reused.
CACHE_VERSION = "v1"


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable model artifact.

    insightface distributes its ONNX models inside zip bundles rather than as
    individual files, so a spec may name an `archive_member` to extract.

    Two hashes, because there are two things to verify at two different times:
      `archive_sha256` — the downloaded zip, checked before extraction.
      `sha256`         — the extracted .onnx, checked on every cache hit.
    Conflating them was a real bug: verifying a cached extracted file against the
    archive's hash fails 100% of the time.
    """

    name: str
    filename: str
    url: str
    sha256: str | None
    # Expected input resolution (H, W). Recorded here so preprocessing can't
    # drift out of sync with the graph the weights were trained for.
    input_size: tuple[int, int]
    description: str = ""
    # Path inside the zip, if the download is an archive.
    archive_member: str | None = None
    archive_sha256: str | None = None


# Hard ceiling on anything written to the model cache. The real artifacts total
# ~17 MB, so 200 MB is ~10x headroom and still far below "filled the disk".
#
# This is NOT the primary defence — SHA-256 verification is, and it means a
# tampered or truncated download can never produce a loadable model. The bound
# closes a narrower gap: bytes are streamed to disk BEFORE the hash is checked, so
# without a cap a hostile mirror or a decompression bomb could fill the cache
# directory (or /tmp) even though the result would then be rejected. Cheap to
# enforce, and it turns an unbounded write into a named error.
MAX_ARTIFACT_BYTES = 200 * 1024 * 1024


def _copy_bounded(src, dst, limit: int = MAX_ARTIFACT_BYTES, what: str = "download") -> None:
    """Stream src -> dst, aborting past `limit` bytes."""
    written = 0
    while chunk := src.read(1 << 16):
        written += len(chunk)
        if written > limit:
            raise ModelDownloadError(
                f"{what} exceeded {limit // (1024 * 1024)} MB and was aborted. The "
                "expected artifacts are ~17 MB total, so this is either a corrupted "
                "source or a hostile one."
            )
        dst.write(chunk)


# Detection + embedding both come from insightface's `buffalo_sc` pack: a 2.4 MB
# SCRFD-500M detector and a 13 MB MobileFaceNet ArcFace embedder.
#
# Chosen over the larger buffalo_l (289 MB, ResNet50) deliberately. This project
# targets edge hardware — a Raspberry Pi and a 4 GB-VRAM laptop GPU — so a 15 MB
# total footprint that runs comfortably on ARM CPU is worth more than the couple
# of points of verification accuracy the ResNet50 variant would add. It also
# keeps first-run download honest for users on slow connections.
#
# Every hash below was computed from an actual download and confirmed to load in
# ONNX Runtime — not copied from documentation.
_BUFFALO_SC_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
_BUFFALO_SC_SHA = "57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72"

DETECTOR = ModelSpec(
    name="scrfd_500m",
    filename="det_500m.onnx",
    url=_BUFFALO_SC_URL,
    sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    input_size=(640, 640),
    description="SCRFD 500M face detector with 5-point keypoints (2.4 MB).",
    archive_member="det_500m.onnx",
    archive_sha256=_BUFFALO_SC_SHA,
)

EMBEDDER = ModelSpec(
    name="arcface_w600k_mbf",
    filename="w600k_mbf.onnx",
    url=_BUFFALO_SC_URL,
    sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
    input_size=(112, 112),
    description="ArcFace MobileFaceNet trained on WebFace600K, 512-D (13 MB).",
    archive_member="w600k_mbf.onnx",
    archive_sha256=_BUFFALO_SC_SHA,
)


def model_dir() -> Path:
    """Where weights live. `TYF_MODEL_DIR` overrides for air-gapped installs."""
    override = os.environ.get("TYF_MODEL_DIR")
    if override:
        p = Path(override).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path(user_cache_dir("trainyourface")) / "models" / CACHE_VERSION
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class ModelDownloadError(RuntimeError):
    """Raised when a model can't be fetched or fails verification."""


def ensure_model(spec: ModelSpec, *, allow_download: bool = True) -> Path:
    """Return a local path to the model, downloading and verifying if needed.

    Downloads to a temporary sibling and renames only after the checksum passes,
    so an interrupted transfer can never leave a corrupt file that later loads as
    a valid-looking model producing nonsense embeddings.
    """
    dest = model_dir() / spec.filename

    if dest.exists():
        if spec.sha256 is not None:
            actual = sha256_file(dest)
            if actual != spec.sha256:
                raise ModelDownloadError(
                    f"Cached {spec.filename} is corrupt "
                    f"(sha256 {actual[:12]}… != expected {spec.sha256[:12]}…). "
                    f"Delete {dest} and retry."
                )
        return dest

    if not allow_download:
        raise ModelDownloadError(
            f"{spec.filename} not found in {model_dir()} and downloads are disabled. "
            f"Fetch it from {spec.url} or set TYF_MODEL_DIR to a directory containing it."
        )

    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(spec.url, timeout=120) as resp, tmp.open("wb") as out:
            _copy_bounded(resp, out, what=f"download of {spec.name}")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Failed to download {spec.name} from {spec.url}: {exc}. "
            f"You can place {spec.filename} manually in {model_dir()}."
        ) from exc

    # Verify the downloaded bytes. For an archive that means the archive hash;
    # the extracted member is verified separately below.
    expected_download = spec.archive_sha256 if spec.archive_member else spec.sha256
    if expected_download is not None:
        actual = sha256_file(tmp)
        if actual != expected_download:
            tmp.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Downloaded {spec.name} failed verification "
                f"(sha256 {actual[:12]}… != expected {expected_download[:12]}…). "
                "The upstream artifact may have been replaced; do not use it."
            )

    # Archive downloads: extract just the member we need, then discard the zip.
    # Extraction is done member-by-member with an explicit basename to avoid
    # zip-slip (a crafted archive containing "../../.ssh/authorized_keys").
    if spec.archive_member is not None:
        try:
            with zipfile.ZipFile(tmp) as zf:
                names = zf.namelist()
                match = next(
                    (n for n in names if Path(n).name == Path(spec.archive_member).name), None
                )
                if match is None:
                    raise ModelDownloadError(
                        f"{spec.archive_member} not found in archive; contains: {names}"
                    )
                with zf.open(match) as src, dest.open("wb") as out:
                    # Bounded because the declared size in a zip header is
                    # attacker-controlled; only the actual read is trustworthy.
                    _copy_bounded(src, out, what=f"extraction of {spec.filename}")
        except zipfile.BadZipFile as exc:
            raise ModelDownloadError(f"{spec.url} is not a valid zip archive: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)

        # Verify the extracted member too — this is the file we actually load,
        # and it's what the cache-hit path above will check on every later run.
        if spec.sha256 is not None:
            actual = sha256_file(dest)
            if actual != spec.sha256:
                dest.unlink(missing_ok=True)
                raise ModelDownloadError(
                    f"Extracted {spec.filename} failed verification "
                    f"(sha256 {actual[:12]}… != expected {spec.sha256[:12]}…)."
                )
        return dest

    tmp.replace(dest)
    return dest


def resolve_providers(prefer_gpu: bool = True) -> list[str]:
    """Pick ONNX Runtime execution providers, best-available first.

    ONNX Runtime silently falls back to CPU when a requested provider is missing,
    which makes "why is this slow" hard to diagnose. We ask only for providers
    actually present in this build so the chosen list reflects reality and can be
    reported to the user.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen: list[str] = []
    if prefer_gpu:
        # Ordered by preference: CUDA (the RTX laptop), then CoreML (Apple
        # Neural Engine / GPU), then the Arm/Android NN path.
        for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "NnapiExecutionProvider"):
            if p in available:
                chosen.append(p)
    chosen.append("CPUExecutionProvider")
    return chosen
