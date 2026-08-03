"""Interactive PAD data collection.

This exists because the standard anti-spoofing datasets (CASIA-FASD,
Replay-Attack, OULU-NPU, SiW) all require signed institutional license agreements
and cannot be redistributed. So the dataset has to be built, and the tool that
builds it is part of the project rather than a side script.

WHAT GETS RECORDED, AND WHY IT'S CROPPED THIS WAY
-------------------------------------------------
Frames are saved as face crops with CROP_MARGIN of context, at INPUT_SIZE — the
exact same crop the model is trained and evaluated on. Saving full frames and
cropping later would work too, but it makes it possible for the capture crop and
the training crop to diverge, which is the train/serve mismatch class of bug this
project already had once (in the recognition path) and is careful not to repeat.

The margin is not cosmetic: screen bezels, paper edges and the hand holding the
attack instrument sit outside the face box and are among the strongest PAD cues.

WHY IT REFUSES TO RECORD WITHOUT A DETECTED FACE
------------------------------------------------
A silently-empty or off-target crop becomes a mislabeled training sample, and
mislabeled PAD data is worse than missing PAD data: the model learns the inverse
of the truth and the metrics still look plausible. So capture requires a
detection and says so on screen when it isn't recording.
"""

from __future__ import annotations

from pathlib import Path

import typer

from trainyourface.core.align import crop_box
from trainyourface.core.contracts import AttackType
from trainyourface.liveness.dataset import CROP_MARGIN, INPUT_SIZE, DatasetManifest, Sample

# Minimum face size to accept. A tiny detection is usually a background face or a
# false positive, and upscaling it to 128px fabricates texture the model would
# then learn from.
MIN_FACE_PX = 80

# Frames are near-identical at camera rate. Keeping every one inflates the
# dataset without adding information and makes a subject-disjoint split look
# larger than it is. 1-in-N keeps some pose/lighting drift between saved frames.
DEFAULT_STRIDE = 3


def run_capture(
    out_dir: Path,
    label: AttackType,
    subject: str,
    session: str,
    target: int,
    instrument: str | None,
    camera: int,
    stride: int = DEFAULT_STRIDE,
) -> int:
    """Capture labeled PAD frames from the webcam. Returns how many were saved."""
    import cv2

    from trainyourface.cli.loader import open_camera, require_gui
    from trainyourface.cli.overlay import draw_capture_hud
    from trainyourface.core.detect import FaceDetector

    require_gui()

    images_dir = out_dir / "images" / label.value
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # Append to any existing manifest so a session can be built up over several
    # runs and several days — session-level splitting is only meaningful if
    # multiple sessions can actually accumulate.
    manifest = DatasetManifest.load(manifest_path) if manifest_path.exists() else DatasetManifest()
    manifest.root = str(out_dir)
    existing = len(manifest.samples)

    detector = FaceDetector()
    cap = open_camera(camera)

    _print_instructions(label, subject, session, target, instrument)

    saved = 0
    frame_index = 0
    window = f"tyf capture [{label.value}]"

    try:
        while saved < target:
            ok, frame = cap.read()
            if not ok:
                typer.secho("camera read failed", fg=typer.colors.RED, err=True)
                break

            frame = cv2.flip(frame, 1)  # mirror: matches how people expect a webcam
            boxes, _ = detector.detect(frame)
            boxes = [b for b in boxes if min(b.width, b.height) >= MIN_FACE_PX]
            # One subject per frame. Two faces means the attack instrument and a
            # real person are both visible, which would be labeled wrong either way.
            box = max(boxes, key=lambda b: b.area) if boxes else None

            key = cv2.waitKey(1) & 0xFF
            recording = key == ord(" ")

            if recording and box is not None and frame_index % stride == 0:
                crop = crop_box(frame, box, size=INPUT_SIZE, margin=CROP_MARGIN)
                name = f"{subject}_{session}_{label.value}_{existing + saved:05d}.png"
                cv2.imwrite(str(images_dir / name), crop)
                manifest.samples.append(
                    Sample(
                        # Relative to the manifest root so the dataset stays
                        # portable — absolute paths break the moment it moves.
                        path=str(Path("images") / label.value / name),
                        attack_type=label,
                        subject=subject,
                        session=session,
                        instrument=instrument,
                    )
                )
                saved += 1

            view = draw_capture_hud(
                frame.copy(),
                label=label.value,
                subject=subject,
                session=session,
                captured=saved,
                target=target,
                has_face=box is not None,
                instrument=instrument,
            )
            if box is not None:
                color = (80, 220, 100) if label == AttackType.BONA_FIDE else (60, 60, 240)
                cv2.rectangle(view, (box.x1, box.y1), (box.x2, box.y2), color, 2)

            cv2.imshow(window, view)
            frame_index += 1

            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Save whatever was collected even on an early quit — losing a partial
        # capture session to Ctrl-C would be a real annoyance, and the manifest
        # write is atomic so a partial file can't result.
        if saved:
            manifest.save(manifest_path)

    typer.secho(f"\nsaved {saved} frames -> {images_dir}", fg=typer.colors.GREEN)
    typer.echo(f"manifest: {manifest_path} ({len(manifest.samples)} total samples)")
    return saved


def _print_instructions(
    label: AttackType, subject: str, session: str, target: int, instrument: str | None
) -> None:
    typer.echo()
    typer.secho(f"Capturing: {label.value}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  subject   {subject}")
    typer.echo(f"  session   {session}")
    typer.echo(f"  instrument{'':<1}{instrument or '(none)'}")
    typer.echo(f"  target    {target} frames")
    typer.echo()

    if label == AttackType.BONA_FIDE:
        typer.echo("  Point the camera at a REAL face.")
        typer.echo("  Vary pose, distance and expression while recording. Move between")
        typer.echo("  lighting conditions if you can — a model trained under one light")
        typer.echo("  learns that light, not liveness.")
    elif label == AttackType.PRINT:
        typer.echo("  Hold a PRINTED PHOTO of the subject up to the camera.")
        typer.echo("  Vary distance and angle. Include some frames where the paper edge")
        typer.echo("  is visible and some where it fills the frame — otherwise the model")
        typer.echo("  learns 'paper edge = attack' rather than the print texture itself.")
    elif label == AttackType.REPLAY:
        typer.echo("  Show a PHOTO OR VIDEO of the subject on a phone/tablet screen.")
        typer.echo("  Tilt the screen to produce moiré at some angles and not others, and")
        typer.echo("  vary brightness. Both bezel-visible and bezel-cropped frames.")
    else:
        typer.echo(f"  Present the {label.value} attack instrument to the camera.")

    typer.echo()
    typer.secho("  HOLD SPACE to record, q to stop.", fg=typer.colors.YELLOW)
    typer.echo()
