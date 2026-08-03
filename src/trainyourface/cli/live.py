"""Live webcam viewer and interactive enrollment.

Replaces the original `1 for training; 2 for identification` stdin menu. Beyond
being nicer, the old menu had a real defect: `register_face` reset an identity's
descriptor list on what it called "initial registration", so re-enrolling a name
silently destroyed that person's existing samples. The store is additive now, and
enrollment reports how many samples it actually kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from trainyourface.cli.overlay import ScoreHistory, draw_face, draw_hud
from trainyourface.core.align import align_face

# Enrollment quality gates. A blurry or tiny enrollment frame produces a bad
# template that then degrades every future match against that identity — and the
# damage is invisible, showing up as unexplained non-matches later. Cheaper to
# refuse the frame.
MIN_ENROLL_FACE_PX = 110
MIN_SHARPNESS = 40.0


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap blur estimate.

    Absolute values are scene-dependent, so this is a threshold tuned for
    "obviously motion-blurred", not a calibrated measure.
    """
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run_watch(
    *,
    camera: int,
    liveness_path: Path | None,
    use_liveness: bool,
    liveness_threshold: float | None,
    match_threshold: float | None,
    store_path: Path | None,
    identify_spoofs: bool,
    mirror: bool = True,
) -> None:
    """The main demo: continuous detect -> liveness -> recognize with an overlay."""
    import cv2

    from trainyourface.cli.loader import load_pipeline, open_camera, require_gui

    require_gui()
    pipeline = load_pipeline(
        liveness_path=liveness_path,
        use_liveness=use_liveness,
        store_path=store_path,
        match_threshold=match_threshold,
        liveness_threshold=liveness_threshold,
        identify_spoofs=identify_spoofs,
    )

    typer.echo()
    for line in pipeline.describe():
        typer.echo(f"  {line}")
    typer.echo()
    typer.secho(
        "  q quit    s screenshot    SPACE pause    l legend    t trace    h hud",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.echo()

    warn = None
    if pipeline.liveness is None:
        warn = "NO LIVENESS MODEL - a photo would pass. Nothing shown as trusted."
    elif identify_spoofs:
        warn = "--identify-spoofs: identities shown for rejected faces (eval mode)"

    subtitle = "trainyourface" + ("" if pipeline.liveness else "  (recognition only)")
    keys = "q quit  s shot  SPACE pause  l legend"

    cap = open_camera(camera)
    history = ScoreHistory()
    shots = 0
    # Legend defaults on only when there's no liveness model: that's the case
    # where the states need explaining, since UNVERIFIED is the confusing one.
    legend = pipeline.liveness is None
    trace = True
    show_hud = True
    paused = False

    # The last raw frame and its result are kept separately from the rendered
    # view. Pausing then re-renders from those instead of re-annotating an
    # already-annotated image, so toggling the HUD or legend while paused works
    # and screenshots never contain doubled overlays.
    frame: np.ndarray | None = None
    result = None

    try:
        while True:
            if not paused or frame is None:
                ok, grabbed = cap.read()
                if not ok:
                    typer.secho("camera read failed", fg=typer.colors.RED, err=True)
                    break
                frame = cv2.flip(grabbed, 1) if mirror else grabbed
                result = pipeline.process(frame)
                for face in result.faces:
                    if face.liveness is not None:
                        history.push(face.box, face.liveness.spoof_probability)

            view = frame.copy()
            for face in result.faces:
                draw_face(view, face, history=history, show_trace=trace)
            if show_hud:
                draw_hud(
                    view,
                    result,
                    warn=warn,
                    keys=keys,
                    subtitle=subtitle,
                    legend=legend,
                    paused=paused,
                )

            cv2.imshow("trainyourface", view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("l"):
                legend = not legend
            elif key == ord("t"):
                trace = not trace
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("s"):
                out = Path(f"screenshot_{shots:03d}.png")
                cv2.imwrite(str(out), view)
                typer.secho(f"  saved {out}", fg=typer.colors.GREEN)
                shots += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_enroll(
    name: str,
    *,
    camera: int,
    samples: int,
    store_path: Path | None,
    relationship: str | None,
    require_live: bool = True,
    liveness_path: Path | None = None,
) -> None:
    """Enroll a face from the webcam.

    `require_live` defaults to True, which is the security-relevant default:
    enrolling from a photo would install an attacker's chosen image as a
    legitimate template, and after that no amount of liveness checking at match
    time helps — the spoof is now the enrolled ground truth.
    """
    import cv2

    from trainyourface.cli.loader import load_pipeline, open_camera, require_gui

    require_gui()
    pipeline = load_pipeline(
        liveness_path=liveness_path,
        use_liveness=require_live,
        store_path=store_path,
    )

    if require_live and pipeline.liveness is None:
        typer.secho(
            "Enrolling without a liveness check. A printed photo held to the camera "
            "would be enrolled as a real identity. Pass --no-require-live to accept "
            "this deliberately, or train a liveness model first with `tyf train`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    store = pipeline.store
    if store is None:
        raise typer.Exit(code=2)

    typer.echo()
    typer.secho(f"Enrolling: {name}", fg=typer.colors.CYAN, bold=True)
    if store.count_for(name):
        typer.echo(f"  {name} already has {store.count_for(name)} samples; adding to them.")
    typer.echo(f"  Need {samples} good frames. Turn your head slowly between captures —")
    typer.echo("  varied pose is what makes the template robust.")
    typer.secho("  SPACE capture    a auto-capture    u undo    q abort", fg=typer.colors.YELLOW)
    typer.echo()

    cap = open_camera(camera)
    collected: list[np.ndarray] = []
    auto = True  # on by default: it produces better pose variety than manual timing
    cooldown = 0
    flash = 0

    try:
        while len(collected) < samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            result = pipeline.process(frame)
            face = max(result.faces, key=lambda f: f.box.area) if result.faces else None

            reason = _reject_reason(frame, face, require_live)
            view = frame.copy()
            for f in result.faces:
                draw_face(view, f, show_trace=False)

            _draw_enroll_guide(
                view,
                name=name,
                collected=len(collected),
                target=samples,
                reason=reason,
                auto=auto,
                cooldown=cooldown,
                flash=flash,
                face=face,
            )
            draw_hud(view, result, keys="SPACE capture  a auto  u undo  q abort")
            cv2.imshow("tyf enroll", view)
            flash = max(0, flash - 1)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                typer.secho("aborted; nothing saved", fg=typer.colors.YELLOW)
                return
            if key == ord("a"):
                auto = not auto
            if key == ord("u") and collected:
                collected.pop()
                typer.echo(f"  undo -> {len(collected)}/{samples}")
            if key == ord(" "):
                cooldown = 0

            want = key == ord(" ") or (auto and cooldown <= 0)
            if want and face is not None and reason is None:
                kps = _keypoints_for(pipeline, frame, face)
                crop = align_face(frame, kps) if kps is not None else None
                if crop is not None:
                    collected.append(crop)
                    typer.echo(f"  captured {len(collected)}/{samples}")
                    flash = 4
                    # Frames this close together are near-duplicates; the store
                    # would dedup them anyway, so waiting produces varied poses
                    # instead of ten copies of one.
                    cooldown = 12
            cooldown -= 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not collected:
        typer.secho("no frames captured", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    embeddings = pipeline.embedder.embed(collected)
    kept = store.add(name, embeddings, relationship=relationship)
    store.save()

    typer.echo()
    typer.secho(f"enrolled {name}: kept {kept} of {len(collected)} frames", fg=typer.colors.GREEN)
    if kept < len(collected):
        # Not a failure — dedup working. Said plainly so the count isn't alarming.
        typer.echo(f"  ({len(collected) - kept} dropped as near-duplicates of existing samples)")
    typer.echo(f"  {name} now has {store.count_for(name)} samples")
    typer.echo(f"  store: {store.path}")


def _draw_enroll_guide(
    frame: np.ndarray,
    *,
    name: str,
    collected: int,
    target: int,
    reason: str | None,
    auto: bool,
    cooldown: int,
    flash: int,
    face,
) -> None:
    """Enrollment-specific overlay: progress dots and what to fix.

    Dots rather than a percentage bar because the counts here are small (8 by
    default) and a discrete "3 of 8 captured" is easier to read at a glance than
    37%. The rejection reason is shown large and centred — enrollment is the one
    place a user is actively trying to satisfy the tool, so telling them exactly
    what's wrong is the whole job.
    """
    import cv2

    from trainyourface.cli.theme import (
        AMBER,
        DIM,
        GREEN,
        PANEL,
        TRACK,
        WHITE,
        panel,
        pill,
        text,
        text_size,
    )

    h, w = frame.shape[:2]

    # A white flash on capture, so it's obvious a frame was taken. Without it,
    # auto-capture feels like nothing is happening.
    if flash:
        cv2.addWeighted(frame, 0.75, np.full_like(frame, 255), 0.25, 0, frame)

    header = f"enrolling {name}"
    hw = text_size(header, 0.5, 1)[0]
    dot_r, gap = 7, 22
    dots_w = target * gap
    box_w = max(hw, dots_w) + 32
    x0 = (w - box_w) // 2
    panel(frame, (x0, h - 96), (x0 + box_w, h - 20), PANEL, alpha=0.72)

    text(frame, header, ((w - hw) // 2, h - 66), WHITE, scale=0.5)
    dx = (w - dots_w) // 2 + gap // 2
    for i in range(target):
        filled = i < collected
        cv2.circle(
            frame,
            (dx + i * gap, h - 42),
            dot_r,
            GREEN if filled else TRACK,
            -1 if filled else 1,
            cv2.LINE_AA,
        )

    if auto:
        badge = "AUTO" if cooldown <= 0 else f"AUTO {cooldown}"
        pill(frame, badge, (x0 + box_w + 8, h - 90), GREEN if cooldown <= 0 else DIM, scale=0.4)

    # The blocker, or an explicit ready state. Never blank: silence during
    # enrollment reads as a frozen program.
    if reason:
        msg, color = reason, AMBER
    elif face is None:
        msg, color = "no face detected", AMBER
    else:
        msg, color = "hold still - capturing", GREEN
    mw = text_size(msg, 0.55, 1)[0]
    panel(frame, ((w - mw) // 2 - 16, 54), ((w + mw) // 2 + 16, 90), PANEL, alpha=0.78)
    text(frame, msg, ((w - mw) // 2, 79), color, scale=0.55)


def _keypoints_for(pipeline, frame, face) -> np.ndarray | None:
    """Re-run detection to recover keypoints for the chosen face.

    FaceObservation intentionally doesn't carry keypoints — they're an alignment
    detail, not a conclusion about the face. Enrollment is the one place that
    needs them after the fact, and it runs at human speed, so re-detecting is
    cheaper than widening the contract for every frame of the live path.
    """
    boxes, kps = pipeline.detector.detect(frame)
    if not boxes or not kps:
        return None
    best = max(range(len(boxes)), key=lambda i: boxes[i].area)
    return kps[best] if kps[best] is not None else None


def _reject_reason(frame, face, require_live: bool) -> str | None:
    """Why this frame is unsuitable for enrollment, or None if it's fine."""
    import cv2

    if face is None:
        return "no face detected"
    if min(face.box.width, face.box.height) < MIN_ENROLL_FACE_PX:
        return "move closer"

    box = face.box.clipped(frame.shape[1], frame.shape[0])
    patch = frame[box.y1 : box.y2, box.x1 : box.x2]
    if patch.size == 0:
        return "face out of frame"
    if _sharpness(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)) < MIN_SHARPNESS:
        return "too blurry - hold still"

    if require_live and face.liveness is not None and not face.liveness.is_live:
        return f"liveness rejected ({face.liveness.spoof_probability:.2f})"
    return None
