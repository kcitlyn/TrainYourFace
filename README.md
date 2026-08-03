# TrainYourFace

**Face recognition that knows when it's being fooled.**

Hold a photo of an enrolled person up to the camera. Most face recognition — including
the first version of this project — will happily unlock. It matched the face, and a
photo of a face is a face.

That is the entire problem this project exists to solve. TrainYourFace trains a small
PyTorch model to tell a live face from a printed photo or a phone screen, evaluates it
with the ISO/IEC 30107-3 metrics used to report presentation-attack detection, and
gates recognition behind it. A face is reported as **trusted** only when it is both
recognized *and* verified live.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│  detect  │───▶│ liveness │───▶│ identify │───▶│ TRUSTED │
└──────────┘    └────┬─────┘    └──────────┘    └─────────┘
                     │ fails
                     ▼
                  ┌───────┐   recognition never runs
                  │ SPOOF │   — the face is never embedded
                  └───────┘
```

Runs fully offline. No cloud, no API keys, no telemetry. Installs with no compiler —
no CMake, no C++ toolchain, no dlib.

---

## Why this exists

I wrote the original version as a webcam face-recognition tool, and it worked. Then I
held up a photo of myself and it greeted me by name. That's not a bug in the matcher —
the matcher was right. It's that "is this the right face" and "is there a person here"
are two different questions, and only answering the first one produces a system that
feels secure and isn't.

So the project changed shape. Recognition is now the easy half. The interesting half
is the liveness model, the metrics that say how much to trust it, and the plumbing
that keeps a reported number honest.

Three things also got fixed along the way that had nothing to do with spoofing and
everything to do with the results being real:

- **dlib is gone.** It required CMake and a C++ toolchain, which is the single most
  common reason someone gives up on installing a CV project. Detection and embedding
  now run on ONNX Runtime, and CI proves the install works on Linux/macOS/Windows ×
  Python 3.10/3.13 with no compiler.
- **A train/serve preprocessing mismatch.** Enrollment ran images through an aligned
  crop; the live path used an unaligned one. Enrolled and live embeddings landed in
  different regions of embedding space, so matching quietly degraded and the threshold
  had to be cranked down to compensate. Nothing ever errored. Both paths now call one
  normalization function, and a test asserts they agree byte-for-byte.
- **`--seed` didn't work.** Training seeded torch and numpy and still produced a
  different model every run, because the augmentation built an unseeded RNG on every
  call. I found it when `train` reported 0% worst-case APCER and `eval` reported 12.5%
  on what I thought was the same model. Augmentation is now seeded from
  `(seed, epoch, index)`, so the result doesn't depend on DataLoader worker count or
  the order workers pull indices. Two runs at the same seed now produce bit-identical
  weights.

That last one matters more than it looks. An unreproducible training run makes every
metric unverifiable — not just by a reader, but by me.

---

## Install

Not on PyPI yet — install from the repo:

```bash
TYF="git+https://github.com/kcitlyn/TrainYourFace"

pip install "$TYF"                 # inference: detect, enroll, recognize, liveness
pip install "trainyourface[demo] @ $TYF"   # + the live webcam window
pip install "trainyourface[train] @ $TYF"  # + PyTorch, to train your own model
pip install "trainyourface[export] @ $TYF" # + ONNX/CoreML export and quantization
```

Every one of those resolves without a compiler, which is the claim CI exists to check:
the matrix builds on Linux/macOS/Windows × Python 3.10/3.13 and would fail outright on
a dlib-based project.

Model weights download on first use and are cached. Nothing to place by hand.

## Use it as a library

```python
from trainyourface import LivenessDetector, FaceID

det = LivenessDetector()  # loads once, reuses the session
result = det.check(frame)  # frame: HWC BGR uint8, e.g. from cv2
result.is_live, result.spoof_probability
# (True, 0.02)

fid = FaceID()  # recognition gated behind liveness
fid.enroll("kaitlyn", frame)  # raises rather than enroll from a photo
for face in fid.identify(frame):
    print(face.status, face.name)  # TRUSTED kaitlyn
```

Two API decisions worth naming, both taken from watching users complain about the
alternative in the incumbent library:

- **A spoof is a return value, not an exception.** Rejecting a presentation attack is
  the normal operation of a PAD system, not an error condition. Making it raise forces
  a `try/except` around the expected case.
- **Models load in `__init__`.** Reloading per call is untenable in a video loop or a
  server, which is the setting this gets used in.

`FaceID()` **refuses to construct** without a liveness model rather than falling back to
recognition-only. That fallback is available as `require_liveness=False`, and the error
names it — but it has to be asked for, because the silent version means a photo passes
on any machine where the model happens to be missing.

---

## Use it

```bash
tyf enroll kaitlyn      # register a face (requires liveness — won't enroll a photo)
tyf watch               # live recognition with liveness gating
tyf info                # what's installed, which execution provider is active
```

`tyf watch` labels every face with one of four states:

| State | Meaning |
| --- | --- |
| `TRUSTED` | Recognized **and** verified live. The only state that means anything. |
| `SPOOF` | Liveness rejected it. Identity is deliberately not shown. |
| `UNKNOWN` | Live person, not enrolled. |
| `UNVERIFIED` | No liveness model loaded. Recognition only — **a photo will pass.** |

Without a liveness model the tool says `UNVERIFIED`, never `TRUSTED`. A liveness check
that defaults to pass is worse than no liveness check at all, because it looks like
protection.

---

## Train your own liveness model

Two data sources, and they answer different questions.

**A public benchmark.** CelebA-Spoof — 625K images, 10,177 subjects — downloads directly
with no license application, unlike CASIA-FASD / Replay-Attack / OULU-NPU / SiW, which
all need signed institutional agreements. Non-commercial research use, per its terms.

```bash
tyf import-celeba ~/Downloads/CelebA_Spoof --limit 40000 --attributes
tyf train --epochs 30
```

`--attributes` carries the CelebA face-attribute labels into the manifest, which is what
makes the per-group BPCER audit below possible. It costs nothing at train time.

**Your own camera.** The benchmark can't tell you whether the model survives *your*
hardware.

```bash
tyf capture --label bona_fide --subject you --session day1
tyf capture --label print     --subject you --session day1 --instrument laser_matte
tyf capture --label replay    --subject you --session day1 --instrument iphone13

tyf manifest                 # inspect what you've collected
tyf train --epochs 30        # subject-disjoint split, evaluates test once
tyf eval --markdown          # regenerate the metrics table below
tyf export --format onnx     # verified against PyTorch on export
```

Capture 3+ subjects across 2+ sessions. The split refuses to run below 3 subjects
rather than silently produce a test set that shares people with train.

### The evaluation that actually matters

Train on one, test on the other:

```bash
tyf eval --data ~/celeba-manifest --cross ~/my-captures
```

Every PAD model scores well on the dataset it was built from — the frames share
cameras, lighting, and attack instruments, so the model can lean on cues specific to
that setup. The number that says whether it's deployable is what happens on a camera
and a room it has never seen. This prints both side by side and names the failure mode:

- **Discrimination collapse** — EER and AUC degrade. The model can't separate the
  classes at *any* threshold; it learned the dataset, not the phenomenon.
- **Calibration collapse** — EER holds steady while APCER/BPCER blow up. The model
  still *ranks* attacks above real faces, but the score distribution shifted, so the
  threshold carried over from training now sits in the wrong place.

The second is more common and more dangerous, because the threshold-free metrics look
pristine while every attack gets accepted. An early version of this report keyed its
verdict on EER alone and cheerfully called a 0% → 100% worst-case APCER regression "the
good outcome". That's now a test.

### What the training pipeline refuses to do

Every one of these is a way to get a better-looking number that means less:

- **Split randomly by image.** A capture session yields hundreds of near-identical
  frames. Split randomly and frame 41 goes to train while frame 42 goes to test — the
  model scores ~99% by memorizing "this person in this room" and collapses on anyone
  new. Splits are **subject-disjoint**, optionally session-disjoint, and integrity is
  asserted after every split rather than assumed.
- **Pick the threshold on test.** Extremely common and easy to do by accident. The
  deployment threshold is chosen on **validation** and applied unchanged to test. Both
  numbers get printed, because the gap between them is itself informative.
- **Trust that "held out" means held out.** `tyf eval` doesn't read a stored split — it
  re-derives one from the manifest using `--seed` and `--split-by`, which default to
  `0`/`subject` no matter what training used. Evaluate a model trained with `--seed 7`
  and you get a *different* partition, with subjects the model trained on sitting in the
  "held-out" test set. Nothing looks wrong: the new split is still internally disjoint,
  so the integrity check passes and the report renders normally with inflated numbers.
  Training now records **which subjects** were held out, and `eval` verifies its
  reconstruction against that list before scoring anything.
- **Select checkpoints on accuracy.** On an attack-heavy set, accuracy rewards a model
  that leans toward predicting "attack" — which looks fine and locks real users out.
  Selection is on validation **EER**, which is threshold-free.
- **Average APCER across attack types.** An attacker picks the attack, so the mean is
  the wrong aggregate. **Worst-case APCER** is the headline number; the per-type
  breakdown is reported alongside so you can say *which* instrument defeats the model.
- **Report a latency without naming the provider.** ONNX Runtime silently falls back to
  CPU. A number without an execution provider attached is unattributable, so `tyf
  bench` always prints it — plus median and p95, never mean, because on an edge device
  the tail is what breaks a frame budget.
- **Quote one aggregate APCER.** A model at 3% overall can be at 30% in backlit
  conditions — and backlit is exactly where someone holds up a phone screen. Where the
  data carries capture metadata, APCER is broken out per illumination and environment,
  worst bucket first. Buckets under n=20 print their raw count and *no rate*, because
  "2 of 3 missed" reads as a 67% finding and is noise.
- **Report one BPCER for everybody.** A false rejection is a harm that lands on a
  *user*, repeatedly, every time they try to unlock. If the model rejects one group
  twice as often as another, that group experiences the product as broken while the
  headline number looks fine. `tyf import-celeba --attributes` records the CelebA face
  attributes, and eval reports **BPCER per group** with the worst/best ratio, flagging
  anything past the four-fifths rule (the US EEOC's disparate-impact screen). Groups
  under n=50 get no rate — a bias claim needs more evidence than a robustness one.

### On the fairness audit's limits

The demographic labels come from CelebA and are **binary crowd annotations**, not
measurements. `Pale_Skin` in particular is one bit decided by an annotator — it is not
Fitzpatrick, not the Monk scale, and it collapses a continuum. A disparity along it is a
**signal worth investigating, not a measured skin-tone bias**, and the report says so
next to the number every time it prints. `Attractive` is parsed but deliberately never
audited: reporting rates by it would treat a subjective annotation as a real category.

This is a screening tool that tells you where to look. It is not a bias certification,
and the honest version of this analysis needs skin-tone labels this dataset doesn't have.

---

## Measured results

**Latency** — Apple M-series, ONNX Runtime CoreML execution provider, 200 timed
iterations, 30 warmup excluded. Reproduce with `tyf bench --full-pipeline`.

| Stage | Median | p95 | Notes |
| --- | --- | --- | --- |
| Liveness, batch 1 | 0.44 ms | 0.46 ms | 226K params, 1.0 MB ONNX |
| Liveness, batch 16 | 4.42 ms | 4.81 ms | 0.28 ms per face amortized |
| Detect, 1280×720 | 17.99 ms | 21.36 ms | SCRFD-500M |
| Embed, 1 face | 1.37 ms | 3.00 ms | ArcFace w600k-MBF |

Liveness costs ~2% of the frame budget. Detection dominates, which is the useful thing
to know before optimizing anything.

**Accuracy** — not yet measured on real data, and deliberately not quoted here. The
pipeline is verified end to end on a synthetic fixture (smooth gradients as "bona fide",
a high-frequency grid standing in for moiré), which is enough to prove the plumbing:
gradients flow, the validation threshold survives export, and the ONNX model agrees with
PyTorch to 1e-4. It is *not* an accuracy result — the cue is one the model finds in six
epochs, and quoting a rate off it would be exactly the kind of flattering number the rest
of this README is about avoiding. Real capture is the next step; `tyf eval --markdown`
writes the table straight into this section so the numbers can't drift from the model
that produced them.

Metrics reported, per ISO/IEC 30107-3: **APCER** per attack type (attack frames wrongly
accepted), **BPCER** (real users wrongly rejected), **ACER**, **EER**, and
BPCER @ APCER ≤ 1%/5%/10% — because the operating point you'd actually deploy at is
the one worth quoting.

---

## Tests

```bash
pytest                        # 401 tests
pytest -m train               # + the end-to-end run (trains a real model, ~3 min)
pytest --cov=trainyourface    # 71% total
```

The suite is organized around the failure modes rather than around the modules, because
the bugs in this project were never "this function returns the wrong number" — they were
"this number is real but it means something other than what the label says."

- **`test_e2e.py`** runs the whole chain for real: capture fixture → `train` → `eval` →
  `export` → load the ONNX artifact → gate recognition through it. It asserts the
  properties that a per-module test can't see, chiefly that the threshold in the test
  report is the *validation* threshold, byte-identical, and that ONNX and PyTorch agree
  to 1e-4 on the exported model.
- **`test_robustness.py`** is the adversarial half — degenerate splits, single-class
  datasets, corrupt summaries, NaNs, and the split-reconstruction leak above.
- **`test_cli.py`** drives every command through `CliRunner` and asserts **exit codes**,
  not just messages. A CI pipeline checks `$?`, not prose, so a guard that prints an
  error and exits 0 is not a guard.

Two things the tests deliberately don't claim: `cli/live.py` and `cli/capture.py` sit at
0% because they need a physical camera, and the e2e liveness numbers are plumbing
verification on synthetic data, not accuracy. Both are stated rather than papered over.

---

## Design decisions worth defending

**Liveness before recognition.** Two reasons, and the second is the real one: it skips
the expensive half of the pipeline for rejected faces, and it stops the UI from
printing "SPOOF — matched Kaitlyn", which tells an attacker their spoof found the right
target. A test asserts rejected faces are never embedded, via a call counter.

**Trust fails closed.** `is_trustworthy` requires a match *and* a passed liveness
check. Missing liveness ⇒ untrusted. Enrollment exits non-zero rather than enroll from
what might be a photo, and rejects faces that are too small or too blurry (Laplacian
variance) — a soft enrollment embedding poisons every comparison made against it
afterward.

**The threshold travels with the model.** `train_summary.json` is copied alongside the
exported weights, and inference records where its threshold came from. Otherwise the
number in a report and the number the demo runs at diverge silently, which is the same
class of bug as the preprocessing mismatch.

**Export is verified, not assumed.** ONNX and CoreML outputs are checked against
PyTorch over batch sizes 1, 1, and 4 — the repeat catches nondeterminism, and batch 4
catches a dynamic axis that got frozen at export. FP32 must agree to 1e-4; INT8 gets a
looser bound and prints the caveat that a quantized model is a *different* model whose
metrics need re-measuring, not inheriting.

**Augmentation chosen not to destroy the signal.** PAD cues are texture — moiré, print
halftone, specular highlights. Heavy blur erases them, so blur stays mild. Aggressive
color jitter can manufacture screen-like tints on genuine faces, teaching a cue that
isn't there. JPEG recompression is deliberately *included*: real cameras deliver
compressed frames, and training through that prevents a model that only works on
pristine captures.

---

## Roadmap

- [ ] Collect a real multi-subject, multi-session PAD dataset and publish the metrics
- [ ] Raspberry Pi 4/5 deployment with measured latency, not projected
- [ ] TensorRT INT8 path on NVIDIA, with accuracy re-measured post-quantization
- [ ] Cutout and 3D-mask attack types (the schema already supports them)
- [ ] Cross-dataset evaluation — train on one capture session, test on another entirely

## Contributing

The most useful contribution is **PAD data from a device or lighting condition I don't
have**, or a report of an attack that gets through. Both are more valuable than
features. Issues and PRs welcome.

## License

MIT.
