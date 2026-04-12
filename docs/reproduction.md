# NIKA Reproduction Notes

This repo is research code. The actual training path is in `src/nika.py`, and the most useful reproduction helpers are now in `src/scripts/`.

## Installation

The existing Docker path is still the simplest supported setup:

```bash
docker compose build
docker compose run --rm backend bash
```

Inside the container, the repo source is mounted at `/app`.

## UVG Dataset Access

The paper draft refers to the standard 7-sequence UVG benchmark:

- Beauty
- Bosphorus
- HoneyBee
- Jockey
- ReadySetGo
- ShakeNDry
- YachtRide

Official source:

- UVG dataset page: https://tie-ultravideo.rd.tuni.fi/dataset.html
- License: CC BY-NC 3.0 https://creativecommons.org/licenses/by-nc/3.0/

The current helper script downloads the official 1920x1080 8-bit YUV RAW archives linked from that UVG page, then converts them into the PNG frame folders expected by the code.

## Scripts

### Prepare UVG

From the repo root:

```bash
python3 src/scripts/prepare_uvg.py
```

Inside the Docker container:

```bash
python3 scripts/prepare_uvg.py
```

This creates:

```text
static/benchmarks/uvg/
  beauty/
  bosphorus/
  honey/
  jockey/
  ready/
  shake/
  yacht/
```

### Train One Sequence

```bash
python3 src/scripts/train_nika.py \
  --dataset-root static/benchmarks/uvg \
  --video beauty \
  --config small \
  --device cuda:0
```

### Train One Full UVG Scale

```bash
bash src/scripts/reproduce_uvg_scale.sh small
```

The current config names in `src/configs.py` are:

- `xxs`
- `xs`
- `small`
- `medium`
- `large`

For the UVG results in the draft, the relevant ones are `small`, `medium`, and `large`.

### Evaluate One Checkpoint

```bash
python3 src/scripts/evaluate_checkpoint.py models/small-beauty-epoch1497-psnr33.26.torch
```

This prints:

- average PSNR
- decode FPS
- elapsed decode time

## Bunny Dataset Access

The draft also reports results on the Big Buck Bunny benchmark, using a single
1280x720 sequence with 132 frames.

For the Bunny path, use `scikit-video`, which exposes a sample Big Buck Bunny
video via `skvideo.datasets.bigbuckbunny()`.

Reference:

- scikit-video dataset helper: https://www.scikit-video.org/stable/modules/generated/skvideo.datasets.bigbuckbunny.html

The helper script below uses that sample video path and extracts the first 132
frames at 1280x720 into the layout expected by this repo.

### Prepare Bunny

From the repo root:

```bash
python3 src/scripts/prepare_bunny.py
```

Inside the Docker container:

```bash
python3 scripts/prepare_bunny.py
```

### Train Bunny

Use the same training wrapper, but point `--dataset-root` at `static/benchmarks`
and `--video` at `bunny`:

```bash
python3 src/scripts/train_nika.py \
  --dataset-root static/benchmarks \
  --video bunny \
  --config xxs \
  --device cuda:0
```

For the Bunny table in the draft, the relevant scales are:

- `xxs`
- `xs`
- `small`

To run the full Bunny scale sweep:

```bash
bash src/scripts/reproduce_bunny_scales.sh
```

### Evaluate Bunny

```bash
python3 src/scripts/evaluate_checkpoint.py \
  models/xxs-bunny-epoch1998-psnr31.06.torch \
  --dataset-root static/benchmarks \
  --video bunny
```

## Scope

This pass intentionally keeps the main source tree unchanged and limits the new workflow to:

- `src/scripts/*`
- this reproduction note

That keeps the repo diff small while still documenting dataset access, installation, and the concrete commands needed to rerun UVG experiments.
