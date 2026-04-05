# NIKA Reproduction Notes

This repo is research code. The actual training path is in `src/nika.py`, and the most useful reproduction helpers are now in `src/scripts/`.

## Installation

The existing Docker path is still the simplest supported setup:

```bash
docker compose build
docker compose run --rm backend bash
```

Inside the container, the repo source is mounted at `/app`.

If you want to prepare UVG inside that container, install one extra Python package first:

```bash
python3 -m pip install py7zr
```

That is only needed for extracting the official UVG `.7z` archives.

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

## Scope

This pass intentionally keeps the main source tree unchanged and limits the new workflow to:

- `src/scripts/*`
- this reproduction note

That keeps the repo diff small while still documenting dataset access, installation, and the concrete commands needed to rerun UVG experiments.
