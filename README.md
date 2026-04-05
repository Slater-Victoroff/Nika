# Nika

For the actual reproduction workflow, start with
[docs/reproduction.md](/home/m/side-projects/Nika/docs/reproduction.md).

To reproduce the UVG results with Docker from the repo root:

```bash
docker compose build
docker compose run --rm backend bash
python3 scripts/prepare_uvg.py
bash scripts/reproduce_uvg_scale.sh small
```

Inside the container, the code is mounted at `/app` and `static/` is mounted at
`/app/static`.
