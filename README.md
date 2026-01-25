# Nika
Some code - maybe even good code, who knows?

## Quick Start

Run the training environment:
```bash
sudo docker compose run --rm --entrypoint bash backend
```

## GitHub Pages (Local Development)

The `docs/` folder contains an interactive website for exploring model visualizations.

**To run locally:**
```bash
# From the repo root
python -m http.server 8000 -d docs

# Then open http://localhost:8000 in your browser
```

**What's included:**
- Model gallery with PSNR scores
- Side-by-side comparison videos (GT / Prediction / Residual)
- Error over time plots (PSNR/MSE per frame)
- Spatial error heatmaps

## Git LFS

This repo uses Git LFS for large files. Install it before cloning:
```bash
git lfs install
git clone <repo-url>
```

**Tracked file types:**
- `docs/assets/videos/*.mp4` - Comparison videos for GitHub Pages
- `*.zip` - Model archives

## TODO
- Ablation error visuals / residuals in video form
- Error over time visual (line plot)
- Average error for each spatial position over course of entire video
- Visuals for ablations (side by side over time?)
- Put together github pages for visuals
- Something like attribution maps for main model branches
- Same plots produced over full dataset + overlaid into one plot
- Bar chart for key metrics
- Investigate why error is highest at beginning (if it is?)
- Motion vs error?
- Rent A100 and get FPS numbers