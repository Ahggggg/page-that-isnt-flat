# How the film and the page were made

The film (2:09, under Releases) and the scroll-driven page are built from the same public data as the measurements, with
`tracecheck` doing the downloads and the model runs. The scripts in `film-making/` are the ones that made this film,
adapted to read `tracecheck`'s files. We re-ran each data script on the inputs the film was made from: they reproduce the
film's data files exactly, except two images noted under "What does not reproduce exactly".

## What is measured, and what is animation

| Scene | What you see | Made from | Liberties (also stated on screen or in the credits) |
|---|---|---|---|
| Title | the model's ink map of the page, dimmed and blurred, behind the title | published prediction | presentation only |
| The scroll | PHerc. 139 from its own CT, cut open at one height; w035 glows where it crosses the cut, then lifts off | whole-scroll CT (76.8 and 38.4 µm levels), the traced mesh | lighting, materials, the lift-off and the glow are presentation |
| Unrolling | the traced sheet in its true 3D shape becomes the flat page | the organizers' mesh, textured with the published prediction | the motion between the two shapes is animation, not a physical unrolling |
| The page, one letter | a continuous zoom from 49 mm down to papyrus fibres | crops of the published prediction, then of the raw CT on the traced surface | at letter scale the ink map dissolves into the CT |
| Not flat | the page tilted: the real sheet rising and dipping around the trace | CT-only sheet depth | height exaggerated ×6 and smoothed over about 70 µm |
| The cut | CT straight down through the page along a cut line | the surface volume (109 layers) | depth stretched for visibility, growing on screen from true scale |
| The window | the model's map as its 62-layer window moves off the trace, with the two curves | one model run per offset | the film crossfades between neighbouring runs; the page shows runs one at a time |
| Flip it | the same layers fed back to front | two model runs | none |
| The whole band | CT-only sheet offset over 17 × 49 mm | CT-only sheet depth | colours smoothed over about 0.1 mm; percentages use the unsmoothed depth |

No letters or ink were drawn, painted or enhanced; every number on screen comes from the measurements.

## Steps

You need the Python packages in `requirements.txt`, Node.js 20 or newer (for Remotion), and Blender 4.5 LTS on a machine
with a GPU (for the three 3D shots). Paths below are from the repository root.

### 1. Data

The film uses the w035 band (rows 8192–15359, all columns) and its model runs at every window offset. The model runs
need a GPU; REPRODUCE.md, section 6, gives the time and cost.

```text
python -m tracecheck fetch   --segment w035 --region 8192 0 7168 20400 --model --workers 24
python -m tracecheck depth   --segment w035 --region 8192 0 7168 20400 --out runs/band
python -m tracecheck offsets --segment w035 --region 8192 0 7168 20400 --out runs/band --device cuda --batch 8 --offsets -23 -20 -16 -12 -8 -6 -4 -2 0 2 4 6 8 12 16 20 23 --reverse-null
python -m tracecheck score   --segment w035 --region 8192 0 7168 20400 --out runs/band
python film-making/fetch_film_inputs.py       # the traced mesh (14 MB) and the whole-scroll CT at two coarse levels (about 2 GB)
```

`film-making/filmdata.py` says where each script looks; set `TRACECHECK_DATA`, `FILM_BAND_RUN` or `FILM_INPUTS` to point
elsewhere. For the takeaway card's other-pages numbers, run `tracecheck` on each labelled page (REPRODUCE.md, section 6),
then `film-making/data/pages.py` (step 2).

### 2. The film's data files

Each script writes into `film-making/app/public/data/` (what the film reads) or `film-making/blender/data/` (what the 3D
shots read). Run them in this order; together they take a few minutes on a laptop.

```text
python film-making/data/sheet_and_ink.py    # the traced sheet (3D and flat positions) and the published prediction as a texture
python film-making/data/band.py             # the text line and 9.8 mm window, chosen by rule from the published map, and its CT cross-sections
python film-making/data/zoom.py             # the five zoom levels, centred by rule on the brightest letter in the window
python film-making/data/relief.py           # height and textures for the 'not flat' shot
python film-making/data/slice.py            # the cut through the whole scroll and where w035 crosses it
python film-making/data/maps.py             # the model's map at each window offset, along the line and over the whole band
python film-making/data/depth_band.py       # the whole-band sheet-offset map
python film-making/data/film_numbers.py     # the film's numbers (page.json)
python film-making/data/pages.py --full w030=runs/w030 w035=runs/w035 w039=runs/w039 w044=runs/w044 --crops w030=runs/w030-crop ...
```

### 3. The 3D shots (Blender)

Three shots: `scroll` (360 frames), `unroll` (300) and `relief` (360). Preview one frame locally, for example

```text
blender -b --factory-startup -P film-making/blender/unroll.py -- --out previews/unroll --preview --still 150
```

and render the full shots on a GPU with `film-making/render_job.sh`, which installs Blender 4.5.14 (checked by SHA-256),
times one frame per shot, then renders all three at 256 samples. On one rented RTX 4090 this took 77 minutes of machine
time, about USD 1. `BLENDER_SCALE=2` renders at 3840 × 2160. The first run of `scroll.py` builds two OpenVDB half-volumes
from the whole-scroll CT and caches them in `film-making/blender/data/`. Put the frames in
`film-making/app/public/frames/<shot>/f_0001.png` and so on.

### 4. The film (Remotion)

```text
cd film-making/app
npm ci
npx remotion studio                                              # preview in the browser
npx remotion render Main out/the-page-that-isnt-flat.mp4         # the released film, with captions
npx remotion render Main out/film-nocaptions.mp4 --props '{"captions":false}' --crf=16    # for the page
```

The scene order, lengths and on-screen text are in `app/src/beats.json`; the page shows the same text as call-outs. A full
render took 3 minutes 44 seconds on our laptop.

### 5. The page

```text
python film-making/build_scrub_film.py
```

This encodes the first 60 seconds of the caption-free film with a keyframe every 6 frames, so the page can show any
moment while you scroll, and splits it into parts under 15 MB in `film/` (about 35 seconds). `index.html` maps the page's
scroll position to the film's time for the first six chapters and draws the later chapters live from `assets/`. Serve the
folder (`python -m http.server`) to view it.

## The page's own data files

The page reads `assets/page.js`, `assets/pages.js` and the images in `assets/`. They are included as we made them.

- `page.js`: the band run's numbers (agreement with the published map and label AUC per window offset, coverage), plus
  two numbers from our per-millimetre block analysis that `tracecheck` does not compute: the Spearman correlation between
  the CT-only sheet depth and the depth at which the model reads best, and the number of blocks.
- `pages.js`: the other pages' AUC per offset and losses at ±38 µm, with 95% intervals from a bootstrap over
  one-millimetre blocks. `tracecheck` reports the losses but not those intervals.
- `depth_data.png`: for every third cell of the band, the smoothed sheet offset (red channel: twice the offset in layers,
  plus 128; 0 = no data) and the unsmoothed offset class (green channel: 1 within 24 µm, 2 within 48 µm, 3 beyond,
  4 outside the model's layers, 0 not measured). Sampling every third cell keeps the class shares within 0.05 percentage
  points of the full-resolution ones.
- `arm_*.jpg` are the line-window strips from `maps.py`; `band_0.jpg` and `band_R0.jpg` are the whole-band maps resized to
  1700 px; `xs_sprite.jpg` stacks the 81 cross-sections from `band.py`, each resized to 2048 × 109; the `thumb_*.jpg`
  images come from our September searches of other scrolls and are included as made.

## What does not reproduce exactly

- `slice_glow.png` (the glow along w035 in the scroll shot): the one-off step that made it was not kept. `slice.py` rebuilds
  it from w035's crossing points (blurred, sigma 3 px); the rebuild correlates 0.986 with the film's image.
- `depth_band.png`: `depth_band.py` matches the film's image to within 3 grey levels (mean 0.46), not bit for bit.
- The released film's numbers came from our earlier pipeline. `film_numbers.py` uses `tracecheck`'s definitions, which
  agree with it to about 0.0003 (REPRODUCE.md, section 4) and can differ in the last printed digit.
- The page's data files, above.

## Cost and time (ours)

| Step | Where | Time | Cost |
|---|---|---|---|
| model runs over the band, 18 window positions | rented 2 × RTX 4090 | about 3 hours | about USD 4.70 |
| film data scripts | laptop | minutes | — |
| Blender shots | rented 1 × RTX 4090 | 77 minutes | about USD 1 |
| Remotion render | laptop | 4 minutes | — |
| page encode | laptop | 35 seconds | — |

Blender is GPL software and Remotion has its own licence (a company licence is required for some commercial uses); neither
is included here. The data-derived images and the film are CC BY-NC 4.0 (`NOTICE.md`).
