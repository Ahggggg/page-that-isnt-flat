# Reproducing a trace check

This page takes you from a clean checkout to the two measurements, with the numbers you should expect, how long
each step takes, how much disk it needs, and the mistakes that are easy to make.

**What is measured.** On a labelled segment of PHerc0139 (scanned at 2.399 um per voxel):

1. *How far the papyrus sits from the traced surface, from the CT alone.* For every 9.6 um cell (4 x 4 pixels of
   the organizers' surface volume) the toolkit finds the brightest of the 109 layers, after light smoothing and a
   mild preference (a Gaussian of sd 30 layers) for layers near the trace. It reports the share of the measured area
   whose brightest layer is more than 10 layers (24 um) or 20 layers (48 um) off the trace, and the share outside
   the 62 layers the ink model reads.
2. *How the ink model's reading changes when its window moves off the trace.* The organizers' canonical 2.4 um model
   reads 62 consecutive layers. Centred on the trace, those are layers 23-84 of 109. The toolkit moves the window k
   layers (k x 2.4 um) and scores each map against the organizers' ink labels (AUC: the chance that a labelled ink
   cell scores above a labelled papyrus cell; 0.5 is chance) and against the organizers' published prediction
   (Pearson r).

**What it is not.** An AUC is a separation score on labelled pixels, not recovered text. The labels may have been
used to train this model, so no AUC here is a held-out accuracy.


## 1. Install

Python 3.10 or newer. Install the PyTorch build for your machine first
(<https://pytorch.org/get-started/locally/>; CPU, CUDA, Intel XPU and Apple MPS builds all work), then:

```text
git clone <this repository> trace-check
cd trace-check
python -m pip install -r requirements.txt
python -m pip install zstandard          # only on Python older than 3.14: the label arrays are zstd-compressed
python -m unittest discover -s tests     # 51 synthetic tests, no data, under 10 s
```

The tests build small fake volumes, label arrays and a local stand-in for the download server; they check software
behaviour only (window geometry, tiling, label decoding, depth, scores, download checks), not the real model.

## 2. Fetch the demo inputs

The demo region is a 1536 x 1536 pixel square (3.7 mm) inside the organizers' labelled area of segment w035:
rows 8960-10495 and columns 8704-10239 of the surface volume, given as `--region 8960 8704 1536 1536`
(first row, first column, height, width; zero-based).

```text
python -m tracecheck fetch --segment w035 --region 8960 8704 1536 1536 --model
```

| What | Size | Pinned by |
|---|---|---|
| model checkpoint (`scrollprize/ink_canonical_2um`, fixed revision) | 1.55 GB | SHA-256 |
| three network-definition files from `ScrollPrize/villa` (fixed commit) | 16 KB | SHA-256 |
| 144 CT chunks of the region (109 x 128 x 128 bytes each) | 257 MB | SHA-256 |
| ink labels and supervision mask (one shard each) and their metadata | 0.15 MB | SHA-256 |
| the organizers' published prediction for w035 | 41 MB | SHA-256 |

About 1.9 GB in all, into `./data/` (change with `--data-dir`). Files already present with the right hash are not
fetched again; `fetch ... --check-only` re-checks everything on disk without downloading. Every other segment or
region works the same way (`--segment w030`, `w039`, `w040`, `w041`, `w043`, `w044`, `w045` are built in, or
`--prefix PHerc0139/segments/<id>` for any segment): label files of the eight built-in pages are pinned by SHA-256;
CT chunks outside the demo region are checked against the bucket's MD5 and byte count, and their SHA-256 is written
to `data/receipt.json`. `fetch --segment <name> --list` shows what the bucket holds for a segment (for example the
file name of its published prediction, which you then pass with `--published`).

## 3. Run

```text
python -m tracecheck depth   --segment w035 --region 8960 8704 1536 1536 --out runs/demo
python -m tracecheck offsets --segment w035 --region 8960 8704 1536 1536 --out runs/demo --offsets -16 -8 0 8 16 --reverse-null --device cpu
python -m tracecheck score   --segment w035 --region 8960 8704 1536 1536 --out runs/demo --alignment
python -m tracecheck report  --out runs/demo
```

Use `--device cuda`, `--device xpu` or `--device mps` for a GPU. `offsets` resumes: a finished offset whose map
matches its recorded hash is skipped. Outputs in `runs/demo/`:

- `depth.json`, `depth.npz`: the CT-only depth per cell and the coverage shares, over all measured cells and over the
  label-scored cells;
- `maps/k-16.npy` ... `maps/k+16.npy`, `maps/R+0.npy`: float32 probability maps (one value per 4 x 4 pixels), each
  with a receipt `maps/<name>.json` (window, tiles, device, timing, SHA-256); `maps.json` holds the run's settings;
- `score.json`: per offset the label AUC, the ink-minus-papyrus mean probability, r with the published map (raw
  and smoothed), the tolerance, and the null checks;
- `report.json`, `report.png`: the summary and a figure of curves (no CT or label image).

Coordinates: output cell (r, c) of a map covers surface-volume pixels (Y0 + 4r .. Y0 + 4r + 3, X0 + 4c .. X0 + 4c + 3).

## 4. Expected results (demo region)

These are the numbers our run of exactly these commands produced (laptop, Intel Arc integrated GPU, FP32). On
another machine expect map values to agree to about 1e-6 in strict FP32 and the scores below to agree to the
printed three digits.

Labels in the region: 44,306 ink cells and 63,167 papyrus cells (9.9 mm2 scored).

| Window offset | Label AUC | Ink minus papyrus (mean probability) | r with published map (raw / smoothed) |
|---|---|---|---|
| -38.4 um (k = -16) | 0.861 | 0.380 | 0.890 / 0.915 |
| -19.2 um (k = -8) | 0.862 | 0.364 | 0.935 / 0.955 |
| 0 (on the trace) | 0.861 | 0.359 | 0.994 / 1.000 |
| +19.2 um (k = +8) | 0.866 | 0.353 | 0.951 / 0.973 |
| +38.4 um (k = +16) | 0.839 | 0.302 | 0.874 / 0.917 |
| 0, depth order reversed | 0.419 | -0.011 | -0.173 / -0.183 |

- Checks against chance: with the labels rotated 180 degrees inside the region the on-trace AUC is 0.489; with the
  model's depth order reversed it is 0.419. The organizers' published map scores 0.865 on the same cells.
- Tolerance: the AUC stays within 0.05 of its on-trace value over every tested offset, -38 to +38 um.
- Label registration check: the best in-plane label shift is (8, -4) cells and beats no shift by 0.013 AUC, below
  the 0.02 that `score` flags, but it lies at the edge of the searched range (flagged `best_at_search_edge`).
- CT-only depth, all measured cells (13.5 mm2): median +1 layer; 27.4% of the area more than 10 layers (24 um) off
  the trace, 5.2% more than 20 layers (48 um), 1.3% outside the model's window. Label-scored cells (9.9 mm2):
  28.0%, 4.6% and 0.30%.

What the curve says here: on this 3.7 mm square the model separates the labelled ink from papyrus about equally
well anywhere within 19 um of the trace and loses about 0.02 AUC at +38 um, while its map drifts away from the
published one (r 0.99 to 0.87-0.89) much faster than its AUC falls. Over the whole labelled part of the band
(about 30 mm2, below) the AUC is about 0.05 higher than on this square; small regions differ.

**Agreement with our earlier, separate pipeline on the same pixels.** We checked this toolkit against values we had
computed before with our own scripts:

- Organizers' tiling (stride 128), away from the region edge: label AUCs agree to 0.0002 and r with the published
  map to 0.0003 at every offset. Map values differ by up to 0.009 (mean 0.0004). That gap is in the earlier maps:
  this toolkit on CPU and on the Arc GPU agree to 6e-7, and both differ from the earlier maps by the same amount.
  The earlier maps came from an RTX 4090 with a script that did not turn off TensorFloat-32, PyTorch's default for
  convolutions on NVIDIA GPUs, which most likely explains the gap (see pitfalls); we did not re-run it on an NVIDIA
  GPU to confirm. For scale, moving the window by 2 layers changes map values by 0.05 on average and up to 0.69.
- Stride-256 tiling in FP16: map values agree to one float16 step (0.00024), AUCs to 0.000002.
- CT-only depth: identical, cell for cell, everywhere at least 12 cells from the region edge.
- Repeat runs on the same machine gave bit-identical maps.

## 5. Time and disk

Measured on our laptop (Intel Core Ultra 9 288V with its Arc 140V integrated GPU, 32 GB RAM, Windows 11,
PyTorch 2.14):

| Step | Time |
|---|---|
| unit tests | 2 s |
| `fetch` (demo, 1.9 GB) | depends on your connection |
| `fetch --check-only` | 2 s |
| `depth` | 3 s |
| `offsets`, 6 window positions, FP32, stride 128 (121 tiles each) | Arc GPU: 13 min (1.0-1.1 s per tile; up to 1.7 s per tile when the laptop was warm). CPU: about 5.5 s per tile, so about 66 min |
| `offsets`, 5 positions, FP16, stride 256 (36 tiles each) | Arc GPU: 1.4 min (0.40-0.46 s per tile) |
| `score --alignment` | 4 s |
| `report` | 2 s |

On a CPU, run fewer offsets first (`--offsets 0 16`), since each offset costs about 11 minutes there. Disk: the
demo inputs take 1.9 GB under `data/`; a run directory takes under 10 MB. Peak memory of `offsets` on the demo was
about 9 GB with the default batch of 4 tiles (lower it with `--batch 1`).

## 6. The full band and other pages (cloud)

The band we measured on w035 is rows 8192-15359 and all 20,400 columns (17 x 49 mm): 8,332 CT chunks (14.9 GB) and
8,745 tiles per window position at the organizers' stride.

```text
python -m tracecheck fetch   --segment w035 --region 8192 0 7168 20400 --model --workers 24
python -m tracecheck depth   --segment w035 --region 8192 0 7168 20400 --out runs/band
python -m tracecheck offsets --segment w035 --region 8192 0 7168 20400 --out runs/band --device cuda --batch 8 --offsets -23 -20 -16 -12 -8 -6 -4 -2 0 2 4 6 8 12 16 20 23 --reverse-null
python -m tracecheck score   --segment w035 --region 8192 0 7168 20400 --out runs/band --alignment
python -m tracecheck report  --out runs/band
```

With two GPUs, start two `offsets` processes on the same `--out` with different `--offsets` (one with
`--device cuda:0`, one with `--device cuda:1`); each writes its own per-offset receipts. `offsets` stages the band's
CT in a 16 GB file under `--out` for each process.

**Cost.** Our own full-band run (18 window positions, the 17 above plus the reversed order) took about 3 hours on a
rented machine with two RTX 4090 GPUs at about USD 1.48 per hour: about USD 4.70 in all, including USD 0.37 for a
first attempt that did not start properly. That run used our earlier script, which did not turn TensorFloat-32 off, at
about 0.066 s per tile on a healthy GPU (18 x 8,745 tiles is about 2.9 GPU-hours). With this toolkit's default
(`--precision fp32`) expect roughly 5-8 GPU-hours (strict FP32 on an RTX 4090 measured 0.11-0.18 s per tile
including reading the data), about USD 4-6 at USD 0.74 per GPU-hour; `--precision tf32` reproduces our earlier
run's arithmetic and speed. These are estimates from our measured rates, not a run of this toolkit on the full
band: the toolkit itself was checked on the demo region only.

**What we measured on the full band** (with the earlier pipeline, which agrees with this toolkit as described in
section 4):

- CT-only depth over the band: median -1 layer; 32.5% of measured cells more than 10 layers (24 um) off the trace,
  6.6% more than 20 layers (48 um), 0.77% outside the model's window.
- Label AUC over the labelled part of the band (about 30 mm2; rows 8192-10804, columns 2249-11323; 103,110 ink and
  218,425 papyrus cells), window offset in um: -55: 0.841, -48: 0.893, -38: 0.905, -29: 0.923, -19: 0.916,
  -14: 0.915, -10: 0.917, -5: 0.921, 0: 0.913, +5: 0.916, +10: 0.917, +14: 0.918, +19: 0.909, +29: 0.901,
  +38: 0.891, +48: 0.882, +55: 0.868; reversed order 0.438; published map 0.914; labels rotated 180 degrees 0.456.
- Agreement with the published map over the whole band (smoothed r): -55 um 0.461, -38 0.707, -19 0.896,
  -10 0.903, 0 1.000, +10 0.907, +19 0.890, +38 0.734, +55 0.551; reversed -0.022.

**Other pages.** `--segment w030`, `w039`, `w040`, `w041`, `w043`, `w044` and `w045` work the same way; `score` uses
only supervised cells, so any rectangle that covers the labelled area works (a tighter one costs fewer tiles). On 6
mm squares of those pages we found the AUC within 0.05 of its on-trace value to +-38 um on five pages and falling by
0.10-0.19 at -38 um on w039 and w044; on the whole labelled region of w044 the fall at -38 um was 0.095, on w039
0.038 (those runs used stride 256 in FP16, see pitfalls). A full labelled page needs on the order of 200-600 tiles
per window position at stride 256, or four times that at the organizers' stride.

## 7. Pitfalls

- **Window centring.** The organizers' 2.4 um surface volumes have 109 layers with the traced surface at layer 54
  (zero-based). The model's 62-layer window centred on it is layers 23-84. The organizers' own inference README
  suggests starting at layer 1, which puts the window 22 layers (53 um) below centre; several community reports in
  September 2026 showed that the centred window reproduces the published maps and the README window does not.
  Offsets in this toolkit are always counted from the centred window, so `--offsets 0` is on the trace, and they
  must lie between -23 and +24 on a 109-layer volume. Another volume depth needs `--mesh-plane`.
- **Tile grid and edges.** The model reads 256-pixel tiles at a 128-pixel stride and blends overlaps. Near a
  region's edge fewer tiles overlap, so the outer 128 pixels (32 cells) of any region differ slightly from a map
  computed over a larger area; the organizers' published map was computed over the whole segment with tiles at
  multiples of 128. Start regions on multiples of 128 and judge agreement away from the edges. `--stride 256`
  (no overlap, 4 times fewer tiles) is faster but is a different map.
- **TensorFloat-32 on NVIDIA GPUs.** PyTorch lets cuDNN convolutions on recent NVIDIA GPUs run in TensorFloat-32
  (about three significant digits) unless told otherwise, so an "FP32" run on CUDA is not FP32 by default. This
  toolkit's `--precision fp32` turns TensorFloat-32 off; `--precision tf32` turns it on. On the demo region the two
  kinds of map differ by up to 0.009, which moved AUCs by under 0.0002.
- **Tiling changes scores.** Tiles at stride 256 (no overlap) are four times cheaper but are a different map: on the
  demo region they scored 0.018-0.030 lower AUC than the organizers' stride-128 tiling at every offset (0.837 vs
  0.861 on the trace; both on the same cells, the stride-256 run in FP16). Compare offsets within one tiling only.
- **Label coverage.** Labels exist only inside the organizers' supervision mask; an unlabelled pixel is not papyrus.
  A cell is scored only if all 16 of its pixels are supervised; it counts as ink if at least half its pixels are
  ink, as papyrus if none are, and is dropped if mixed. On w035 the supervised area inside the band we measured is
  about 30 mm2 (rows 8192-10804 only). Small regions give noisy AUCs: on 6 mm squares, differences below about
  0.03-0.05 were not resolved in our runs, and `score` warns below 20,000 ink or papyrus cells.
- **Training labels.** w035 is listed among the organizers' ink-training entries, and the labels on all eight
  built-in pages are probably training labels for this model family. A high AUC shows the map separates the
  labelled classes; it is not a held-out accuracy, and a loss of AUC with offset is a property of this model on
  this page, not a general rule. On PHerc0139 the tolerance differs from page to page: on two of seven other
  labelled pages the AUC fell by 0.10-0.19 at -38 um in our 6 mm crops, while w035 lost almost nothing.
- **Label registration.** The label arrays have the same height and width as the surface volume, and the toolkit
  treats label pixel (y, x) as surface-volume pixel (y, x). The label metadata also carries a `canvas_offset_yx`
  attribute, which is not applied. `score --alignment` re-scores the on-surface map with the labels moved up to
  8 cells (77 um) in each direction; a best shift well above zero shift means misregistration that biases every
  offset's AUC. Over w035's labelled band region the best shift beat zero shift by under 0.005.
- **Brightest layer is not always the traced sheet.** In tightly wound papyrus the brightest nearby layer can be the
  next winding. The prior (sd 30 layers) only discourages that, and cells whose smoothed peak is 60 or less are not
  measured. The depth is a CT-only estimate, not a re-trace.
- **Axis order.** Arrays are (layers, rows, columns). The surface volume's `.zattrs` lists `canvas_size` as
  [width, height], while the label metadata lists [height, width]; the toolkit relies on array shapes only.
- **Missing chunks.** The bucket stores no chunk where a surface volume is empty (outside the traced sheet). The
  toolkit reads a missing chunk as empty only when `fetch` recorded that the bucket does not have it; any other
  missing file stops the run, so an interrupted download cannot pass as empty papyrus.
- **Reduced precision.** `--precision fp16` or `bf16` is several times faster on GPUs but changes map values by up
  to about 0.01; the published maps match FP32.
- **Memory.** Regions larger than `--ram-gb` (default 4 GB of CT) are staged to a file `region.u8` under `--out`
  (the full band needs 16 GB of free disk for it, per process); `depth` works in strips and needs about 1 GB.
  Several `offsets` processes may share one `--out` if they run different offsets (each writes its own per-offset
  receipt and staging file); do not run the same offset in two processes at once.
- **Two-socket cloud machines.** In our full-band run, a process reading a memory-mapped volume whose pages sat on
  the other processor socket slowed to a crawl. If you split offsets across GPUs on such a machine, give each
  process its own copy of the volume or pin it to the socket that holds the data.
