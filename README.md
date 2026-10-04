# The Page That Isn't Flat

*Jack Gilman and Andrew Gilman*

**[Open the interactive page](https://ahggggg.github.io/page-that-isnt-flat/)** · **[Download the film (2:09)](https://github.com/Ahggggg/page-that-isnt-flat/releases)** · **[Run the trace check on your own segment](REPRODUCE.md)**

A virtually unwrapped Herculaneum page looks flat. The papyrus inside the scan is not. On PHerc. 139, segment w035, we measured
from the CT alone how far the real sheet strays from its traced surface, and how far the organizers' canonical 2.4 µm ink model
can have its 62-layer reading window moved off that surface before it stops finding the organizers' labelled letters. We then
checked that tolerance on seven more labelled pages of the same scroll.

The interactive page is the report: scroll through it. The first minute is the film, scrubbed by the scroll wheel; every later
scene is drawn live from the measured data, and you can drag the cut, slide the reading window, flip the sheet and point at the
sheet-offset map.

## What it shows, and how sure

- **The sheet wanders, but the model's slab mostly holds it.** On a 17 × 49 mm band of w035, 32.5% of the papyrus sits more than
  24 µm off its traced surface and 6.6% more than 48 µm, yet 99.2% stays inside the 62 layers (about 150 µm) the model reads.
  This uses the CT alone, no ink model.
- **Usually about ±40 µm.** With its window 38 µm off the trace, the model still separates the organizers' labelled ink from
  papyrus within 0.05 AUC of its on-trace score on 6 of 8 labelled page samples (6 mm crops) and on 3 of 4 full labelled regions.
- **Not on every page.** Over its full labelled region, w044 loses 0.095 AUC at −38 µm (95% interval 0.067–0.123). We tested a
  tilted sheet, a model that is sensitive to depth everywhere, and misplaced labels; none is supported, so the cause is
  undetermined.
- **Not to ±55 µm.** Every sample lost more than 0.05 AUC on one side at ±55 µm, and which side varies by page.
- **On w035's first labelled region the tolerance is wider,** about ±50 µm (AUC 0.91 on the trace, 0.88–0.89 at ±48 µm). The
  map's finer detail changes sooner: its agreement with the organizers' published map stays at r ≥ 0.89 only within ±20 µm.
- **Control.** The same 62 layers fed back to front lose the writing (agreement r = −0.02; AUC 0.44, at chance).
- **A prediction that failed.** We predicted, before running it, that the depth where the model reads best would follow the
  sheet from one millimetre to the next. It does not (Spearman ρ = 0.01 across 598 one-millimetre blocks). A separate test on
  seven more pages found no evidence that snapping the trace onto the papyrus with the CT raises the model's score (effects
  below about 0.03 are not resolved).

**Why it matters.** The organizers ask for a way to tell "no ink" apart from "no ink recovered yet". When a model finds no ink on
a traced sheet, one suspect is always the trace itself. These measurements give a working rule for this model at 2.4 µm: within
about 40 µm of the sheet the trace is usually not why ink goes missing (w044 is an exception), while traces 100 µm off, as
reported for meshes carried over from another scan (TAUIL Abd Elilah, ScrollPrize/villa issue #1912), can lose much of it.
`REPRODUCE.md` turns this into a check you can run on any labelled segment.

**Limits.** One scroll (PHerc. 139, 2.399 µm scan) and one model. The organizers' labels were probably among this model
family's training labels, so the AUCs are not held-out accuracies. On 6 mm crops, AUC differences below about 0.03–0.05 are not
resolved. The published map is itself this model's output, so agreement with it favours the traced surface. The brightest-layer
sheet estimate can pick a neighbouring layer where sheets touch. Every test was preregistered, and the page reports the
predictions that failed.

## Run the trace check yourself

`tracecheck` measures both things on any labelled PHerc0139 segment, with only torch, numpy, scipy, pillow and tifffile (plus a
zstd decoder for the labels on Python older than 3.14). Every download is checked against a pinned hash or the bucket's own.
On a 3.7 mm demo square of w035 it runs in about 15 minutes on a laptop GPU; [REPRODUCE.md](REPRODUCE.md) has every command, the
numbers you should get, runtimes, disk, the cloud cost of the full band, and the pitfalls we hit.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests          # synthetic tests, no data
python -m tracecheck fetch   --segment w035 --region 8960 8704 1536 1536 --model
python -m tracecheck depth   --segment w035 --region 8960 8704 1536 1536 --out runs/demo
python -m tracecheck offsets --segment w035 --region 8960 8704 1536 1536 --out runs/demo --offsets -16 -8 0 8 16 --reverse-null
python -m tracecheck score   --segment w035 --region 8960 8704 1536 1536 --out runs/demo --alignment
python -m tracecheck report  --out runs/demo
```

Before publishing, the toolkit was checked against our earlier, separate pipeline on the same pixels: the CT-only depth is
identical cell for cell, and label AUCs agree to 0.0002.

## What is here

| Path | What it is |
|---|---|
| `index.html`, `assets/`, `film/` | The interactive page (served by GitHub Pages). |
| `REPRODUCE.md`, `tracecheck/`, `tests/`, `requirements.txt` | The trace check: step-by-step guide, code and tests. |
| `FILM.md`, `film-making/` | How the film and the scroll-driven page were made. |
| `LICENSE` | MIT, for the code and text in this repository. |
| `NOTICE.md` | Scan-derived images and the film: CC BY-NC 4.0, from Vesuvius Challenge data; third-party code and weights. |

To view the page from a local copy, serve the folder (the film loads with `fetch`, which browsers block for `file://` pages):

```text
python -m http.server 8000
```

then open http://localhost:8000.

## Credit

Scroll data, meshes, surface volumes, the published prediction and the ink labels used for scoring: Vesuvius Challenge, CC BY-NC
4.0. Model: `scrollprize/ink_canonical_2um` (MIT). Prior work this builds on: window centring by flummoxjr, kadenpool and
AndreasHad04 (ScrollPrize/villa issues #1765, #1766, #1813); transferred meshes and depth search by TAUIL Abd Elilah (#1912); the
sheet's wander off the trace, noted in the organizers' Hecate model card and 2026 open-problems page. Code and documentation were
written with AI assistance and checked against the measurements; no papyrological reading is claimed.
