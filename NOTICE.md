# Notices: data, weights and third-party code

The code and text written for this repository are under the MIT licence in `LICENSE`. Other material keeps its own terms.

## Images and film derived from Vesuvius Challenge data (CC BY-NC 4.0)

The page's images (`assets/`), its film parts (`film/`) and the released film are derived from Vesuvius Challenge public data
for PHerc. 139 (segment w035 and the labelled pages w030, w039–w041, w043–w045): the CT scan and its surface volumes, the
organizers' traced mesh, their published ink prediction, and our runs of their `scrollprize/ink_canonical_2um` model on that data.
That data is licensed CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), so these derived images and the film are
too: non-commercial use only, with attribution to the Vesuvius Challenge.

The organizers' ink labels were used only to compute scores. No label image is included in this repository.

## Downloaded, not included

`python -m tracecheck fetch` downloads the following from their public locations and checks every file it keeps
(see `REPRODUCE.md`). They are not redistributed here.

1. `scrollprize/ink_canonical_2um` checkpoint, revision 075855bc69317ef6febf39a0d9d687b27d2b7c29
   (https://huggingface.co/scrollprize/ink_canonical_2um). MIT License per the model card.
2. `ScrollPrize/villa` source files, commit af8bac80e3c04ba7f1af19a6aef22bf720874949 (https://github.com/ScrollPrize/villa).
   MIT License, Copyright (c) 2024 Vesuvius Challenge. Three small files (the ResNet3D-152 3D-decoder model definition, its
   backbone and an empty package `__init__`) from `ink-detection/optimized_inference` at the pinned commit, staged locally and
   hash-checked.
3. Vesuvius Challenge public data (https://vesuvius-challenge-open-data.s3.amazonaws.com/): surface volumes, ink labels and
   supervision masks, and published ink predictions for PHerc0139 segments. CC BY-NC 4.0. Maps, depth arrays and scores you
   compute from it are derived material under the same terms; the toolkit's report figure draws curves only.

## Film tools

The film was made with Blender (GPL; https://www.blender.org) and Remotion (https://www.remotion.dev; its own licence, which
requires a company licence for some commercial uses). Neither is included here; `film-making/` holds only our scripts.
