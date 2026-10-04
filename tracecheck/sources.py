"""Where the public inputs live, and a download step that checks every file it keeps.

Local layout mirrors the public keys: an object `PHerc0139/segments/<id>/...` on the Vesuvius Challenge open-data
bucket is stored at `<data-dir>/PHerc0139/segments/<id>/...`. The model goes to `<data-dir>/model/`.

Checks, in order of strength: a SHA-256 pinned in manifest.json (the model, the vendored network code, and for the
preset segments the label arrays, published map, volume metadata and the demo region's CT chunks); otherwise, for
bucket objects, the MD5 the bucket reports as ETag (single-part uploads) and the byte count. Every kept file's SHA-256
is recorded in `<data-dir>/receipt.json`, and later runs check existing files against it. A file that fails a check
is deleted, never kept.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import labels as L
from .volume import SurfaceVolume

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
HEADERS = {"User-Agent": "trace-check/1.0", "Accept-Encoding": "identity"}
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
MD5_ETAG = re.compile(r'^"?([0-9a-f]{32})"?$')


class CheckFailed(RuntimeError):
    pass


def load_manifest(path=MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------ download
def download(url: str, dest: Path, sha256: str | None = None, allow_absent: bool = False, retries: int = 4,
             timeout: float = 120.0, opener=urllib.request.urlopen) -> dict | None:
    """Fetch url to dest through dest.part. Returns {"bytes", "sha256", "check"}, or None for HTTP 404 when
    allow_absent. Raises CheckFailed (and keeps nothing) when the byte count, ETag MD5 or pinned SHA-256 disagree."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(retries + 1):
        sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
        try:
            with opener(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as response, \
                    partial.open("wb") as stream:
                length = response.headers.get("Content-Length")
                etag = response.headers.get("ETag") or ""
                for block in iter(lambda: response.read(1 << 20), b""):
                    sha.update(block)
                    md5.update(block)
                    stream.write(block)
                    size += len(block)
            break
        except urllib.error.HTTPError as error:
            partial.unlink(missing_ok=True)
            if error.code == 404 and allow_absent:
                return None
            if error.code in (403, 404) or attempt == retries:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
        time.sleep(min(30, 2 ** attempt))
    checks = []
    try:
        if length is not None and int(length) != size:
            raise CheckFailed(f"{url}: got {size} bytes, Content-Length said {length}")
        checks.append("size")
        m = MD5_ETAG.match(etag.strip()) if url.startswith(S3) else None
        if m:
            if m.group(1) != md5.hexdigest():
                raise CheckFailed(f"{url}: MD5 {md5.hexdigest()} does not match the bucket's ETag {m.group(1)}")
            checks.append("etag-md5")
        if sha256 is not None:
            if sha.hexdigest() != sha256:
                raise CheckFailed(f"{url}: SHA-256 {sha.hexdigest()} does not match the pinned {sha256}")
            checks.append("pinned-sha256")
    except CheckFailed:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return {"bytes": size, "sha256": sha.hexdigest(), "check": "+".join(checks)}


def list_keys(prefix: str, opener=urllib.request.urlopen) -> list[tuple[str, int]]:
    """(key, size) of every bucket object under prefix (anonymous ListObjectsV2)."""
    out, token = [], None
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        url = S3 + "?" + urllib.parse.urlencode(query)
        with opener(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s:Contents", ns):
            out.append((item.find("s:Key", ns).text, int(item.find("s:Size", ns).text)))
        if root.findtext("s:IsTruncated", default="false", namespaces=ns) != "true":
            return out
        token = root.findtext("s:NextContinuationToken", namespaces=ns)


# --------------------------------------------------------------------------------------------------- segments
@dataclass
class Segment:
    name: str
    prefix: str                       # "PHerc0139/segments/<id>/"
    surface: str                      # surface-volume zarr, relative to prefix
    labels: str | None                # label version directory, relative to prefix
    published: str | None             # organizers' published prediction (.tif), relative to prefix
    pinned: dict = field(default_factory=dict)       # relative key -> {"bytes", "sha256"}

    def key(self, rel: str) -> str:
        return self.prefix + rel

    def surface_level(self, data_dir) -> Path:
        return Path(data_dir) / self.key(self.surface) / "0"

    def label_array(self, data_dir, array: str) -> Path:
        return Path(data_dir) / self.key(f"{self.labels}/{array}")

    def published_path(self, data_dir) -> Path | None:
        return Path(data_dir) / self.key(self.published) if self.published else None


def segment(manifest: dict, name: str | None = None, prefix: str | None = None, surface: str | None = None,
            labels: str | None = None, published: str | None = None) -> Segment:
    """A preset from manifest.json by short name (w035, w044, ...), or any segment by its bucket prefix."""
    d = manifest["defaults"]
    if name and name in manifest["segments"]:
        s = manifest["segments"][name]
        return Segment(name, f"{s['scroll']}/segments/{s['id']}/", surface or s.get("surface", d["surface"]),
                       labels or s.get("labels", d["labels"]), published or s.get("published"), s.get("pinned", {}))
    if not prefix:
        raise ValueError(f"unknown segment {name!r}: use one of {sorted(manifest['segments'])} or pass --prefix")
    prefix = prefix.rstrip("/") + "/"
    return Segment(name or prefix.rstrip("/").rsplit("/", 1)[-1], prefix, surface or d["surface"],
                   labels or d["labels"], published, {})


# ------------------------------------------------------------------------------------------------------- plans
@dataclass
class Item:
    url: str
    dest: Path
    sha256: str | None = None
    allow_absent: bool = False
    absent_list: Path | None = None   # where to record a 404 (the zarr level directory's absent.json)
    rel: str = ""                     # key relative to the segment, for display


def model_items(manifest: dict, model_dir: Path) -> list[Item]:
    ck, v = manifest["checkpoint"], manifest["villa"]
    items = [Item(f"https://huggingface.co/{ck['repository']}/resolve/{ck['revision']}/{ck['file']}",
                  model_dir / ck["file"], ck["sha256"], rel=ck["file"])]
    raw = v["repository"].replace("https://github.com/", "https://raw.githubusercontent.com/")
    for row in v["files"]:
        items.append(Item(f"{raw}/{v['revision']}/{v['base_path']}/{row['path']}", model_dir / "villa" / row["path"],
                          row["sha256"], rel="villa/" + row["path"]))
    return items


def _item(seg: Segment, data_dir, rel: str, **kw) -> Item:
    return Item(S3 + seg.key(rel), Path(data_dir) / seg.key(rel), seg.pinned.get(rel, {}).get("sha256"), rel=rel, **kw)


def meta_items(seg: Segment, data_dir, want_labels=True, want_published=True) -> list[Item]:
    items = [_item(seg, data_dir, f"{seg.surface}/.zattrs"), _item(seg, data_dir, f"{seg.surface}/0/.zarray")]
    if want_labels and seg.labels:
        items += [_item(seg, data_dir, f"{seg.labels}/{a}/0/zarr.json") for a in ("inklabels.zarr", "supervision.zarr")]
    if want_published and seg.published:
        items.append(_item(seg, data_dir, seg.published))
    return items


def region_items(seg: Segment, data_dir, region, want_labels=True) -> list[Item]:
    """CT chunks and label shards a region reads; needs the metadata from meta_items on disk first."""
    y0, x0, h, w = region
    level = seg.surface_level(data_dir)
    vol = SurfaceVolume(level)
    vol.check_region(y0, x0, h, w)
    items = [_item(seg, data_dir, f"{seg.surface}/0/{vol.key(c)}", allow_absent=True,
                   absent_list=level / "absent.json") for c in vol.chunk_coords(y0, x0, h, w)]
    if want_labels and seg.labels:
        for a in ("inklabels.zarr", "supervision.zarr"):
            root = seg.label_array(data_dir, a)
            meta = L.array_meta(root)
            for key, _, _ in L.shard_keys(meta, y0, x0, h, w):
                items.append(_item(seg, data_dir, f"{seg.labels}/{a}/0/{key}", allow_absent=True,
                                   absent_list=root / "0" / "absent.json"))
    return items


# --------------------------------------------------------------------------------------------------------- run
def run(items: list[Item], data_dir, check_only: bool = False, workers: int = 8, log=print) -> dict:
    """Download what is missing, check what exists. Returns counts; raises CheckFailed listing every failure."""
    data_dir = Path(data_dir)
    receipt_path = data_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    absent_lists = {}
    for it in items:
        if it.absent_list is not None and it.absent_list not in absent_lists:
            p = it.absent_list
            absent_lists[p] = set(json.loads(p.read_text())) if p.exists() else set()

    def rkey(it):
        return it.url

    def one(it: Item):
        name = it.dest.name if it.absent_list is None else it.dest.relative_to(it.absent_list.parent).as_posix()
        if it.dest.exists():
            want = it.sha256 or receipt.get(rkey(it), {}).get("sha256")
            if want is None:
                if check_only:
                    return it, "unverified", None
            else:
                got = sha256_file(it.dest)
                if got == want:
                    return it, "ok", None
                if check_only:
                    return it, "mismatch", f"{it.dest}: SHA-256 {got}, expected {want}"
                it.dest.unlink()
        elif it.absent_list is not None and name in absent_lists[it.absent_list]:
            return it, "absent", None
        if check_only:
            return it, "missing", f"{it.dest} is not on disk"
        try:
            rec = download(it.url, it.dest, it.sha256, allow_absent=it.allow_absent)
        except (CheckFailed, urllib.error.HTTPError) as error:
            return it, "failed", str(error)
        if rec is None:
            return it, "absent", None
        return it, "fetched", rec

    counts, failures, fetched = {}, [], 0
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for n, (it, status, info) in enumerate(pool.map(one, items), 1):
            counts[status] = counts.get(status, 0) + 1
            if status == "fetched":
                receipt[rkey(it)] = {**info, "path": it.dest.relative_to(data_dir).as_posix(),
                                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                fetched += info["bytes"]
            elif status == "absent" and it.absent_list is not None:
                absent_lists[it.absent_list].add(it.dest.relative_to(it.absent_list.parent).as_posix())
            elif status in ("mismatch", "missing", "failed", "unverified") and info:
                failures.append(info)
            if n % 200 == 0:
                log(json.dumps({"files": n, "of": len(items), "downloaded_bytes": fetched}))
    if not check_only:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=0, sort_keys=True))
        for p, keys in absent_lists.items():
            if keys:
                p.write_text(json.dumps(sorted(keys), indent=0))
    result = {"files": len(items), **counts, "downloaded_bytes": fetched}
    if failures or counts.get("unverified"):
        raise CheckFailed(json.dumps({**result, "problems": failures[:20]}, indent=1))
    return result
