"""The download step against a local HTTP server standing in for the bucket: byte count, ETag MD5 and pinned
SHA-256 checks, absent objects, existing files and check-only mode. No internet access."""
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecheck import sources  # noqa: E402

FILES = {"/a/good.bin": b"good bytes" * 100, "/a/bad-etag.bin": b"payload", "/a/second.bin": b"second"}
LISTING = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>
<Contents><Key>a/good.bin</Key><Size>1000</Size></Contents><Contents><Key>a/second.bin</Key><Size>6</Size></Contents>
<Contents><Key>X/segments/s/ink-labels/v1/d2/inklabels.zarr/0/c/0/0</Key><Size>9</Size></Contents>
</ListBucketResult>"""


class Handler(BaseHTTPRequestHandler):
    hits = []

    def do_GET(self):
        Handler.hits.append(self.path)
        if self.path.startswith("/?list-type=2"):
            body, etag = LISTING, None
        elif self.path in FILES:
            body = FILES[self.path]
            etag = '"' + ("0" * 32 if "bad-etag" in self.path else hashlib.md5(body).hexdigest()) + '"'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}/"
        cls.saved = sources.S3
        sources.S3 = cls.base

    @classmethod
    def tearDownClass(cls):
        sources.S3 = cls.saved
        cls.server.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        Handler.hits.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_good_download_passes_all_checks(self):
        body = FILES["/a/good.bin"]
        rec = sources.download(self.base + "a/good.bin", self.dir / "g", hashlib.sha256(body).hexdigest())
        self.assertEqual(rec["check"], "size+etag-md5+pinned-sha256")
        self.assertEqual((self.dir / "g").read_bytes(), body)

    def test_pinned_mismatch_keeps_nothing(self):
        with self.assertRaises(sources.CheckFailed):
            sources.download(self.base + "a/good.bin", self.dir / "g", "0" * 64)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_etag_mismatch_keeps_nothing(self):
        with self.assertRaises(sources.CheckFailed):
            sources.download(self.base + "a/bad-etag.bin", self.dir / "b")
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_absent(self):
        self.assertIsNone(sources.download(self.base + "a/none.bin", self.dir / "n", allow_absent=True))
        with self.assertRaises(Exception):
            sources.download(self.base + "a/none.bin", self.dir / "n", retries=0)

    def test_run_records_reuses_and_checks(self):
        level = self.dir / "lvl"
        items = [sources.Item(self.base + "a/good.bin", level / "0" / "1"),
                 sources.Item(self.base + "a/missing.bin", level / "0" / "2", allow_absent=True,
                              absent_list=level / "absent.json")]
        res = sources.run(items, self.dir, workers=2, log=lambda *_: None)
        self.assertEqual((res.get("fetched"), res.get("absent")), (1, 1))
        self.assertEqual(json.loads((level / "absent.json").read_text()), ["0/2"])
        receipt = json.loads((self.dir / "receipt.json").read_text())
        self.assertEqual(receipt[self.base + "a/good.bin"]["sha256"], hashlib.sha256(FILES["/a/good.bin"]).hexdigest())
        Handler.hits.clear()
        res = sources.run(items, self.dir, workers=2, log=lambda *_: None)
        self.assertEqual((res.get("ok"), res.get("absent"), Handler.hits), (1, 1, []))
        res = sources.run(items, self.dir, check_only=True, log=lambda *_: None)
        self.assertEqual(res.get("ok"), 1)
        (level / "0" / "1").write_bytes(b"tampered")
        with self.assertRaises(sources.CheckFailed):
            sources.run(items, self.dir, check_only=True, log=lambda *_: None)
        res = sources.run(items, self.dir, log=lambda *_: None)          # re-fetches the tampered file
        self.assertEqual(res.get("fetched"), 1)
        self.assertEqual((level / "0" / "1").read_bytes(), FILES["/a/good.bin"])

    def test_check_only_never_downloads(self):
        items = [sources.Item(self.base + "a/second.bin", self.dir / "s")]
        with self.assertRaises(sources.CheckFailed):
            sources.run(items, self.dir, check_only=True, log=lambda *_: None)
        self.assertEqual(Handler.hits, [])

    def test_list_keys(self):
        self.assertEqual(sources.list_keys("a/")[:2], [("a/good.bin", 1000), ("a/second.bin", 6)])

    def test_cli_list(self):
        import io
        from contextlib import redirect_stdout
        from tracecheck import cli
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(cli.main(["fetch", "--prefix", "X/segments/s", "--list"]), 0)
        lines = [json.loads(x) for x in buf.getvalue().splitlines()]
        self.assertIn("v1/d2", lines[1]["entries"])

    def test_presets(self):
        manifest = sources.load_manifest()
        seg = sources.segment(manifest, "w035")
        self.assertTrue(seg.prefix.startswith("PHerc0139/segments/") and "w035" in seg.prefix)
        self.assertTrue(seg.published.endswith(".tif"))
        other = sources.segment(manifest, prefix="X/segments/abc")
        self.assertEqual((other.prefix, other.name), ("X/segments/abc/", "abc"))
        with self.assertRaises(ValueError):
            sources.segment(manifest, "nope")


if __name__ == "__main__":
    unittest.main()
