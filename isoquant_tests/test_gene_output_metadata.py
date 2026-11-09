"""Final GTF gene metadata must include models emitted in later gene blocks."""
import re
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import gffutils

from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.transcript_printer import GFFPrinter


class GeneOutputMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.printer = GFFPrinter(self.temp.name, "sample", NS(get_id=lambda *a: "exon1"),
                                  header="# retained header\n")
        self.addCleanup(self.printer.close)
        self.path = Path(self.printer.model_fname)

    @staticmethod
    def model(tid, start=100, end=200, gene="G", strand="+"):
        return TranscriptModel("chr1", strand, tid, gene, [(start, end)], MT.novel_not_in_catalog)

    def dump(self, models, gene_regions=None):
        valid = [m for m in models if m.exon_blocks[0][0] > 0 and
                 m.exon_blocks[0][0] <= m.exon_blocks[0][1]]
        gi = GeneInfo.from_models(valid or [self.model("annotation")], 6)
        if gene_regions is not None:
            gi.gene_regions = gene_regions
        self.printer.dump(gi, models)

    def gene(self, identifier="G"):
        rows = [line.split("\t") for line in self.path.read_text().splitlines()]
        rows = [row for row in rows if len(row) == 9 and row[2] == "gene" and
                'gene_id "%s";' % identifier in row[8]]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        return int(row[3]), int(row[4]), int(re.search(r'transcripts "(\d+)";', row[8]).group(1))

    def test_interleaved_blocks_extend_both_ends_and_preserve_feature_bytes(self):
        self.dump([self.model("first"), self.model("other", 800, 900, "H")])
        self.dump([self.model("later", 80, 350)])
        before = self.path.read_text()
        self.printer.close()
        self.assertEqual(self.gene(), (80, 350, 2))
        self.assertEqual(self.gene("H"), (800, 900, 1))
        non_gene = lambda text: [line for line in text.splitlines(keepends=True)
                                 if len(line.split("\t")) != 9 or line.split("\t")[2] != "gene"]
        self.assertEqual(non_gene(before), non_gene(self.path.read_text()))
        # Validate the finalized annotation through the real downstream DB parser.
        db = gffutils.create_db(str(self.path), dbfn=":memory:", force=True,
                                disable_infer_genes=True, disable_infer_transcripts=True,
                                merge_strategy="create_unique")
        self.addCleanup(db.conn.close)
        self.assertEqual((db["G"].start, db["G"].end), (80, 350))
        self.assertEqual(len(list(db.children("G", featuretype="transcript"))), 2)

    def test_reference_gene_bounds_are_preserved_and_extended(self):
        self.dump([self.model("first")], {"G": (90, 700)})
        self.dump([self.model("second", 80, 600)], {"G": (90, 700)})
        self.printer.close()
        self.assertEqual(self.gene(), (80, 700, 2))

    def test_invalid_models_do_not_extend_gene_or_increase_count(self):
        self.dump([self.model("first")])
        with self.assertLogs("IsoQuant", level="WARNING"):
            self.dump([self.model("invalid", 400, 350)])
        self.printer.close()
        self.assertEqual(self.gene(), (100, 200, 1))
        self.assertNotIn('transcript_id "invalid";', self.path.read_text())

    def test_count_is_distinct_across_blocks(self):
        model = self.model("same")
        self.dump([model])
        self.dump([model, self.model("new", 100, 300)])
        self.printer.close()
        self.assertEqual(self.gene(), (100, 300, 2))

    def test_single_block_needs_no_rewrite_and_close_is_idempotent(self):
        self.dump([self.model("a"), self.model("b", 80, 300)])
        before = self.path.read_bytes()
        with patch.object(self.printer, "_rewrite_gene_metadata", side_effect=AssertionError("unneeded rewrite")):
            self.printer.close()
            self.printer.close()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.gene(), (80, 300, 2))

    def test_atomic_failure_preserves_output_and_can_be_retried(self):
        self.dump([self.model("a")])
        self.dump([self.model("b", 80, 300)])
        before = self.path.read_bytes()
        with patch("isoquant_lib.model_construction.transcript_printer.os.replace",
                   side_effect=OSError("replacement failed")):
            with self.assertRaisesRegex(OSError, "replacement failed"):
                self.printer.close()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.printer._metadata_finalized)
        self.assertFalse(list(Path(self.temp.name).glob(".isoquant_gene_metadata_*")))
        self.printer.close()
        self.assertEqual(self.gene(), (80, 300, 2))

    def test_rewrite_preserves_file_mode(self):
        self.dump([self.model("a")])
        self.dump([self.model("b", 80, 300)])
        self.path.chmod(0o640)
        self.printer.close()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o640)

    def test_direct_merge_handle_does_not_trigger_gene_rewrite(self):
        # Dataset merging writes through out_gff directly, with no dump calls.
        self.printer.out_gff.write("# merged body\n")
        with patch.object(self.printer, "_rewrite_gene_metadata", side_effect=AssertionError("merge rewrite")):
            self.printer.close()
        self.assertIn("# merged body\n", self.path.read_text())


if __name__ == "__main__":
    unittest.main()
