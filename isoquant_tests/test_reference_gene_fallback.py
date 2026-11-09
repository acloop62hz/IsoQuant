"""Gene evidence survives absent models without changing transcript assignments."""
import tempfile
import unittest
import io
from pathlib import Path
from types import SimpleNamespace as NS

from isoquant_lib.assignment.isoform_assignment import (
    IsoformMatch, MatchClassification, ReadAssignment, ReadAssignmentType as T)
from isoquant_lib.assignment.assignment_io import ReadInfoPrinter
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.quantification.long_read_counter import create_gene_counter, create_transcript_counter
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_lib.utils.string_pools import StringPoolManager


class ReferenceGeneFallbackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pools = StringPoolManager()
        self.ref = [TranscriptModel("chr1", "+", "refA", "geneA",
                    [(100, 200), (300, 400)], TranscriptModelType.known),
                    TranscriptModel("chr1", "-", "refB", "geneB",
                    [(500, 600)], TranscriptModelType.known)]
        self.gi = GeneInfo.from_models(self.ref, 6)
        self.args = NS(delta=6, graph_clustering_distance=20,
                       check_canonical=False, stranded="forward")

    def source(self, kind=T.unique, gene="geneA", tid="refA", strand="+", exons=None):
        r = ReadAssignment("read", kind, self.pools, match=[IsoformMatch(
            MatchClassification.mono_exon_match, self.pools,
            assigned_gene=gene, assigned_transcript=tid)])
        r.chr_id = "chr1"
        r.strand = r.mapped_strand = strand
        r.exons = r.corrected_exons = exons or [(130, 180)]
        r.polya_info = PolyAInfo(-1, -1, -1, -1)
        r.additional_attributes = {"original": "unchanged"}
        return r

    def counter(self, models=()):
        path = Path(self.temp.name)
        gene = create_gene_counter(str(path / "gene"), "unique_splicing_consistent")
        transcript = create_transcript_counter(str(path / "tx"), "unique_only")
        ctx = NS(store=NS(transcript_model_storage=list(models), construction_assignment={}),
                 gene_info=self.gi, args=self.args, string_pools=self.pools)
        return ModelReadCounter(ctx, transcript, gene), gene, transcript

    def test_dropped_known_gene_counts_without_any_surviving_model(self):
        counter, gene, tx = self.counter()
        source = self.source()
        before = list(counter.store.transcript_model_storage)
        counter.build_model_read_assignments([source])
        result = counter.model_read_assignments[0]
        self.assertEqual(result.assignment_type, T.discarded)
        self.assertEqual(result.gene_assignment_type, T.unique)
        self.assertEqual(result.isoform_matches[0].assigned_gene, "geneA")
        self.assertIsNone(result.isoform_matches[0].assigned_transcript)
        self.assertEqual(gene.feature_counter["geneA"].get(0), 1)
        self.assertIn("geneA", gene.confirmed_features)
        self.assertEqual(dict(tx.feature_counter), {})
        self.assertEqual(counter.store.transcript_model_storage, before)
        self.assertEqual(result.additional_attributes["gene_assignment_source"], "reference_gene_only")
        self.assertEqual(source.additional_attributes, {"original": "unchanged"})
        self.assertEqual(source.isoform_matches[0].assigned_transcript, "refA")
        gene.dump()
        self.assertIn("geneA\t1.00", Path(gene.output_counts_file_name).read_text())
        printer = object.__new__(ReadInfoPrinter)
        printer.output_file = io.StringIO()
        printer.assignment_checker = NS(check=lambda _: True)
        printer.params = NS(check_canonical=False, cage=None)
        printer.add_read_info(result)
        fields = printer.output_file.getvalue().strip().split("\t")
        self.assertEqual(fields[3], "geneA")
        self.assertEqual(fields[5], ".")
        self.assertIn("gene_assignment_source=reference_gene_only;", fields[-1])

    def test_nonintronic_gene_only_preserves_counting_and_confirmation_policy(self):
        counter, gene, tx = self.counter()
        source = self.source(T.inconsistent_non_intronic)
        counter.build_model_read_assignments([source])
        result = counter.model_read_assignments[0]
        self.assertEqual(result.assignment_type, T.noninformative)
        self.assertEqual(result.gene_assignment_type, T.inconsistent_non_intronic)
        self.assertEqual(gene.feature_counter["geneA"].get(0), 1)
        self.assertNotIn("geneA", gene.confirmed_features)
        self.assertEqual(dict(tx.feature_counter), {})
        gene.dump()
        self.assertIn("geneA\t0.00", Path(gene.output_counts_file_name).read_text())

    def test_other_surviving_genes_and_same_gene_isoform_ambiguity(self):
        counter, gene, tx = self.counter([self.ref[1]])
        source = self.source()
        before = [m.transcript_id for m in counter.store.transcript_model_storage]
        counter.build_model_read_assignments([source])
        self.assertEqual(gene.feature_counter["geneA"].get(0), 1)
        self.assertFalse(tx.feature_counter)
        self.assertEqual([m.transcript_id for m in counter.store.transcript_model_storage], before)
        counter, gene, tx = self.counter()
        source = self.source(T.ambiguous)
        source.isoform_matches.append(IsoformMatch(
            MatchClassification.genic, self.pools, assigned_gene="geneA", assigned_transcript="refA2"))
        counter.build_model_read_assignments([source])
        self.assertEqual(gene.feature_counter["geneA"].get(0), 1)
        self.assertEqual(len(counter.model_read_assignments[0].isoform_matches), 1)
        self.assertFalse(tx.feature_counter)

    def test_rejects_opposite_unknown_conflicting_direction_and_nonexonic_overlap(self):
        for kwargs, additional in [
                ({"strand": "-"}, {}), ({"strand": "."}, {}),
                ({"exons": [(220, 280)]}, {}), ({"exons": [(700, 750)]}, {}),
                ({}, {"pt_terminal": (2, 130)})]:
            with self.subTest(kwargs=kwargs, additional=additional):
                counter, gene, tx = self.counter()
                source = self.source(**kwargs)
                source.additional_info.update(additional)
                counter.build_model_read_assignments([source])
                self.assertFalse(counter.model_read_assignments[0].isoform_matches)
                self.assertEqual(dict(gene.feature_counter), {})

    def test_rejects_multigene_and_major_splice_sources(self):
        for kind in (T.inconsistent, T.inconsistent_ambiguous, T.noninformative):
            with self.subTest(kind=kind):
                counter, gene, tx = self.counter()
                source = self.source(kind)
                counter.build_model_read_assignments([source])
                self.assertFalse(counter.model_read_assignments[0].isoform_matches)
        counter, gene, tx = self.counter()
        source = self.source(T.ambiguous)
        source.isoform_matches.append(IsoformMatch(
            MatchClassification.genic, self.pools, assigned_gene="geneB"))
        # Even an incorrectly promoted source label must not bypass multi-gene checks.
        source.gene_assignment_type = T.unique
        counter.build_model_read_assignments([source])
        self.assertFalse(counter.model_read_assignments[0].isoform_matches)

    def test_preserves_existing_final_gene_and_transcript_assignments(self):
        counter, gene, tx = self.counter()
        source = self.source()
        for final_gene, final_tid in [("other_gene", None), ("other_gene", "other_tx"), (None, "other_tx")]:
            with self.subTest(gene=final_gene, tid=final_tid):
                result = ReadAssignment("read", T.unique, self.pools, match=[IsoformMatch(
                    MatchClassification.genic, self.pools,
                    assigned_gene=final_gene, assigned_transcript=final_tid)])
                old = result.isoform_matches[0]
                counter._reference_gene_only_fallback(source, result)
                self.assertIs(result.isoform_matches[0], old)
                self.assertNotIn("gene_assignment_source", result.additional_attributes)

    def test_reverse_protocol_and_external_direction_without_tail_requirement(self):
        counter, gene, tx = self.counter()
        self.args.stranded = "reverse"
        source = self.source(strand="-")
        counter.build_model_read_assignments([source])
        self.assertEqual(gene.feature_counter["geneA"].get(0), 1)
        counter, gene, tx = self.counter()
        self.args.stranded = "unstranded"
        source = self.source()
        source.additional_info["pt_terminal"] = (1, 180)
        counter.build_model_read_assignments([source])
        self.assertEqual(gene.feature_counter["geneA"].get(0), 1)
        self.assertFalse(source.polyA_found)


if __name__ == "__main__":
    unittest.main()
