"""Model comparison and terminal dedup cannot merge opposite known strands."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from isoquant_lib.assignment.isoform_assignment import ReadAssignment, ReadAssignmentType as T
from isoquant_lib.common import junctions_from_blocks
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.context import ModelStore
from isoquant_lib.model_construction.end_processor import TranscriptEndProcessor
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.quantification.long_read_counter import create_gene_counter, create_transcript_counter
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_tests import test_model_filter_eligibility as eligibility_tests


class ModelStrandIdentityTest(unittest.TestCase):
    def filter(self, strands=("+", "-"), reverse=False):
        fixture = eligibility_tests.ModelFilterEligibilityTest()
        fixture.setUp()
        models = [fixture.model("first"), fixture.model("second")]
        for model, strand in zip(models, strands):
            model.strand = strand
            model.gene_id = "gene_" + model.transcript_id
        if reverse:
            models.reverse()
        model_filter, _ = fixture.filter(models, [0, 0])
        reads = []
        for model in models:
            for i in range(10):
                read = ReadAssignment(model.transcript_id + str(i), T.noninformative, fixture.pools)
                read.chr_id = "chr1"
                read.strand = read.mapped_strand = model.strand
                read.exons = read.corrected_exons = model.exon_blocks
                read.corrected_introns = junctions_from_blocks(read.exons)
                read.mapping_quality = 60
                read.polya_info = PolyAInfo(-1, -1, -1, -1)
                if model.strand == "+":
                    read.additional_info["pt_terminal"] = (1, model.get_end())
                elif model.strand == "-":
                    read.additional_info["pt_terminal"] = (2, model.get_start())
                self.assertTrue(model_filter.store.save_assigned_read(read, model.transcript_id))
                reads.append(read)
        return model_filter, reads

    def count(self, model_filter, reads):
        with tempfile.TemporaryDirectory() as directory:
            transcript = create_transcript_counter(str(Path(directory) / "transcript"), "unique_only")
            gene = create_gene_counter(str(Path(directory) / "gene"), "unique_splicing_consistent")
            counter = ModelReadCounter(model_filter.ctx, transcript, gene)
            counter.build_model_read_assignments(reads)
            transcript.dump()
            counts = {tid: values.get(0) for tid, values in transcript.feature_counter.items()}
        return counts, counter.model_read_assignments

    def test_opposite_strands_keep_both_models_and_all_twenty_reads_in_both_orders(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                model_filter, reads = self.filter(reverse=reverse)
                model_filter.filter_transcripts()
                self.assertEqual({m.transcript_id for m in model_filter.store.transcript_model_storage},
                                 {"first", "second"})
                counts, assignments = self.count(model_filter, reads)
                self.assertEqual(counts, {"first": 10, "second": 10})
                self.assertTrue(all(a.assignment_type.is_unique() for a in assignments))

    def test_same_strand_duplicates_still_merge_and_all_twenty_reads_count(self):
        for strand in ("+", "-"):
            with self.subTest(strand=strand):
                model_filter, reads = self.filter((strand, strand))
                model_filter.filter_transcripts()
                self.assertEqual(len(model_filter.store.transcript_model_storage), 1)
                counts, assignments = self.count(model_filter, reads)
                self.assertEqual(list(counts.values()), [20])
                self.assertTrue(all(a.assignment_type.is_unique() for a in assignments))

    def test_unknown_model_strands_retain_legacy_similarity_comparison(self):
        for strands in ((".", "+"), ("-", "."), (".", ".")):
            with self.subTest(strands=strands):
                model_filter, _ = self.filter(strands)
                self.assertEqual(model_filter.detect_similar_isoforms(model_filter.store.transcript_model_storage),
                                 {"second": "first"})

    def model(self, tid, strand, mono=False, kind=MT.novel_in_catalog, end=600):
        exons = [(100, end)] if mono else [(100, 200), (300, 400), (500, end)]
        model = TranscriptModel("chr1", strand, tid, "gene_" + tid, exons, kind)
        model.intron_path = junctions_from_blocks(exons)
        return model

    def processor(self, references, models):
        fixture = eligibility_tests.ModelFilterEligibilityTest()
        fixture.setUp()
        gi = NS(all_isoforms_exons={m.transcript_id: m.exon_blocks for m in references},
                isoform_strands={m.transcript_id: m.strand for m in references})
        store = ModelStore(gi)
        store.args = fixture.args
        store.transcript_model_storage = list(models)
        ctx = NS(store=store, gene_info=gi, args=fixture.args, use_tss_model=False, novel_apa=False)
        return TranscriptEndProcessor(ctx)

    def test_unemitted_reference_dedup_respects_strand_for_spliced_and_mono_models(self):
        for mono in (False, True):
            for ref_strand, novel_strand, retained in (
                    ("+", "-", True), ("-", "+", True), ("+", "+", False),
                    ("-", "-", False), (".", "+", False), ("-", ".", False)):
                with self.subTest(mono=mono, reference=ref_strand, novel=novel_strand):
                    reference = self.model("unemitted", ref_strand, mono, MT.known)
                    novel = self.model("novel", novel_strand, mono)
                    processor = self.processor([reference], [novel])
                    processor._drop_duplicate_alt_end_models()
                    self.assertEqual(bool(processor.store.transcript_model_storage), retained)

    def test_novel_pair_dedup_respects_strand_and_unknown_legacy_for_empty_chains(self):
        for mono in (False, True):
            for strands, expected in ((("+", "-"), 2), (("+", "+"), 1), ((".", "-"), 1)):
                with self.subTest(mono=mono, strands=strands):
                    models = [self.model("first", strands[0], mono), self.model("second", strands[1], mono)]
                    processor = self.processor([], models)
                    processor._drop_duplicate_alt_end_models()
                    self.assertEqual(len(processor.store.transcript_model_storage), expected)

    def test_alternative_end_creation_ignores_opposite_reference_but_keeps_same_and_unknown_dedup(self):
        for mono in (False, True):
            for reference_strand, created in (("-", True), ("+", False), (".", False)):
                with self.subTest(mono=mono, reference=reference_strand):
                    source = self.model("source", "+", mono, MT.known, end=900)
                    reference = self.model("unemitted", reference_strand, mono, MT.known)
                    nic = self.model("nic", "+", mono)
                    processor = self.processor([source, reference], [source])
                    # Exercise the dedup boundary independently of trained peak calling.
                    processor.derive_alternative_end_models = Mock(return_value=[(nic, [])])
                    processor._add_known_alternative_end_models()
                    self.assertEqual([m.transcript_id for m in processor.store.transcript_model_storage],
                                     ["source", "nic"] if created else ["source"])


if __name__ == "__main__":
    unittest.main()
