"""Reference IDs sharing a graph chain retain independent model eligibility."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import isoquant
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from isoquant_lib.common import junctions_from_blocks
from isoquant_lib.gene_info import GeneInfo
from isoquant_lib.model_construction.assignment_based_constructor import AssignmentBasedConstructor
from isoquant_lib.model_construction.context import ModelConstructionContext, ModelStore
from isoquant_lib.model_construction.end_processor import TranscriptEndProcessor
from isoquant_lib.model_construction.fl_graph_constructor import GraphBasedConstructor
from isoquant_lib.model_construction.model_filter import ModelFilter
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.model_construction.read_assigner import ConstructionReadAssigner
from isoquant_lib.model_construction.transcript_to_gene_joiner import TranscriptToGeneJoiner
from isoquant_lib.quantification.long_read_counter import create_gene_counter, create_transcript_counter
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_tests import test_read_direction as direction_tests


class KnownChainIndexTest(unittest.TestCase):
    def context(self, reverse=False, counts=None):
        fixture = direction_tests.ReadDirectionTest()
        fixture.setUp()
        args = fixture.args
        args.model_construction_strategy = "sensitive_ont"
        args.polya_requirement = "never"
        args.polya_trimmed = "tag:pt"
        args.report_canonical = "all"
        args.no_model_construction = False
        args.genedb = "reference"
        args.fl_data = args.novel_apa = args.debug = False
        isoquant.set_model_construction_options(args)
        short = fixture.model("+", "short", [(100, 200), (300, 400), (500, 600)], gene="G")
        long = fixture.model("+", "long", [(100, 200), (300, 400), (500, 900)], gene="G")
        extension = fixture.model("+", "extension", [(10, 49), (91, 200), (300, 400), (500, 1000)], gene="G")
        models = [extension, long, short] if reverse else [extension, short, long]
        gi = GeneInfo.from_models(models, args.delta)
        gi.gene_regions = {"G": (10, 1000)}
        store = ModelStore(gi)
        store.detected_known_isoforms = set()
        ctx = ModelConstructionContext(gi, SeqRecord(Seq("N" * 2000)), args,
                                       NS(increment=lambda: 999), fixture.pools, None, None, store)
        reads = []
        for model in (short, long, extension):
            for i in range((counts or {}).get(model.transcript_id, 3)):
                read = fixture.read("+", rid=model.transcript_id + str(i), exons=model.exon_blocks)
                read.polya_info = PolyAInfo(model.get_end() + 1, -1, -1, -1)
                read.polyA_found = True
                read.additional_info["pt_terminal"] = (1, model.get_end())
                profile = ctx.profile_constructor.construct_profiles(read.corrected_exons, read.polya_info, [])
                ctx.assigner.set_candidate_read_strand(profile, "+", "+")
                assignment = ctx.assigner.assign_to_isoform(read.read_id, profile)
                read.assignment_type = assignment.assignment_type
                read.gene_assignment_type = assignment.gene_assignment_type
                read.isoform_matches = assignment.isoform_matches
                self.assertTrue(read.assignment_type.is_unique())
                self.assertEqual(read.isoform_matches[0].assigned_transcript, model.transcript_id)
                reads.append(read)
        ctx.build_graph(reads)
        return ctx, reads

    def construct_and_count(self, ctx, reads):
        GraphBasedConstructor(ctx).construct_fl_isoforms()
        # The upstream extension blocks the bare 5' vertices of both siblings.
        self.assertEqual([m.transcript_id for m in ctx.store.transcript_model_storage], ["extension"])
        AssignmentBasedConstructor(ctx).construct_assignment_based_isoforms(reads)
        assigner = ConstructionReadAssigner(ctx)
        model_filter = ModelFilter(ctx, TranscriptEndProcessor(ctx))
        model_filter.pre_filter_transcripts()
        assigner.assign_reads_to_models(reads)
        # Model loading is outside this bounded construction test. No read has
        # an alternative peak within its own known-model support pool.
        with patch("isoquant_lib.model_construction.end_processor.get_polya_model", return_value=None):
            model_filter.filter_transcripts()
        model_filter.correct_transcript_splice_sites()
        assigner.assign_reads_to_models(reads)
        assigner.resolve_read_ownership_conflicts()
        ctx.store.transcript_model_storage = TranscriptToGeneJoiner(
            ctx.store.transcript_model_storage, ctx.gene_info).join_transcripts()
        with tempfile.TemporaryDirectory() as directory:
            gene = create_gene_counter(str(Path(directory) / "gene"), "unique_splicing_consistent")
            transcript = create_transcript_counter(str(Path(directory) / "transcript"), "unique_only")
            counter = ModelReadCounter(ctx, transcript, gene)
            counter.build_model_read_assignments(reads)
            transcript.dump()
            gene.dump()
            transcript_counts = {tid: values.get(0) for tid, values in transcript.feature_counter.items()}
            gene_counts = {gid: values.get(0) for gid, values in gene.feature_counter.items()}
            dumped = (Path(directory) / "transcript_counts.tsv").read_text()
        return transcript_counts, gene_counts, dumped

    def test_same_chain_different_ends_keep_all_ids_and_unique_counts_in_both_orders(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                ctx, reads = self.context(reverse)
                chain = tuple(junctions_from_blocks([(100, 200), (300, 400), (500, 600)]))
                self.assertEqual(ctx.known_isoforms_in_graph[chain], "short" if reverse else "long")
                self.assertEqual(ctx.get_known_spliced_isoforms(ctx.gene_info), ctx.known_isoforms_in_graph)
                self.assertEqual(ctx.known_isoforms_in_graph_ids["short"], chain)
                self.assertEqual(ctx.known_isoforms_in_graph_ids["long"], chain)
                tx, gene, dumped = self.construct_and_count(ctx, reads)
                self.assertEqual(tx, {"short": 3, "long": 3, "extension": 3})
                self.assertEqual(gene, {"G": 9})
                self.assertIn("short\t3.00", dumped)
                self.assertIn("long\t3.00", dumped)

    def test_complete_index_does_not_create_an_unsupported_reference_model(self):
        ctx, reads = self.context(counts={"long": 0})
        self.assertIn("long", ctx.known_isoforms_in_graph_ids)
        tx, gene, _ = self.construct_and_count(ctx, reads)
        self.assertEqual(tx, {"short": 3, "extension": 3})
        self.assertEqual(gene, {"G": 6})

    def test_existing_minimum_known_read_support_still_applies(self):
        ctx, reads = self.context()
        ctx.args.min_known_count = 4
        GraphBasedConstructor(ctx).construct_fl_isoforms()
        AssignmentBasedConstructor(ctx).construct_assignment_based_isoforms(reads)
        self.assertEqual(ctx.store.transcript_model_storage, [])

    def test_unthreadable_or_empty_reference_chains_are_excluded(self):
        ctx, _ = self.context()
        ctx.path_processor = NS(thread_introns=lambda introns: None if len(introns) == 2 else introns)
        self.assertEqual(list(ctx._known_isoform_paths(ctx.gene_info)), [
            ("extension", ((50, 90), (201, 299), (401, 499)))])
        ctx.gene_info.all_isoforms_introns["mono"] = []
        self.assertNotIn("mono", ctx.get_known_spliced_isoforms(ctx.gene_info).values())


if __name__ == "__main__":
    unittest.main()
