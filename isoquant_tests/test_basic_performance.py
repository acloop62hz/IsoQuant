"""Performance changes preserve membership, terminal geometry and tie handling."""
import copy
import random
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from isoquant_lib.gene_info import TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.context import ModelStore
from isoquant_lib.model_construction.end_processor import TranscriptEndProcessor
from isoquant_lib.model_construction.intron_graph import TerminalVertex as V
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_tests.test_intron_path_threading import _processor
from isoquant_tests import test_read_direction as direction_tests


class BasicPerformanceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = direction_tests.ReadDirectionTest()
        self.fixture.setUp()

    def store(self, reads):
        store = ModelStore(NS(isoform_strands={'source': '+', 'target': '+'}))
        store.args = self.fixture.args
        for read in reads:
            store.save_assigned_read(read, 'source')
        return store

    @staticmethod
    def snapshot(store):
        return ({k: [id(r) for r in v] for k, v in store.transcript_read_ids.items()},
                dict(store.internal_counter), dict(store.read_assignment_counts),
                dict(store.construction_assignment))

    def test_batch_matches_scalar_with_duplicates_rejections_and_source_floor(self):
        rng = random.Random(983)
        for case in range(100):
            reads = [self.fixture.read('+', rid=str(i)) for i in range(40)]
            members = reads + reads[:5]
            scalar, batch = self.store(members), self.store(members)
            for read in reads:
                if rng.random() < .2:
                    read.mapped_strand = '-'
            requested = list(members)
            rng.shuffle(requested)
            requested = requested[:rng.randrange(16, 46)]
            target_strand = '-' if case % 9 == 0 else '+'
            floor = case % 3
            moved = 0
            for read in requested:
                if len(scalar.transcript_read_ids['source']) <= floor:
                    break
                moved += scalar.move_read_to_model(read, 'source', 'target', target_strand)
            self.assertEqual(batch.move_reads_to_model(requested, 'source', 'target', target_strand, floor), moved)
            self.assertEqual(self.snapshot(batch), self.snapshot(scalar))

    def test_missing_member_preserves_scalar_partial_progress_and_error(self):
        reads = [self.fixture.read('+', rid=str(i)) for i in range(40)]
        scalar, batch = self.store(reads), self.store(reads)
        missing = self.fixture.read('+', rid='absent')
        requested = reads[3:20] + [missing]
        with self.assertRaises(ValueError):
            for read in requested:
                scalar.move_read_to_model(read, 'source', 'target', '+')
        with self.assertRaises(ValueError):
            batch.move_reads_to_model(requested, 'source', 'target', '+')
        self.assertEqual(self.snapshot(batch), self.snapshot(scalar))

    def test_histogram_reuse_matches_recomputation_including_monoexon_clamps(self):
        for strand in ('+', '-', '.'):
            for mono in (False, True):
                for correction in (False, True):
                    exons = [(100, 900)] if mono else [(100, 200), (300, 400), (500, 900)]
                    read_exons = ([(120, 800)] if mono else
                                  [(120, 200), (300, 400), (500, 800)]) if correction else exons
                    read = self.fixture.read(strand, exons=read_exons)
                    read.polyA_found = True
                    read.polya_info = PolyAInfo(read_exons[-1][1]+1, -1, -1, -1) if strand == '+' else PolyAInfo(-1, read_exons[0][0]-1, -1, -1)
                    model = TranscriptModel('chr1', strand, 'model', 'gene', exons, MT.novel_in_catalog)
                    store = NS(transcript_read_ids={'model': [read]})
                    ctx = NS(store=store, gene_info=None, args=self.fixture.args, use_tss_model=False, novel_apa=False)
                    processor = TranscriptEndProcessor(ctx)
                    processor._terminal_model = lambda *a, **kw: None
                    processor._terminal_histograms = Mock(wraps=processor._terminal_histograms)
                    baseline, optimized = copy.deepcopy(model), copy.deepcopy(model)
                    processor.correct_novel_transcript_ends(baseline, [read])
                    processor._mark_terminal_confirmation(baseline)
                    processor._terminal_histograms.reset_mock()
                    histograms = processor.correct_novel_transcript_ends(optimized, [read])
                    processor._mark_terminal_confirmation(optimized, histograms)
                    self.assertEqual((optimized.exon_blocks, optimized.polya_confirmed, optimized.tss_confirmed),
                                     (baseline.exon_blocks, baseline.polya_confirmed, baseline.tss_confirmed))
                    self.assertEqual(processor._terminal_histograms.call_count, 2 if histograms is None else 1)
                    if mono and correction:
                        self.assertIsNone(histograms)

    def test_terminal_options_reused_without_changing_proximity_ties(self):
        intron = (1000, 2000)
        processor = _processor(intron,
            outgoing=[(V.polya, 2500), (V.polya, 2540), (V.tss_right, 2500), (V.read_end, 2700)],
            incoming=[(V.polyt, 500), (V.polyt, 540), (V.tss_left, 500), (V.read_start, 300)])
        graph = processor.intron_graph
        graph.get_outgoing = Mock(wraps=graph.get_outgoing)
        graph.get_incoming = Mock(wraps=graph.get_incoming)
        expected_right = next(v for v in graph.get_outgoing(intron, V.polya) if abs(v[1]-2520) <= 50)
        expected_left = next(v for v in graph.get_incoming(intron, V.polyt) if abs(v[1]-520) <= 50)
        graph.get_outgoing.reset_mock()
        graph.get_incoming.reset_mock()
        for _ in range(10):
            self.assertEqual(processor.thread_ends(intron, 2520, True), expected_right)
            self.assertEqual(processor.thread_starts(intron, 520, True), expected_left)
        self.assertEqual(graph.get_outgoing.call_count, 4)
        self.assertEqual(graph.get_incoming.call_count, 4)


if __name__ == '__main__':
    unittest.main()
