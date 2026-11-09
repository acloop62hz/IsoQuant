"""Direction regressions across candidate ranking, model support and counting."""
import io
import unittest
from collections import defaultdict
from types import SimpleNamespace as NS
from unittest.mock import patch

import isoquant
from isoquant_lib.alignment.alignment_processor import AlignmentCollector
from isoquant_lib.assignment.isoform_assignment import (
    IsoformMatch, MatchClassification as C, MatchEvent, MatchEventSubtype as E,
    ReadAssignment, ReadAssignmentType as T)
from isoquant_lib.assignment.long_read_assigner import LongReadAssigner
from isoquant_lib.assignment.long_read_profiles import CombinedProfileConstructor
from isoquant_lib.assignment.read_direction import (
    CONFLICT, direction_compatible, resolve_read_direction, trusted_read_direction)
from isoquant_lib.common import junctions_from_blocks
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.assignment_based_constructor import AssignmentBasedConstructor
from isoquant_lib.model_construction.context import ModelStore, StrandnessReportingLevel
from isoquant_lib.model_construction.end_processor import TranscriptEndProcessor
from isoquant_lib.model_construction.fl_graph_constructor import GraphBasedConstructor
from isoquant_lib.model_construction.intron_graph import TerminalVertex, IntronGraph
from isoquant_lib.model_construction.intron_path import IntronPathStorage
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.model_construction.read_assigner import ConstructionReadAssigner
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_lib.utils.serialization import read_dict, write_dict
from isoquant_lib.utils.string_pools import StringPoolManager


class ReadDirectionTest(unittest.TestCase):
    def setUp(self):
        self.args = NS(delta=None, matching_strategy="default", resolve_ambiguous="default",
                       count_exons=True, stranded="forward",
                       check_canonical=False, graph_clustering_distance=20)
        # Same matching setup as the existing exact-chain real-geometry regression.
        isoquant.set_matching_options(self.args)
        self.args.min_known_count = self.args.min_novel_count = 2
        self.args.simple_alignments_mapq_cutoff = 0
        self.args.requires_polya_for_construction = False
        self.args.require_monointronic_polya = False
        self.args.report_canonical_strategy = StrandnessReportingLevel.all
        self.args.fl_only = False
        self.args.report_novel_unspliced = True
        self.args.min_mono_exon_coverage = .9
        self.args.require_monoexonic_polya = False
        self.pools = StringPoolManager()
        self.exons = [(100, 200), (300, 400), (500, 600)]

    def model(self, strand="+", tid="plus", exons=None, kind=MT.known, gene=None):
        return TranscriptModel("chr1", strand, tid, gene or ("gene_" + tid),
                               exons or self.exons, kind)

    def read(self, strand="+", tid=None, gene=None, exons=None, rid="r", kind=None, events=None):
        match = None if tid is None else IsoformMatch(C.full_splice_match, self.pools,
                assigned_gene=gene or ("gene_" + tid), assigned_transcript=tid,
                match_subclassification=events)
        r = ReadAssignment(rid, kind or (T.noninformative if tid is None else T.unique), self.pools, match=match)
        r.chr_id = "chr1"
        r.mapped_strand = strand
        r.strand = strand
        r.exons = r.corrected_exons = exons or self.exons
        r.corrected_introns = junctions_from_blocks(r.exons)
        r.polya_info = PolyAInfo(-1, -1, -1, -1)
        r.mapping_quality = 60
        return r

    def assign(self, models, strand="+", exons=None, external=None):
        gi = GeneInfo.from_models(models, self.args.delta)
        assigner = LongReadAssigner(gi, self.args, self.pools)
        profile = CombinedProfileConstructor(gi, self.args).construct_profiles(
            exons or self.exons, PolyAInfo(-1, -1, -1, -1), [])
        if strand is not None:
            assigner.set_candidate_read_strand(profile, strand, external)
        return assigner.assign_to_isoform("r", profile)

    def counter(self, refs, models, bound=None):
        ctx = NS(gene_info=GeneInfo.from_models(refs, self.args.delta), args=self.args,
                 string_pools=self.pools, store=NS(transcript_model_storage=models,
                                                  construction_assignment=bound or {}))
        tx, gene = [], []
        return ModelReadCounter(ctx, NS(add_read_info=tx.append, add_confirmed_features=lambda _: None),
                    NS(add_read_info=gene.append, add_confirmed_features=lambda _: None))

    def test_protocol_external_conflict_and_unknown_are_distinct(self):
        for mode, mapped, external, expected in [
            ("forward", "+", None, "+"), ("reverse", "+", None, "-"),
            ("unstranded", "+", None, "."), ("unstranded", "+", "-", "-"),
            ("forward", "+", "-", CONFLICT), ("reverse", "+", "+", CONFLICT)]:
            self.args.stranded = mode
            self.assertEqual(resolve_read_direction(self.args, mapped, external), expected)
        self.assertTrue(direction_compatible(".", "+"))
        self.assertFalse(direction_compatible(CONFLICT, "+"))
        self.assertFalse(direction_compatible("+", "."))
        r = self.read("+")
        r.additional_info = {"external_read_strand": "+", "pt_terminal": (2, 100)}
        self.assertEqual(trusted_read_direction(r, self.args), CONFLICT)
        buf = io.BytesIO()
        write_dict(r.additional_info, buf)
        buf.seek(0)
        self.assertEqual(read_dict(buf), r.additional_info)

    def test_spliced_and_mono_candidates_filtered_before_ranking(self):
        for exons in (self.exons, [(100, 600)]):
            models = [self.model("-", "minus", exons), self.model("+", "plus", exons)]
            self.assertEqual({m.assigned_transcript for m in self.assign(models, exons=exons).isoform_matches}, {"plus"})
            self.assertEqual({m.assigned_transcript for m in self.assign(models, "-", exons).isoform_matches}, {"minus"})
            self.assertEqual(self.assign(models, strand=None, exons=exons).assignment_type, T.ambiguous)
            twin = self.model("+", "twin", exons, gene="gene_plus")
            result = self.assign(models + [twin], exons=exons)
            self.assertEqual(result.assignment_type, T.ambiguous)
            self.assertEqual({m.assigned_transcript for m in result.isoform_matches}, {"plus", "twin"})

    def test_no_opposite_gene_fallback_in_exonic_and_intronic_paths(self):
        for exons in ([(130, 180)], [(220, 280)]):
            r = self.assign([self.model("-", "minus")], exons=exons)
            self.assertTrue(all(not m.assigned_gene and not m.assigned_transcript for m in r.isoform_matches))
        r = self.assign([self.model()], external="-")
        self.assertFalse(r.isoform_matches)
        self.assertEqual(r.additional_attributes["direction_conflict"], "true")

    def test_collector_uses_trusted_direction_without_flipping_models(self):
        collector = object.__new__(AlignmentCollector)
        collector.params = self.args
        r = self.read("+", "minus")
        r.isoform_matches[0].transcript_strand = "-"
        self.assertEqual(collector.get_assignment_strand(r), "+")
        r.additional_info["external_read_strand"] = "-"
        self.assertEqual(collector.get_assignment_strand(r), ".")
        self.assertEqual(r.additional_attributes["direction_conflict"], "true")
        self.assertEqual(r.isoform_matches[0].transcript_strand, "-")

    def test_final_construction_and_kept_known_shortcuts_reroute_to_sense(self):
        plus, minus = self.model(), self.model("-", "minus")
        for binding in ({"r": "minus"}, {}):
            r = self.read("+", "minus")
            counter = self.counter([plus, minus], [plus, minus], binding)
            counter.build_model_read_assignments([r])
            result = counter.model_read_assignments[0]
            self.assertEqual({m.assigned_transcript for m in result.isoform_matches}, {"plus"})
            self.assertEqual(result.gene_assignment_type, T.unique)
            self.assertEqual(r.isoform_matches[0].assigned_transcript, "minus")

    def test_dropped_known_cannot_keep_opposite_gene_even_when_gene_survives(self):
        minus = self.model("-", "minus")
        sibling = self.model("-", "sibling", gene=minus.gene_id)
        for models in ([sibling], []):
            r = self.read("+", "minus")
            counter = self.counter([minus, sibling], models)
            counter.build_model_read_assignments([r])
            self.assertTrue(all(m.assigned_gene is None for m in counter.model_read_assignments[0].isoform_matches))
        # Unknown-direction reference membership retains the legacy behavior.
        counter = self.counter([minus], [minus])
        counter.build_model_read_assignments([self.read(".", "minus")])
        self.assertEqual(counter.model_read_assignments[0].isoform_matches[0].assigned_transcript, "minus")

    def test_conflict_cannot_bypass_final_fastpaths(self):
        plus = self.model()
        r = self.read("+", "plus")
        r.additional_info["pt_terminal"] = (2, 100)
        counter = self.counter([plus], [plus], {"r": "plus"})
        counter.build_model_read_assignments([r])
        result = counter.model_read_assignments[0]
        self.assertFalse(result.isoform_matches)
        self.assertEqual(result.additional_attributes["direction_conflict"], "true")

    def fl_constructor(self, reads, strand="-", known=False):
        gi = GeneInfo.from_models([self.model(strand, "known")], self.args.delta)
        store = ModelStore(gi)
        store.args = self.args
        # Keep the class-level upstream dedup state isolated from other tests.
        store.detected_known_isoforms = set()
        path = ((TerminalVertex.read_start, 100), (201, 299), (401, 499), (TerminalVertex.read_end, 600))
        assigned = (ReadAssignment("path", T.unique, self.pools, match=IsoformMatch(
                    C.full_splice_match, self.pools, assigned_transcript="known", assigned_gene="gene_known"))
                    if known else self.read())
        ctx = NS(store=store, args=self.args, gene_info=gi, use_tss_model=False,
                 create_alt_ts_nics=False, use_technical_replicas=False, file_name_group_idx=-1,
                 assigner=NS(assign_to_isoform=lambda *a: assigned),
                 profile_constructor=NS(construct_profiles=lambda *a: None),
                 strand_detector=NS(get_strand=lambda *a: strand, get_clean_strand=lambda *a: strand),
                 path_storage=NS(fl_paths={path}, paths={path: len(reads)}, paths_to_reads={path: reads}),
                 get_transcript_id=lambda: 999, known_isoforms_in_graph={}, known_introns=set(),
                 select_reference_gene=lambda *a: None)
        return GraphBasedConstructor(ctx)

    def test_fl_motif_conflict_no_flip_and_rechecked_support_threshold(self):
        for strands, expected in [("++", 0), ("+-", 0), ("++--", 1), ("--", 1), ("..", 1)]:
            with self.subTest(strands=strands):
                reads = [self.read(s, rid=str(i)) for i, s in enumerate(strands)]
                c = self.fl_constructor(reads)
                c.construct_fl_isoforms()
                self.assertEqual(len(c.store.transcript_model_storage), expected)
                if expected:
                    m = c.store.transcript_model_storage[0]
                    self.assertEqual(m.strand, "-")
                    self.assertEqual(c.store.internal_counter[m.transcript_id], 2)
                    self.assertTrue(all(a.mapped_strand in ("-", ".") for a in c.store.transcript_read_ids[m.transcript_id]))

    def test_fl_known_support_also_filtered_before_detection(self):
        for strands, expected in [("++", 0), ("+-", 0), ("--", 1)]:
            c = self.fl_constructor([self.read(s, rid=str(i)) for i, s in enumerate(strands)], known=True)
            c.construct_fl_isoforms()
            self.assertEqual(len(c.store.transcript_model_storage), expected)
            self.assertEqual("known" in c.store.detected_known_isoforms, bool(expected))

    def assignment_constructor(self, models):
        gi = GeneInfo.from_models(models, self.args.delta)
        store = ModelStore(gi)
        store.args = self.args
        store.detected_known_isoforms = set()
        ctx = NS(store=store, args=self.args, gene_info=gi,
                 known_isoforms_in_graph_ids={m.transcript_id: True for m in models},
                 get_transcript_id=lambda: 999,
                 intron_graph=NS(outgoing_edges={}, incoming_edges={}))
        return AssignmentBasedConstructor(ctx)

    def test_assignment_based_mono_coverage_and_spliced_ends_cannot_combine_wrong_strands(self):
        mono = self.model(exons=[(100, 600)])
        c = self.assignment_constructor([mono])
        events = [MatchEvent(E.mono_exon_match)]
        c.construct_assignment_based_isoforms([
            self.read("+", "plus", exons=[(100, 349)], events=events),
            self.read("-", "plus", exons=[(350, 600)], rid="wrong", events=events)])
        self.assertFalse(c.store.transcript_model_storage)
        # Spliced non-FL: right-end support only comes from an opposite read.
        for strands, expected in [("+-", 0), ("++", 1)]:
            c = self.assignment_constructor([self.model()])
            c.construct_assignment_based_isoforms([
                self.read(strands[0], "plus", exons=[(100, 200), (300, 400), (500, 540)]),
                self.read(strands[1], "plus", exons=[(160, 200), (300, 400), (500, 600)], rid="right")])
            self.assertEqual(len(c.store.transcript_model_storage), expected)

    def test_novel_mono_tail_side_filters_before_clustering(self):
        for strand, expected in [("-", 0), ("+", 1)]:
            c = self.assignment_constructor([self.model()])
            c.collect_terminal_exons_from_graph = lambda: ([], [])
            reads = [self.read(strand, exons=[(100, 600)], rid=str(i)) for i in range(2)]
            for r in reads:
                r.polyA_found = True
                r.polya_info = PolyAInfo(600, -1, -1, -1)
            c.construct_monoexon_novel(reads)
            self.assertEqual(len(c.store.transcript_model_storage), expected)

    def test_pending_nic_move_is_atomic_and_same_strand_move_works(self):
        plus = self.model()
        store = ModelStore(GeneInfo.from_models([plus], self.args.delta))
        store.args = self.args
        store.transcript_model_storage = [plus]
        r = self.read()
        self.assertTrue(store.save_assigned_read(r, "plus"))
        self.assertFalse(store.move_read_to_model(r, "plus", "pending_wrong", target_strand="-"))
        self.assertEqual(store.transcript_read_ids["plus"], [r])
        self.assertEqual(store.internal_counter["plus"], 1)
        self.assertEqual(store.read_assignment_counts["r"], 1)
        self.assertEqual(store.construction_assignment["r"], "plus")
        self.assertTrue(store.move_read_to_model(r, "plus", "pending_plus", target_strand="+"))
        self.assertEqual(store.construction_assignment["r"], "pending_plus")
        self.assertEqual(store.read_assignment_counts["r"], 1)
        self.assertEqual(store.transcript_read_ids["pending_plus"], [r])

    def test_construction_reassignment_shortcut_and_profile_both_use_direction(self):
        plus, minus = self.model(), self.model("-", "minus")
        c = self.assignment_constructor([plus, minus])
        c.store.transcript_model_storage = [plus, minus]
        c.ctx.string_pools = self.pools
        r = self.read("+", "minus")
        ConstructionReadAssigner(c.ctx).assign_reads_to_models([r])
        self.assertEqual(c.store.transcript_read_ids["plus"], [r])
        self.assertFalse(c.store.transcript_read_ids["minus"])

    def test_alt_end_histograms_exclude_reverse_support(self):
        c = object.__new__(TranscriptEndProcessor)
        c.args = self.args
        right, wrong = self.read(), self.read("-", rid="wrong")
        for r in (right, wrong):
            r.polyA_found = True
            r.polya_info = PolyAInfo(600, -1, -1, -1)
        starts, ends = c._terminal_histograms(self.model(), [right, wrong])
        self.assertEqual(starts[100], [right])
        self.assertEqual(ends[600], [right])

    def test_graph_and_path_terminal_evidence_prefers_protocol_over_old_strand(self):
        r = self.read("+")
        r.strand = "-"  # Old cache/motif-derived value must not control evidence.
        r.polya_info = PolyAInfo(600, 100, -1, -1)
        graph = object.__new__(IntronGraph)
        graph.params = self.args
        graph.read_assignments = [r]
        graph.intron_collector = NS(discarded_introns=set(), substitute=lambda x: x)
        graph.is_start_internal = graph.is_end_internal = lambda *a: False
        polya, bare_end, polyt, bare_start = graph.collect_terminal_positions()
        self.assertEqual(polya[(401, 499)][600], 1)
        self.assertFalse(polyt)
        processor = NS(intron_graph=graph, thread_introns=lambda x: list(x),
            thread_ends=lambda intron, pos, trusted: (TerminalVertex.polya if trusted else TerminalVertex.read_end, pos),
            thread_starts=lambda intron, pos, trusted: (TerminalVertex.polyt if trusted else TerminalVertex.read_start, pos))
        storage = IntronPathStorage(self.args, processor)
        storage.fill([r])
        self.assertEqual(next(iter(storage.paths))[0], (TerminalVertex.read_start, 100))
        self.assertEqual(next(iter(storage.paths))[-1], (TerminalVertex.polya, 600))
        r.additional_info["pt_terminal"] = (2, 100)
        self.assertEqual(sum(sum(x.values()) for h in graph.collect_terminal_positions() for x in h.values()), 0)
        storage = IntronPathStorage(self.args, processor)
        storage.fill([r])
        self.assertFalse(storage.paths)


if __name__ == "__main__":
    unittest.main()
