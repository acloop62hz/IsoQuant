"""Terminal transcript ambiguity must not erase independently supported gene counts."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import isoquant
from isoquant_lib.assignment.assignment_io import ReadInfoPrinter
from isoquant_lib.assignment.isoform_assignment import (
    IsoformMatch, MatchClassification as C, MatchEvent, MatchEventSubtype as E,
    ReadAssignment, ReadAssignmentType as T)
from isoquant_lib.assignment.long_read_assigner import LongReadAssigner
from isoquant_lib.assignment.long_read_profiles import CombinedProfileConstructor
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.quantification.long_read_counter import (
    ReadWeightCounter, create_gene_counter, create_transcript_counter)
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_lib.utils.string_pools import StringPoolManager


class TerminalGeneAmbiguityTest(unittest.TestCase):
    def setUp(self):
        self.args = NS(delta=None, matching_strategy="default", resolve_ambiguous="default",
                       count_exons=True, stranded="forward", check_canonical=False,
                       graph_clustering_distance=20)
        isoquant.set_matching_options(self.args)
        self.pools = StringPoolManager()
        self.exons = [(100, 200), (300, 400), (500, 600)]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def setup_assigner(self, strand="+", genes=("G", "G")):
        models = [TranscriptModel("chr1", strand, "T"+str(i), gene, self.exons, MT.known)
                  for i, gene in enumerate(genes)]
        gi = GeneInfo.from_models(models, self.args.delta)
        return models, gi, LongReadAssigner(gi, self.args, self.pools)

    def match(self, strand="+", direction=None, genes=("G", "G"), partial=False):
        models, gi, assigner = self.setup_assigner(strand, genes)
        exons = ([(100, 200), (300, 400), (500, 900)] if strand == "+" else
                 [(20, 200), (300, 400), (500, 600)])
        if partial:
            exons = [exons[-1] if strand == "+" else exons[0]]
        if partial and strand == "+":
            exons = [(500, 680)]
        polya = PolyAInfo(exons[-1][1]+1, -1, -1, -1) if strand == "+" else PolyAInfo(-1, exons[0][0]-1, -1, -1)
        profile = CombinedProfileConstructor(gi, self.args).construct_profiles(exons, polya, [])
        assigner.set_candidate_read_strand(profile, strand if direction is None else direction)
        read = assigner.assign_to_isoform("read", profile)
        read.exons = read.corrected_exons = exons
        read.chr_id = "chr1"
        read.strand = read.mapped_strand = strand
        read.polya_info = polya
        read.gene_info = gi
        return read, models, gi, assigner

    def counters(self, strategy="unique_splicing_consistent", suffix=""):
        p = Path(self.temp.name)
        return (create_gene_counter(str(p/('gene'+suffix)), strategy),
                create_transcript_counter(str(p/('tx'+suffix)), "unique_only"))

    @staticmethod
    def signature(r):
        return [(m.assigned_gene, m.assigned_transcript, m.match_classification,
                 [(e.event_type, e.isoform_region, e.read_region, e.event_info)
                  for e in m.match_subclassifications]) for m in r.isoform_matches]

    def test_real_matching_preserves_both_strands_transcript_ambiguity_and_events(self):
        for strand in ("+", "-"):
            for partial in (False, True):
                with self.subTest(strand=strand, partial=partial):
                    before, _, _, _ = self.match(strand, direction=".", partial=partial)
                    after, _, _, _ = self.match(strand, partial=partial)
                    self.assertEqual(before.assignment_type, T.inconsistent_ambiguous)
                    self.assertEqual(before.gene_assignment_type, T.inconsistent)
                    self.assertEqual(after.assignment_type, before.assignment_type)
                    self.assertEqual(after.gene_assignment_type, T.inconsistent_non_intronic)
                    self.assertEqual(self.signature(after), self.signature(before))
                    self.assertEqual(after.additional_attributes["gene_assignment_reason"], "same_gene_terminal_ambiguity")
                    gene, tx = self.counters(suffix=strand+str(partial))
                    gene.add_read_info(after)
                    tx.add_read_info(after)
                    self.assertEqual(gene.feature_counter["G"].get(0), 1)
                    self.assertFalse(tx.all_features)
                    self.assertNotIn("G", gene.confirmed_features)

    def test_count_once_per_read_and_existing_confirmation_gate(self):
        read, _, _, _ = self.match()
        for confirmed in (False, True):
            gene, tx = self.counters(suffix=str(confirmed))
            gene.add_read_info(read)
            self.assertEqual(gene.feature_counter["G"].get(0), 1)
            if confirmed:
                gene.add_confirmed_features(["G"])
            gene.dump()
            self.assertIn("G\t"+('1.00' if confirmed else '0.00'), Path(gene.output_counts_file_name).read_text())
        # A stricter user-selected strategy still excludes non-intronic evidence.
        gene, tx = self.counters(strategy="unique_only", suffix="strict")
        gene.add_read_info(read)
        self.assertFalse(gene.all_features)

    def test_final_leftover_gene_count_and_models_unchanged(self):
        source, models, gi, _ = self.match()
        snapshot = [(m.transcript_id, m.gene_id, m.strand, list(m.exon_blocks), m.transcript_type) for m in models]
        gene, tx = self.counters()
        ctx = NS(store=NS(transcript_model_storage=models, construction_assignment={}),
                 gene_info=gi, args=self.args, string_pools=self.pools)
        counter = ModelReadCounter(ctx, tx, gene)
        counter.build_model_read_assignments([source])
        result = counter.model_read_assignments[0]
        self.assertEqual(result.assignment_type, T.inconsistent_ambiguous)
        self.assertEqual(result.gene_assignment_type, T.inconsistent_non_intronic)
        self.assertEqual(gene.feature_counter["G"].get(0), 1)
        self.assertFalse(tx.all_features)
        gene.dump()
        self.assertIn("G\t1.00", Path(gene.output_counts_file_name).read_text())
        self.assertEqual(snapshot, [(m.transcript_id, m.gene_id, m.strand, list(m.exon_blocks), m.transcript_type) for m in models])

    def test_gene_only_fallback_composes_without_inventing_or_confirming_model(self):
        source, _, gi, _ = self.match(partial=True)
        gene, tx = self.counters()
        ctx = NS(store=NS(transcript_model_storage=[], construction_assignment={}),
                 gene_info=gi, args=self.args, string_pools=self.pools)
        counter = ModelReadCounter(ctx, tx, gene)
        counter.build_model_read_assignments([source])
        result = counter.model_read_assignments[0]
        self.assertEqual(result.assignment_type, T.noninformative)
        self.assertEqual(result.gene_assignment_type, T.inconsistent_non_intronic)
        self.assertEqual(result.isoform_matches[0].assigned_gene, "G")
        self.assertIsNone(result.isoform_matches[0].assigned_transcript)
        self.assertEqual(result.additional_attributes["gene_assignment_source"], "reference_gene_only")
        self.assertEqual(result.additional_attributes["gene_assignment_reason"], "same_gene_terminal_ambiguity")
        self.assertEqual(gene.feature_counter["G"].get(0), 1)
        self.assertNotIn("G", gene.confirmed_features)
        self.assertFalse(tx.all_features)
        self.assertFalse(ctx.store.transcript_model_storage)

    def test_serialized_and_read_table_gene_subtype_and_provenance(self):
        read, _, gi, _ = self.match()
        buf = io.BytesIO()
        read.serialize(buf)
        buf.seek(0)
        loaded = ReadAssignment.deserialize(buf, gi, self.pools)
        self.assertEqual(loaded.gene_assignment_type, T.inconsistent_non_intronic)
        self.assertEqual(loaded.assignment_type, T.inconsistent_ambiguous)
        self.assertEqual(self.signature(loaded), self.signature(read))
        self.assertEqual(loaded.additional_attributes, read.additional_attributes)
        printer = object.__new__(ReadInfoPrinter)
        printer.output_file = io.StringIO()
        printer.assignment_checker = NS(check=lambda _: True)
        printer.params = NS(check_canonical=False, cage=None)
        printer.add_read_info(loaded)
        rows = [line.split("\t") for line in printer.output_file.getvalue().strip().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual({fields[5] for fields in rows}, {"T0", "T1"})
        for fields in rows:
            self.assertEqual(fields[3:5], ["G", "inconsistent_non_intronic"])
            self.assertEqual(fields[6], "inconsistent_ambiguous")
            self.assertIn("gene_assignment_reason=same_gene_terminal_ambiguity;", fields[-1])

    def test_multigene_and_unknown_direction_still_excluded(self):
        for genes, direction in [(("G", "G"), "."), (("G", "H"), "+")]:
            read, _, _, _ = self.match(genes=genes, direction=direction)
            self.assertNotEqual(read.gene_assignment_type, T.inconsistent_non_intronic)
            self.assertEqual(ReadWeightCounter("unique_splicing_consistent").process_inconsistent(
                read.gene_assignment_type, len(set(genes))), 0)
            self.assertNotIn("gene_assignment_reason", read.additional_attributes)
        _, gi, assigner = self.setup_assigner()
        profile = CombinedProfileConstructor(gi, self.args).construct_profiles(self.exons, PolyAInfo(-1,-1,-1,-1), [])
        assigner.set_candidate_read_strand(profile, "+", "-")
        self.assertFalse(assigner.assign_to_isoform("conflict", profile).isoform_matches)

    def synthetic(self, event_lists, strand="+", genes=("G", "G")):
        _, gi, assigner = self.setup_assigner(strand, genes)
        events = {"T"+str(i): [MatchEvent(e) for e in es] for i, es in enumerate(event_lists)}
        matches = [IsoformMatch(C.novel_in_catalog, self.pools, assigned_gene=g,
                    assigned_transcript=t, match_subclassification=events[t], transcript_strand=strand)
                   for t, g in zip(events, genes)]
        kind = assigner.classify_assignment(list(events), events)
        read = ReadAssignment("synthetic", kind, self.pools, matches)
        return read, assigner

    def test_other_terminal_events_allowed_without_changing_transcript_policy(self):
        for event in (E.alternative_polya_site_left, E.alternative_polya_site_right,
                      E.alternative_tss_left, E.alternative_tss_right,
                      E.major_exon_elongation_left, E.major_exon_elongation_right):
            read, assigner = self.synthetic([[event, E.exon_elongation_left], [event]])
            assigner._preserve_terminal_gene_assignment(read, NS(candidate_read_strand="+"))
            self.assertEqual(read.gene_assignment_type, T.inconsistent_non_intronic)
            self.assertEqual(read.assignment_type, T.inconsistent_ambiguous)

    def test_splicing_internal_polya_artifacts_and_unknown_events_not_promoted(self):
        for event in (E.alt_left_site_novel, E.unspliced_intron_retention,
                      E.internal_polya_left, E.internal_polya_right, E.intron_shift,
                      E.exon_misalignment, E.fake_micro_intron_retention, E.undefined):
            read, assigner = self.synthetic([[E.alternative_polya_site_right], [E.alternative_polya_site_right, event]])
            assigner._preserve_terminal_gene_assignment(read, NS(candidate_read_strand="+"))
            self.assertEqual(read.gene_assignment_type, T.inconsistent)
            self.assertNotIn("gene_assignment_reason", read.additional_attributes)

    def test_gene_and_all_candidate_strands_and_identifiers_must_be_valid(self):
        for mutation in ("gene_strand", "match_strand", "model_strand", "unknown_tid", "wrong_gene_map", "missing_events"):
            read, assigner = self.synthetic([[E.alternative_polya_site_right]]*2)
            m = read.isoform_matches[1]
            if mutation == "gene_strand": assigner.gene_info.gene_strands["G"] = "-"
            elif mutation == "match_strand": m.transcript_strand = "-"
            elif mutation == "model_strand": assigner.gene_info.isoform_strands["T1"] = "-"
            elif mutation == "unknown_tid": m.assigned_transcript = "absent"
            elif mutation == "wrong_gene_map": assigner.gene_info.gene_id_map["T1"] = "other"
            elif mutation == "missing_events": m.match_subclassifications = []
            assigner._preserve_terminal_gene_assignment(read, NS(candidate_read_strand="+"))
            self.assertEqual(read.gene_assignment_type, T.inconsistent, mutation)


if __name__ == "__main__":
    unittest.main()
