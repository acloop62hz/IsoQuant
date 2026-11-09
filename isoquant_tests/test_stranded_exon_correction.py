"""Fuzzy correction must use trusted direction before BAM aux fields are copied.

Matching and correction are real; only the local error measurement is controlled.
"""
import unittest
from types import SimpleNamespace as NS

import isoquant
from isoquant_lib.assignment.exon_corrector import ExonCorrector
from isoquant_lib.assignment.isoform_assignment import ReadAssignmentType
from isoquant_lib.assignment.long_read_assigner import LongReadAssigner
from isoquant_lib.assignment.long_read_profiles import CombinedProfileConstructor
from isoquant_lib.assignment.read_direction import CONFLICT
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_lib.utils.string_pools import StringPoolManager


class StrandedExonCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.args = NS(delta=None, matching_strategy="default", resolve_ambiguous="default",
                       count_exons=True, stranded="forward", data_type="nanopore",
                       model_construction_strategy="sensitive_ont",
                       graph_clustering_distance=None, report_novel_unspliced=False,
                       no_model_construction=False, polya_requirement="never",
                       polya_trimmed="tag:pt", report_canonical="auto",
                       splice_correction_strategy="default_ont")
        isoquant.set_matching_options(self.args)
        isoquant.set_model_construction_options(self.args)
        isoquant.set_splice_correction_options(self.args)
        self.pools = StringPoolManager()
        self.raw = [(100, 202), (303, 400)]
        self.sense = [(100, 204), (305, 400)]

    def make_case(self, direction="+", both=True, closer=False, error=(1, 2), sense=None):
        strand = direction if direction in ("+", "-") else "+"
        sense = self.sense if sense is None else sense
        opposite = [(100, 201), (302, 400)] if closer else [(100, 200), (301, 400)]
        models = [TranscriptModel("chr1", strand, "sense", "sense_gene", sense,
                                  TranscriptModelType.known)]
        if both:
            models.append(TranscriptModel("chr1", "-" if strand == "+" else "+",
                                          "opposite", "opposite_gene", opposite,
                                          TranscriptModelType.known))
        gi = GeneInfo.from_models(models, self.args.delta)
        profile = CombinedProfileConstructor(gi, self.args).construct_profiles(
            self.raw, PolyAInfo(-1, -1, -1, -1), [])
        assigner = LongReadAssigner(gi, self.args, self.pools)
        assigner.set_candidate_read_strand(profile, strand)
        assignment = assigner.assign_to_isoform("read", profile)
        self.assertEqual([m.assigned_transcript for m in assignment.isoform_matches], ["sense"])
        # Match the collector ordering: BAM/protocol direction is on the profile,
        # while the ReadAssignment still has its default mapped_strand.
        self.assertEqual(assignment.mapped_strand, ".")
        assignment.strand = "-" if strand == "+" else "+"  # Never trust this fallback.
        profile.candidate_read_strand = direction
        errors = []

        def get_error_count(*args, **kwargs):
            errors.append((args, kwargs))
            return error

        alignment = NS(read_exons=self.raw, read_start=100, read_end=400,
                       combined_profile=profile, get_error_count=get_error_count)
        return gi, alignment, assignment, opposite, errors

    def test_both_strands_ties_and_closer_opposite_introns(self):
        # The previous sixteen-case reproduction, including absence of any
        # sequence-error evidence and a same-strand-only annotation control.
        for direction in ("+", "-"):
            for closer in (False, True):
                for error in ((0, 0), (1, 2)):
                    for both in (False, True):
                        with self.subTest(direction=direction, closer=closer, error=error, both=both):
                            gi, alignment, assignment, _, _ = self.make_case(
                                direction, both, closer, error)
                            result = ExonCorrector(gi, self.args, None).correct_assigned_read(
                                alignment, assignment)
                            self.assertEqual(result, self.raw if error == (0, 0) else self.sense)
                            self.assertEqual(assignment.assignment_type, ReadAssignmentType.unique)
                            self.assertEqual(assignment.isoform_matches[0].assigned_gene, "sense_gene")

    def test_unknown_or_absent_profile_direction_keeps_legacy_pool(self):
        for absent in (False, True):
            with self.subTest(absent=absent):
                gi, alignment, assignment, opposite, _ = self.make_case(".")
                assignment.strand = "+"  # Annotation/motif direction cannot narrow the pool.
                if absent:
                    del alignment.combined_profile.candidate_read_strand
                corrector = ExonCorrector(gi, self.args, None)
                self.assertEqual(corrector.correct_assigned_read(alignment, assignment), opposite)
                self.assertFalse(corrector._stranded_intron_profiles)

    def test_explicit_conflict_keeps_original_exons_without_error_scan(self):
        gi, alignment, assignment, _, errors = self.make_case(CONFLICT)
        corrector = ExonCorrector(gi, self.args, None)
        self.assertEqual(corrector.correct_assigned_read(alignment, assignment), self.raw)
        self.assertFalse(errors)
        self.assertFalse(corrector._stranded_intron_profiles)

    def test_per_direction_pool_is_built_once_and_reused(self):
        gi, alignment, assignment, _, _ = self.make_case()
        minus_profile = CombinedProfileConstructor(gi, self.args).construct_profiles(
            self.raw, PolyAInfo(-1, -1, -1, -1), [])
        assigner = LongReadAssigner(gi, self.args, self.pools)
        assigner.set_candidate_read_strand(minus_profile, "-")
        minus_assignment = assigner.assign_to_isoform("minus_read", minus_profile)
        self.assertEqual([m.assigned_transcript for m in minus_assignment.isoform_matches], ["opposite"])
        minus_alignment = NS(**{**vars(alignment), "combined_profile": minus_profile})

        class CountedIntrons(dict):
            scans = 0

            def items(self):
                self.scans += 1
                return super().items()

        gi.all_isoforms_introns = CountedIntrons(gi.all_isoforms_introns)
        corrector = ExonCorrector(gi, self.args, None)
        for _ in range(3):
            self.assertEqual(corrector.correct_assigned_read(alignment, assignment), self.sense)
        self.assertEqual(gi.all_isoforms_introns.scans, 1)
        plus = corrector._stranded_intron_profiles["+"]
        # A reused corrector sees reads from both directions in a gene block.
        for _ in range(2):
            self.assertEqual(corrector.correct_assigned_read(minus_alignment, minus_assignment),
                             [(100, 200), (301, 400)])
        self.assertEqual(gi.all_isoforms_introns.scans, 2)
        self.assertIs(corrector._stranded_intron_profiles["+"], plus)
        alignment.combined_profile.candidate_read_strand = "."
        corrector.correct_assigned_read(alignment, assignment)
        self.assertEqual(gi.all_isoforms_introns.scans, 2)

    def test_original_error_and_distance_thresholds_remain(self):
        for error, correct in [((0, 1), False), ((0, 2), True), ((1, 0), True)]:
            with self.subTest(error=error):
                gi, alignment, assignment, _, _ = self.make_case(error=error)
                self.assertEqual(ExonCorrector(gi, self.args, None).correct_assigned_read(
                    alignment, assignment), self.sense if correct else self.raw)
        for delta, correct in ((self.args.delta, True), (self.args.delta + 1, False)):
            with self.subTest(delta=delta):
                shifted = [(100, 202 + delta), (303 + delta, 400)]
                gi, alignment, assignment, _, _ = self.make_case(sense=shifted)
                self.assertEqual(ExonCorrector(gi, self.args, None).correct_assigned_read(
                    alignment, assignment), shifted if correct else self.raw)

    def test_empty_same_direction_pool_cannot_use_opposite_introns(self):
        gi, alignment, assignment, _, _ = self.make_case("-", both=False)
        alignment.combined_profile.candidate_read_strand = "+"
        corrector = ExonCorrector(gi, self.args, None)
        self.assertEqual(corrector.correct_assigned_read(alignment, assignment), self.raw)
        self.assertEqual(corrector._stranded_intron_profiles["+"].known_features, [])


if __name__ == "__main__":
    unittest.main()
