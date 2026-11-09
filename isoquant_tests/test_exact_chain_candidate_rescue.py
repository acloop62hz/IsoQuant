"""Candidate rescue must preserve fine verification and strand/chain limits."""
import unittest
from types import SimpleNamespace as NS

import isoquant
from isoquant_lib.assignment.isoform_assignment import MatchClassification, ReadAssignmentType
from isoquant_lib.assignment.long_read_assigner import LongReadAssigner
from isoquant_lib.assignment.long_read_profiles import CombinedProfileConstructor
from isoquant_lib.common import junctions_from_blocks
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_lib.utils.string_pools import StringPoolManager


class ExactChainCandidateRescueTest(unittest.TestCase):
    def synthetic(self, strand="+"):
        models = {
            "exact": [(100, 200), (300, 400), (500, 600)],
            "duplicate": [(100, 200), (300, 400), (500, 600)],
            "opposite": [(100, 200), (300, 400), (500, 600)],
            "partial": [(300, 400), (500, 600)],
            "shifted_junction": [(100, 201), (300, 400), (500, 600)],
            "distant_start": [(49, 200), (300, 400), (500, 600)],
            "distant_end": [(100, 200), (300, 400), (500, 651)],
        }
        assigner = object.__new__(LongReadAssigner)
        assigner.params = NS(apa_delta=50, stranded="forward")
        assigner.gene_info = NS(
            all_isoforms_introns={k: junctions_from_blocks(v) for k, v in models.items()},
            isoform_strands={k: ("-" if strand == "+" else "+") if k == "opposite" else strand
                             for k in models},
            transcript_region=lambda k: (models[k][0][0], models[k][-1][1]))
        profile = NS(read_split_exon_profile=NS(read_features=models["exact"]))
        assigner.set_candidate_read_strand(profile, strand)
        return assigner, profile

    def test_full_chain_both_strands_and_existing_candidates_preserved(self):
        for strand in ("+", "-"):
            with self.subTest(strand=strand):
                assigner, profile = self.synthetic(strand)
                # Exclusion applies only to restored candidates, not the original shortlist.
                original = ["opposite", "existing"]
                self.assertEqual(assigner._restore_exact_chain_candidates(profile, original),
                                 original + ["duplicate", "exact"])
                self.assertEqual(original, ["opposite", "existing"])
                index = assigner._exact_chain_candidate_index
                assigner._restore_exact_chain_candidates(profile, [])
                self.assertIs(assigner._exact_chain_candidate_index, index)

    def test_end_window_inclusive_and_exact_observed_junctions(self):
        assigner, profile = self.synthetic()
        for shift, expected in [(50, True), (51, False), (-50, True), (-51, False)]:
            with self.subTest(shift=shift):
                profile.read_split_exon_profile.read_features = [(100, 200), (300, 400), (500, 600 + shift)]
                self.assertEqual("exact" in assigner._restore_exact_chain_candidates(profile, []), expected)
        # A one-base junction mismatch must not qualify through profile snapping.
        profile.read_split_exon_profile.read_features = [(100, 201), (300, 400), (500, 600)]
        self.assertNotIn("exact", assigner._restore_exact_chain_candidates(profile, []))

    def test_unknown_direction_monoexonic_and_model_profiles_unchanged(self):
        assigner, profile = self.synthetic()
        del profile.candidate_read_strand
        self.assertEqual(assigner._restore_exact_chain_candidates(profile, ["old"]), ["old"])
        assigner.set_candidate_read_strand(profile, ".")
        self.assertIsNone(assigner._restore_exact_chain_candidates(profile, None))
        assigner.set_candidate_read_strand(profile, "+")
        profile.read_split_exon_profile.read_features = [(100, 600)]
        self.assertEqual(assigner._restore_exact_chain_candidates(profile, []), [])

    def test_direction_does_not_require_tail_evidence(self):
        assigner, profile = self.synthetic()
        for stranded, mapped, external, expected in [
                ("forward", "+", None, "+"), ("forward", "-", None, "-"),
                ("reverse", "+", None, "-"), ("reverse", "-", None, "+"),
                ("unstranded", "+", None, "."), ("unstranded", "+", "-", "-")]:
            with self.subTest(stranded=stranded, mapped=mapped, external=external):
                assigner.params.stranded = stranded
                assigner.set_candidate_read_strand(profile, mapped, external)
                self.assertEqual(profile.candidate_read_strand, expected)

    def test_multiple_compatible_models_are_all_admitted(self):
        assigner, profile = self.synthetic()
        self.assertEqual(assigner._restore_exact_chain_candidates(profile, None), ["duplicate", "exact"])
        self.assertEqual(assigner._restore_exact_chain_candidates(profile, ["exact"]), ["exact", "duplicate"])

    def test_rpl31_real_geometry_and_unchanged_model_set(self):
        shared = [(101002702, 101002808), (101004158, 101004283)]
        models = {
            "ENST00000409028": [(101002283, 101002315)] + shared + [(101005959, 101006071), (101018998, 101019728)],
            "ENST00000409320": [(101002293, 101002315)] + shared + [(101005959, 101006418)],
            "ENST00000419276": [(101002311, 101002315), shared[0], (101004158, 101004361), (101005959, 101006423)],
            "ENST00000409733": [(101002293, 101002808), shared[1], (101005959, 101006071), (101006350, 101006419)],
            "ENST00000264258": [(101002289, 101002315)] + shared + [(101005959, 101006071), (101006350, 101007267)],
            "ENST00000456292": [(101002691, 101002808), shared[1], (101005959, 101006071)],
            "ENST00000441435": [(101002739, 101002808), shared[1], (101005959, 101006071), (101023625, 101024032)],
            "ENST00000409711": [(101002691, 101002808), (101004158, 101006367)],
        }
        params = NS(delta=None, matching_strategy="default", resolve_ambiguous="default",
                    count_exons=True, stranded="forward")
        isoquant.set_matching_options(params)
        storage = [TranscriptModel("chr2", "+", tid, "RPL31", exons, TranscriptModelType.known)
                   for tid, exons in models.items()]
        gi = GeneInfo.from_models(storage, params.delta)
        assigner = LongReadAssigner(gi, params, StringPoolManager())
        constructor = CombinedProfileConstructor(gi, params)
        exons = [(101002698, 101002808), shared[1], (101005959, 101006115)]
        profile = constructor.construct_profiles(exons, PolyAInfo(101006116, -1, -1, -1), [])
        self.assertNotIn("ENST00000456292", assigner.select_similar_isoforms(profile))
        baseline = assigner.assign_to_isoform("test", profile)
        self.assertEqual(baseline.assignment_type, ReadAssignmentType.inconsistent_ambiguous)
        assigner.set_candidate_read_strand(profile, "+")
        restored = assigner.assign_to_isoform("test", profile)
        self.assertEqual(restored.assignment_type, ReadAssignmentType.unique)
        self.assertEqual(restored.isoform_matches[0].assigned_transcript, "ENST00000456292")
        self.assertEqual(restored.isoform_matches[0].match_classification, MatchClassification.full_splice_match)
        self.assertEqual(set(gi.all_isoforms_exons), set(models))
        # Tail evidence is not a prerequisite for candidate admission.
        no_tail = constructor.construct_profiles(exons, PolyAInfo(-1, -1, -1, -1), [])
        assigner.set_candidate_read_strand(no_tail, "+")
        self.assertIn("ENST00000456292", assigner._restore_exact_chain_candidates(no_tail, []))

        # Exercise the final leftover-read path, including direction propagation.
        baseline.exons = baseline.corrected_exons = exons
        baseline.polya_info = profile.polya_info
        baseline.polyA_found = True
        baseline.mapped_strand = baseline.strand = "+"
        params.check_canonical = False
        ctx = NS(store=NS(transcript_model_storage=storage, construction_assignment={}),
                 gene_info=gi, args=params, string_pools=assigner.string_pools)
        transcript_reads, gene_reads = [], []
        counter = ModelReadCounter(ctx,
            NS(add_read_info=transcript_reads.append, add_confirmed_features=lambda _: None),
            NS(add_read_info=gene_reads.append, add_confirmed_features=lambda _: None))
        counter.build_model_read_assignments([baseline])
        self.assertEqual(transcript_reads[0].assignment_type, ReadAssignmentType.unique)
        self.assertEqual(gene_reads[0].gene_assignment_type, ReadAssignmentType.unique)
        self.assertEqual(transcript_reads[0].isoform_matches[0].assigned_transcript, "ENST00000456292")

        # Admission does not bypass the existing polyA verifier.
        distant_tail = constructor.construct_profiles(exons, PolyAInfo(101006800, -1, -1, -1), [])
        assigner.set_candidate_read_strand(distant_tail, "+")
        self.assertTrue(assigner.assign_to_isoform("test", distant_tail).assignment_type.is_inconsistent())

        # Two equally compatible same-gene models remain ambiguous, not forced unique.
        twin = TranscriptModel("chr2", "+", "twin", "RPL31", models["ENST00000456292"], TranscriptModelType.known)
        twin_gi = GeneInfo.from_models(storage + [twin], params.delta)
        twin_assigner = LongReadAssigner(twin_gi, params, StringPoolManager())
        twin_profile = CombinedProfileConstructor(twin_gi, params).construct_profiles(exons, profile.polya_info, [])
        twin_assigner.set_candidate_read_strand(twin_profile, "+")
        self.assertEqual(twin_assigner.assign_to_isoform("test", twin_profile).assignment_type,
                         ReadAssignmentType.ambiguous)


if __name__ == "__main__":
    unittest.main()
