"""Coarse candidates preserve splice evidence without an exact-chain fallback."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import isoquant
from isoquant_lib.assignment.isoform_assignment import MatchClassification, ReadAssignment, ReadAssignmentType as T
from isoquant_lib.assignment.long_read_assigner import LongReadAssigner
from isoquant_lib.assignment.long_read_profiles import CombinedProfileConstructor
from isoquant_lib.common import difference_in_present_features
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.quantification.long_read_counter import create_gene_counter, create_transcript_counter
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_lib.utils.string_pools import StringPoolManager


class CandidatePrefilterTest(unittest.TestCase):
    def setUp(self):
        self.params = NS(delta=None, matching_strategy="default", resolve_ambiguous="default",
                         count_exons=True, stranded="forward", check_canonical=False)
        isoquant.set_matching_options(self.params)
        self.pools = StringPoolManager()

    @staticmethod
    def mirror(exons):
        return [(202100000 - end, 202100000 - start) for start, end in reversed(exons)]

    def context(self, strand="+", twin_gene=None):
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
        storage = [TranscriptModel("chr2", strand, tid, "RPL31",
                                  self.mirror(exons) if strand == "-" else exons, MT.known)
                   for tid, exons in models.items()]
        if twin_gene is not None:
            target = next(m for m in storage if m.transcript_id == "ENST00000456292")
            storage.append(TranscriptModel("chr2", strand, "twin", twin_gene, target.exon_blocks, MT.known))
        gi = GeneInfo.from_models(storage, self.params.delta)
        assigner = LongReadAssigner(gi, self.params, self.pools)
        return storage, gi, assigner, CombinedProfileConstructor(gi, self.params)

    def profile(self, constructor, strand="+", shift=7, tail=True):
        exons = [(101002691 + shift, 101002808), (101004158, 101004283), (101005959, 101006115)]
        if strand == "-":
            exons = self.mirror(exons)
        polya = (PolyAInfo(exons[-1][1] + 1, -1, -1, -1) if strand == "+" else
                 PolyAInfo(-1, exons[0][0] - 1, -1, -1)) if tail else PolyAInfo(-1, -1, -1, -1)
        return constructor.construct_profiles(exons, polya, [])

    def test_absent_and_outside_states_are_not_splice_conflicts(self):
        for model_state in (-2, -1, 0, 1):
            for read_state in (-2, -1, 0, 1):
                with self.subTest(model=model_state, read=read_state):
                    result = LongReadAssigner.match_profile([read_state], {"m": [model_state]})
                    expected = int((model_state == 1 and read_state < 0) or
                                   (read_state == 1 and model_state < 0))
                    self.assertEqual(result[0].diff, expected)

    def test_generic_profile_distance_is_unchanged(self):
        self.assertEqual(difference_in_present_features([-1], [-2]), 1)
        self.assertEqual(LongReadAssigner.match_profile([-1], {"m": [-2]})[0].diff, 0)
        self.assertEqual(LongReadAssigner.match_profile_exact_in_range([-1], {"m": [-2]}, (0, 1)), [])

    def test_range_hint_sorting_and_difference_limit_are_preserved(self):
        read = [1, -1, 1, -2]
        models = {"a": [-1, -2, -1, -1], "b": [1, -2, -1, -1], "c": [1, -1, 1, 1]}
        result = LongReadAssigner.match_profile(read, models)
        self.assertEqual([(m.id, m.diff) for m in result], [("b", 1), ("c", 1), ("a", 2)])
        self.assertEqual([(m.id, m.diff) for m in LongReadAssigner.match_profile(read, models, hint=["a", "b"], diff_limit=1)], [("b", 1)])
        self.assertEqual([(m.id, m.diff) for m in LongReadAssigner.match_profile(read, models, hint=["a"], profile_range=(1, 2), diff_limit=0)], [("a", 0)])
        with self.assertRaises(AssertionError):
            LongReadAssigner.match_profile([1], {"bad": []})

    def test_original_and_five_prime_truncated_rpl31_enter_normal_candidates_on_both_strands(self):
        for strand in ("+", "-"):
            storage, gi, assigner, constructor = self.context(strand)
            before = [(m.transcript_id, list(m.exon_blocks)) for m in storage]
            for shift in (7, 50, 51, 89, 107):
                with self.subTest(strand=strand, shift=shift):
                    profile = self.profile(constructor, strand, shift)
                    assigner.set_candidate_read_strand(profile, strand)
                    self.assertIn("ENST00000456292", assigner.select_similar_isoforms(profile))
                    result = assigner.assign_to_isoform("read", profile)
                    self.assertEqual(result.assignment_type, T.unique)
                    self.assertEqual(result.isoform_matches[0].assigned_transcript, "ENST00000456292")
                    self.assertEqual(result.isoform_matches[0].match_classification, MatchClassification.full_splice_match)
            self.assertEqual([(m.transcript_id, m.exon_blocks) for m in storage], before)

    def test_added_downstream_reference_structures_cannot_remove_matching_target(self):
        target_exons = [(100, 200), (300, 400), (500, 600)]
        for count in (0, 2, 4, 6):
            models = [TranscriptModel("chr1", "+", "target", "G", target_exons, MT.known)]
            models.extend(TranscriptModel("chr1", "+", "other" + str(i), "G",
                target_exons + [(800 + i * 100, 850 + i * 100)], MT.known) for i in range(count))
            gi = GeneInfo.from_models(models, self.params.delta)
            assigner = LongReadAssigner(gi, self.params, self.pools)
            profile = CombinedProfileConstructor(gi, self.params).construct_profiles(
                [(151, 200), (300, 400), (500, 644)], PolyAInfo(645, -1, -1, -1), [])
            assigner.set_candidate_read_strand(profile, "+")
            self.assertIn("target", assigner.select_similar_isoforms(profile))
            result = assigner.assign_to_isoform("read", profile)
            self.assertEqual(result.assignment_type, T.unique)
            self.assertEqual(result.isoform_matches[0].assigned_transcript, "target")

    def test_equal_same_gene_and_multigene_models_remain_ambiguous(self):
        for gene in ("RPL31", "other_gene"):
            _, _, assigner, constructor = self.context(twin_gene=gene)
            profile = self.profile(constructor, shift=51)
            assigner.set_candidate_read_strand(profile, "+")
            result = assigner.assign_to_isoform("read", profile)
            self.assertEqual(result.assignment_type, T.ambiguous)
            self.assertEqual(result.gene_assignment_type, T.unique if gene == "RPL31" else T.ambiguous)
            self.assertEqual({m.assigned_transcript for m in result.isoform_matches}, {"twin", "ENST00000456292"})

    def test_candidate_admission_still_uses_direction_and_fine_polya_verification(self):
        _, _, assigner, constructor = self.context()
        profile = self.profile(constructor, shift=51)
        assigner.set_candidate_read_strand(profile, "-")
        self.assertFalse(assigner.select_similar_isoforms(profile))
        assigner.set_candidate_read_strand(profile, "+", "-")
        self.assertEqual(assigner.assign_to_isoform("read", profile).assignment_type, T.noninformative)
        profile = self.profile(constructor, shift=51)
        profile.polya_info.external_polya_pos = 101006800
        assigner.set_candidate_read_strand(profile, "+")
        self.assertTrue(assigner.assign_to_isoform("read", profile).assignment_type.is_inconsistent())
        no_tail = self.profile(constructor, shift=51, tail=False)
        assigner.set_candidate_read_strand(no_tail, "+")
        self.assertIn("ENST00000456292", assigner.select_similar_isoforms(no_tail))

    def test_model_comparison_profiles_use_corrected_distance_without_real_read_hint(self):
        _, _, assigner, constructor = self.context()
        profile = self.profile(constructor, shift=51)
        self.assertIn("ENST00000456292", assigner.select_similar_isoforms(profile))
        self.assertEqual(assigner.assign_to_isoform("model", profile).assignment_type, T.unique)

    def test_final_leftover_read_reaches_real_gene_and_transcript_counts(self):
        storage, gi, _, constructor = self.context()
        profile = self.profile(constructor, shift=51)
        source = ReadAssignment("leftover", T.noninformative, self.pools)
        source.chr_id = "chr2"
        source.strand = source.mapped_strand = "+"
        source.exons = source.corrected_exons = profile.read_split_exon_profile.read_features
        source.polya_info = profile.polya_info
        source.polyA_found = True
        ctx = NS(store=NS(transcript_model_storage=storage, construction_assignment={}),
                 gene_info=gi, args=self.params, string_pools=self.pools)
        with tempfile.TemporaryDirectory() as directory:
            tx = create_transcript_counter(str(Path(directory) / "tx"), "unique_only")
            gene = create_gene_counter(str(Path(directory) / "gene"), "unique_splicing_consistent")
            counter = ModelReadCounter(ctx, tx, gene)
            counter.build_model_read_assignments([source])
            self.assertEqual(counter.model_read_assignments[0].assignment_type, T.unique)
            tx.dump()
            gene.dump()
            self.assertEqual(tx.feature_counter["ENST00000456292"].get(0), 1)
            self.assertEqual(gene.feature_counter["RPL31"].get(0), 1)
            self.assertIn("ENST00000456292\t1.00", Path(tx.output_counts_file_name).read_text())


if __name__ == "__main__":
    unittest.main()
