"""Reference assignment reasons must not describe a different final assignment."""
import io
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from isoquant_lib.assignment.assignment_io import ReadInfoPrinter
from isoquant_lib.assignment.isoform_assignment import ReadAssignment, ReadAssignmentType as T
from isoquant_lib.gene_info import TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_tests import test_reference_gene_fallback as fallback_fixture
from isoquant_tests import test_terminal_gene_ambiguity as terminal_fixture


class ReferenceReasonProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = terminal_fixture.TerminalGeneAmbiguityTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)

    def source(self):
        return self.fixture.match()[0]

    def counter(self, source, models):
        gene, tx = self.fixture.counters()
        ctx = NS(store=NS(transcript_model_storage=models, construction_assignment={}),
                 gene_info=source.gene_info, args=self.fixture.args,
                 string_pools=self.fixture.pools)
        counter = ModelReadCounter(ctx, tx, gene)
        counter.build_model_read_assignments([source])
        return counter.model_read_assignments[0], gene, tx

    def unique_result(self, gene="G"):
        source = self.source()
        source.additional_attributes["unrelated"] = "retained"
        model = TranscriptModel("chr1", "+", "novel_pas", gene,
                                source.corrected_exons, MT.novel_in_catalog)
        result, gene_counter, tx_counter = self.counter(source, [model])
        return source, result, gene_counter, tx_counter

    def test_final_unique_does_not_inherit_reference_terminal_reason(self):
        source, result, gene, tx = self.unique_result()
        self.assertEqual(result.assignment_type, T.unique)
        self.assertNotIn("gene_assignment_reason", result.additional_attributes)
        self.assertEqual(result.additional_attributes["reference_gene_assignment_reason"],
                         "same_gene_terminal_ambiguity")
        self.assertEqual(result.additional_attributes["unrelated"], "retained")
        self.assertEqual(gene.feature_counter["G"].get(0), 1)
        self.assertEqual(tx.feature_counter["novel_pas"].get(0), 1)
        self.assertIn("gene_assignment_reason", source.additional_attributes)
        self.assertNotIn("reference_gene_assignment_reason", source.additional_attributes)
        self.assertIsNot(result.additional_attributes, source.additional_attributes)

    def test_changed_final_gene_does_not_receive_old_generic_reason(self):
        # Counter boundary: model-to-gene joining is independent of this fixture.
        source, result, gene, tx = self.unique_result(gene="H")
        self.assertEqual({m.assigned_gene for m in result.isoform_matches}, {"H"})
        self.assertNotIn("gene_assignment_reason", result.additional_attributes)
        self.assertEqual(result.additional_attributes["reference_gene_assignment_reason"],
                         source.additional_attributes["gene_assignment_reason"])
        self.assertEqual(gene.feature_counter["H"].get(0), 1)

    def test_final_reason_takes_precedence_without_mutating_either_input(self):
        source = self.source()
        result = ReadAssignment(source.read_id, T.noninformative, self.fixture.pools)
        result.additional_attributes = {"gene_assignment_reason": "final_reason", "final_only": "yes"}
        source_attrs = dict(source.additional_attributes)
        final_attrs = result.additional_attributes
        ModelReadCounter._copy_read_aux(result, source)
        self.assertEqual(result.additional_attributes["gene_assignment_reason"], "final_reason")
        self.assertEqual(result.additional_attributes["reference_gene_assignment_reason"],
                         "same_gene_terminal_ambiguity")
        self.assertEqual(result.additional_attributes["final_only"], "yes")
        self.assertEqual(source.additional_attributes, source_attrs)
        self.assertEqual(final_attrs, {"gene_assignment_reason": "final_reason", "final_only": "yes"})

    def test_final_matcher_can_independently_report_terminal_ambiguity(self):
        source, models, _, _ = self.fixture.match()
        result, gene, tx = self.counter(source, models)
        self.assertEqual(result.assignment_type, T.inconsistent_ambiguous)
        self.assertEqual(result.gene_assignment_type, T.inconsistent_non_intronic)
        self.assertEqual(result.additional_attributes["gene_assignment_reason"],
                         "same_gene_terminal_ambiguity")
        self.assertEqual(result.additional_attributes["reference_gene_assignment_reason"],
                         "same_gene_terminal_ambiguity")
        self.assertEqual(gene.feature_counter["G"].get(0), 1)
        self.assertFalse(tx.all_features)

    def test_gene_only_fallback_keeps_reference_provenance_and_confirmation_gate(self):
        source, _, _, _ = self.fixture.match(partial=True)
        result, gene, tx = self.counter(source, [])
        self.assertEqual(result.assignment_type, T.noninformative)
        self.assertEqual(result.gene_assignment_type, T.inconsistent_non_intronic)
        self.assertEqual(result.additional_attributes["gene_assignment_source"], "reference_gene_only")
        self.assertEqual(result.additional_attributes["reference_gene_assignment"],
                         "inconsistent_non_intronic")
        self.assertEqual(result.additional_attributes["reference_gene_assignment_reason"],
                         "same_gene_terminal_ambiguity")
        self.assertNotIn("gene_assignment_reason", result.additional_attributes)
        self.assertEqual({m.assigned_gene for m in result.isoform_matches}, {"G"})
        self.assertTrue(all(m.assigned_transcript is None for m in result.isoform_matches))
        self.assertEqual(gene.feature_counter["G"].get(0), 1)
        self.assertNotIn("G", gene.confirmed_features)
        self.assertFalse(tx.all_features)
        gene.dump()
        self.assertIn("G\t0.00", Path(gene.output_counts_file_name).read_text())
        self.assertEqual(source.additional_attributes["gene_assignment_reason"],
                         "same_gene_terminal_ambiguity")

    def test_gene_only_fallback_still_rejects_opposite_and_nonexonic_evidence(self):
        fixture = fallback_fixture.ReferenceGeneFallbackTest()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        for kwargs in ({"strand": "-"}, {"exons": [(220, 280)]}):
            with self.subTest(kwargs=kwargs):
                source = fixture.source(T.inconsistent_non_intronic, **kwargs)
                source.additional_attributes["gene_assignment_reason"] = "same_gene_terminal_ambiguity"
                counter, gene, tx = fixture.counter()
                counter.build_model_read_assignments([source])
                result = counter.model_read_assignments[0]
                self.assertFalse(result.isoform_matches)
                self.assertNotIn("gene_assignment_reason", result.additional_attributes)
                self.assertEqual(result.additional_attributes["reference_gene_assignment_reason"],
                                 "same_gene_terminal_ambiguity")
                self.assertFalse(gene.all_features)
                self.assertFalse(tx.all_features)

    def test_serialization_and_read_info_keep_reference_marker(self):
        source, result, _, _ = self.unique_result()
        buffer = io.BytesIO()
        result.serialize(buffer)
        buffer.seek(0)
        loaded = ReadAssignment.deserialize(buffer, result.gene_info, self.fixture.pools)
        self.assertEqual(loaded.additional_attributes, result.additional_attributes)
        printer = object.__new__(ReadInfoPrinter)
        printer.output_file = io.StringIO()
        printer.assignment_checker = NS(check=lambda _: True)
        printer.params = NS(check_canonical=False, cage=None)
        printer.add_read_info(loaded)
        fields = printer.output_file.getvalue().strip().split("\t")
        self.assertEqual(fields[3], "G")
        self.assertEqual(fields[6], "unique")
        self.assertIn("reference_gene_assignment_reason=same_gene_terminal_ambiguity;", fields[-1])
        self.assertNotIn(" gene_assignment_reason=", " " + fields[-1])


if __name__ == "__main__":
    unittest.main()
