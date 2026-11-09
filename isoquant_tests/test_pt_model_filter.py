"""Isolated regressions for pt-supported model-similarity comparisons."""
import io
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from isoquant_lib.alignment.alignment_processor import AlignmentCollector, PolyATrimmed
from isoquant_lib.assignment.isoform_assignment import ReadAssignmentType
from isoquant_lib.gene_info import TranscriptModel, TranscriptModelType
from isoquant_lib.model_construction.model_filter import ModelFilter
from isoquant_lib.terminal_prediction.external_polya import external_polya_from_tag
from isoquant_lib.utils.serialization import read_dict, write_dict


class PtModelFilterTest(unittest.TestCase):
    def setUp(self):
        self.args = NS(polya_trimmed=PolyATrimmed.tag, polya_trimmed_tag="pt",
                       apa_delta=50, min_novel_count=3, delta=6)
        self.filter = object.__new__(ModelFilter)
        self.filter.args = self.args
        self.filter.use_tss_model = False
        self.filter.string_pools = None
        self.filter.store = NS(transcript_read_ids={})

    def model(self, strand="+", tid="candidate", end=1000):
        exons = [(100, 200), (300, 400), (500, end)]
        m = TranscriptModel("chr1", strand, tid, "gene", exons,
                            TranscriptModelType.novel_in_catalog)
        m.intron_path = [(201, 299), (401, 499)]
        return m

    def reads(self, m, count=3, offset=0, strand_code=None):
        code = strand_code or (1 if m.strand == "+" else 2)
        end = m.get_end() if m.strand == "+" else m.get_start()
        reads = [NS(read_id=str(i), read_group_ids=[0], polyA_found=False,
                    additional_info={"pt_terminal": (code, end + offset)})
                 for i in range(count)]
        self.filter.store.transcript_read_ids[m.transcript_id] = reads
        return reads

    def test_supported_ends_both_strands_without_sequence_polya(self):
        for strand in ("+", "-"):
            with self.subTest(strand=strand):
                m = self.model(strand)
                self.reads(m)
                info, supported = self.filter._comparison_polya_info(m)
                self.assertTrue(supported)
                self.assertEqual(info.external_polya_pos, 1001 if strand == "+" else -1)
                self.assertEqual(info.external_polyt_pos, 99 if strand == "-" else -1)
                self.assertFalse(m.polya_confirmed)
                self.assertEqual(m.intron_path, [(201, 299), (401, 499)])

    def test_window_threshold_direction_and_deduplication(self):
        m = self.model()
        for count, offset, code, expected in [(3, 50, 1, True), (3, -50, 1, True),
                                             (3, 51, 1, False), (2, 0, 1, False),
                                             (3, 0, 2, False)]:
            with self.subTest(count=count, offset=offset, code=code):
                self.reads(m, count, offset, code)
                self.assertEqual(self.filter._comparison_polya_info(m)[1], expected)
        reads = self.reads(m)
        self.filter.store.transcript_read_ids[m.transcript_id] = [reads[0]] * 3
        self.assertFalse(self.filter._comparison_polya_info(m)[1])
        reads = self.reads(m)
        for read in reads:
            read.additional_info.clear()
        self.assertFalse(self.filter._comparison_polya_info(m)[1])

    def test_other_modes_unchanged_and_end_refinement_recomputed(self):
        m = self.model()
        self.reads(m)
        for mode, tag in [(PolyATrimmed.none, None), (PolyATrimmed.all, "pt"),
                          (PolyATrimmed.tag, "xy")]:
            self.args.polya_trimmed, self.args.polya_trimmed_tag = mode, tag
            self.assertFalse(self.filter._comparison_polya_info(m)[1])
        self.args.polya_trimmed, self.args.polya_trimmed_tag = PolyATrimmed.tag, "pt"
        self.assertTrue(self.filter._comparison_polya_info(m)[1])
        m.exon_blocks[-1] = (500, 1100)
        self.assertFalse(self.filter._comparison_polya_info(m)[1])

    def test_tag_provenance_and_serialization(self):
        collector = object.__new__(AlignmentCollector)
        collector.params, collector.polya_tag = self.args, "pt"
        for value in (0, 42, -1, True, "42", None):
            for mapped in ("+", "-"):
                with self.subTest(value=value, mapped=mapped):
                    tags = {} if value is None else {"pt": value}
                    bam = NS(reference_start=99, reference_end=1000,
                             has_tag=lambda k: k in tags, get_tag=tags.__getitem__)
                    read = NS(additional_info={})
                    external = external_polya_from_tag(bam, "pt", "+")
                    collector._record_pt_terminal(bam, read, external, mapped)
                    valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    expected = {"pt_terminal": (1, 1000) if mapped == "+" else (2, 100)} if valid else {}
                    self.assertEqual(read.additional_info, expected)
                    buffer = io.BytesIO()
                    write_dict(read.additional_info, buffer)
                    buffer.seek(0)
                    self.assertEqual(read_dict(buffer), expected)

    def test_unresolved_direction_not_promoted(self):
        collector = object.__new__(AlignmentCollector)
        collector.params, collector.polya_tag = self.args, "pt"
        tags = {"pt": 20}
        bam = NS(reference_start=99, reference_end=1000,
                 has_tag=lambda k: k in tags, get_tag=tags.__getitem__)
        read = NS(additional_info={})
        collector._record_pt_terminal(bam, read, external_polya_from_tag(bam, "pt"), "+")
        self.assertEqual(read.additional_info, {})

    def test_second_pass_protects_supported_apa_but_still_merges_matches(self):
        reference = self.model(tid="reference", end=1500)
        reference.transcript_type = TranscriptModelType.known
        candidate = self.model()
        reads = self.reads(candidate)
        module = "isoquant_lib.model_construction.model_filter"
        with patch(module + ".GeneInfo"), patch(module + ".CombinedProfileConstructor"), \
                patch(module + ".LongReadAssigner") as assigner, \
                patch(module + ".is_matching_assignment", return_value=False) as matching:
            assigner.return_value.assign_to_isoform.return_value = NS(
                assignment_type=ReadAssignmentType.inconsistent_non_intronic)
            self.assertEqual(self.filter.detect_similar_isoforms([reference, candidate], True), {})
            for read in reads:
                read.additional_info.clear()
            self.assertEqual(self.filter.detect_similar_isoforms([reference, candidate], True),
                             {"candidate": "reference"})
            self.reads(candidate)
            matching.return_value = True
            self.assertEqual(self.filter.detect_similar_isoforms([reference, candidate], True),
                             {"candidate": "reference"})


if __name__ == "__main__":
    unittest.main()
