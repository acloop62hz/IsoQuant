"""An ineligible replacement must not delete a supported transcript model."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import isoquant
from isoquant_lib.alignment.alignment_processor import PolyATrimmed
from isoquant_lib.assignment.isoform_assignment import ReadAssignment,ReadAssignmentType as T
from isoquant_lib.common import junctions_from_blocks
from isoquant_lib.gene_info import GeneInfo,TranscriptModel,TranscriptModelType as MT
from isoquant_lib.model_construction.context import ModelStore
from isoquant_lib.model_construction.model_filter import ModelFilter
from isoquant_lib.model_construction.model_read_counter import ModelReadCounter
from isoquant_lib.quantification.long_read_counter import create_gene_counter,create_transcript_counter
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_lib.utils.string_pools import StringPoolManager


class ModelFilterEligibilityTest(unittest.TestCase):
    def setUp(self):
        self.args=NS(delta=None,matching_strategy="default",resolve_ambiguous="default",
                     count_exons=True,stranded="forward",check_canonical=False,graph_clustering_distance=20)
        isoquant.set_matching_options(self.args)
        self.args.min_novel_count=3
        self.args.min_novel_count_rel=.05
        self.args.min_mono_count_rel=.08
        self.args.simple_models_mapq_cutoff=10
        self.args.polya_trimmed=PolyATrimmed.tag
        self.args.polya_trimmed_tag="pt"
        self.pools=StringPoolManager()

    def model(self,tid,start=100,end=600,kind=MT.novel_in_catalog,two_exons=False):
        exons=[(start,200),(300,400),(500,end)]
        if two_exons:exons=[exons[0],(300,end)]
        m=TranscriptModel("chr1","+",tid,"gene",exons,kind)
        m.intron_path=junctions_from_blocks(exons)
        return m

    def filter(self,models,counts,component=10,isolated=False,mapqs=None):
        gi=GeneInfo.from_models(models,self.args.delta)
        store=ModelStore(gi)
        store.args=self.args
        store.transcript_model_storage=list(models)
        reads=[]
        for m,count in zip(models,counts):
            for i in range(count):
                r=ReadAssignment(m.transcript_id+str(i),T.noninformative,self.pools)
                r.chr_id="chr1"
                r.strand=r.mapped_strand="+"
                r.exons=r.corrected_exons=m.exon_blocks
                r.polya_info=PolyAInfo(m.get_end()+1,-1,-1,-1)
                r.additional_info["pt_terminal"]=(1,m.get_end())
                r.mapping_quality=(mapqs or {}).get(m.transcript_id,60)
                store.save_assigned_read(r,m.transcript_id)
                reads.append(r)
        f=ModelFilter.__new__(ModelFilter)
        f.store=store;f.args=self.args;f.gene_info=gi;f.string_pools=self.pools
        f.use_tss_model=False;f.create_nics=False;f.novel_apa=False
        f.ctx=NS(gene_info=gi,args=self.args,string_pools=self.pools,store=store,
                 intron_graph=NS(get_max_component_coverage=lambda *a:component,
                                 get_overlapping_component_max_coverage=lambda *a:component,
                                 is_monointron=lambda *a:isolated))
        # Isolate replacement eligibility from trained end-refinement models.
        f.end_processor=NS(correct_novel_transcript_ends=Mock(),_mark_terminal_confirmation=Mock())
        f.detect_similar_isoforms=Mock(wraps=f.detect_similar_isoforms)
        return f,reads

    def assert_retained(self,f,expected):
        self.assertEqual({m.transcript_id for m in f.store.transcript_model_storage},set(expected))
        for call in f.detect_similar_isoforms.call_args_list:
            self.assertTrue(set(m.transcript_id for m in call.args[0]).issubset(expected))

    def test_low_support_cannot_delete_ten_read_model_in_either_input_order(self):
        for reverse in (False,True):
            low,high=self.model("low",90),self.model("high")
            models,counts=([high,low],[10,1]) if reverse else ([low,high],[1,10])
            f,reads=self.filter(models,counts)
            snapshot=list(high.exon_blocks)
            f.filter_transcripts()
            self.assert_retained(f,{"high"})
            self.assertEqual(f.store.internal_counter["high"],10)
            self.assertNotIn("low",f.store.internal_counter)
            self.assertEqual(high.exon_blocks,snapshot)
            for r in reads:
                self.assertEqual(f.store.read_assignment_counts[r.read_id],int(r.read_id.startswith("high")))
            f.end_processor.correct_novel_transcript_ends.assert_called_once()

    def test_relative_count_cutoff_applies_to_replacement_target(self):
        f,_=self.filter([self.model("low",90),self.model("high")],[4,10],component=100)
        f.filter_transcripts()  # Existing relative cutoff is 5, not absolute 3.
        self.assert_retained(f,{"high"})

    def test_existing_cutoff_boundary_is_inclusive(self):
        for count,expected in [(4,{"high"}),(5,{"low"})]:
            f,_=self.filter([self.model("low",90),self.model("high")],[count,10],component=100)
            f.filter_transcripts()
            self.assertEqual({m.transcript_id for m in f.store.transcript_model_storage},expected)

    def test_isolated_single_intron_uses_existing_mono_relative_cutoff(self):
        f,_=self.filter([self.model("low",90,two_exons=True),self.model("high",two_exons=True)],
                        [6,10],component=100,isolated=True)
        f.filter_transcripts()  # Existing isolated-intron cutoff is 8, not 5.
        self.assert_retained(f,{"high"})

    def test_simple_low_mapq_models_are_excluded_before_comparison(self):
        f,_=self.filter([self.model("low",90,two_exons=True),self.model("high")],
                        [10,10],mapqs={"low":9})
        f.filter_transcripts()
        self.assert_retained(f,{"high"})

    def test_qualified_similar_models_still_merge(self):
        f,_=self.filter([self.model("representative",90),self.model("duplicate")],[3,10])
        f.filter_transcripts()
        self.assertEqual([m.transcript_id for m in f.store.transcript_model_storage],["representative"])
        self.assertEqual(f.store.internal_counter["representative"],3)
        self.assertNotIn("duplicate",f.store.internal_counter)

    def test_known_reference_exemption_is_preserved(self):
        f,_=self.filter([self.model("known",90,kind=MT.known),self.model("duplicate")],[0,10])
        f.filter_transcripts()
        self.assertEqual([m.transcript_id for m in f.store.transcript_model_storage],["known"])
        f.end_processor.correct_novel_transcript_ends.assert_not_called()

    def test_all_ineligible_models_are_removed_without_refinement(self):
        f,reads=self.filter([self.model("first",90),self.model("second")],[1,2])
        f.filter_transcripts()
        self.assert_retained(f,set())
        self.assertFalse(f.store.internal_counter)
        self.assertTrue(all(f.store.read_assignment_counts[r.read_id]==0 for r in reads))
        f.end_processor.correct_novel_transcript_ends.assert_not_called()

    def test_qualified_distinct_supported_apa_models_remain_separate(self):
        f,_=self.filter([self.model("long",end=900),self.model("short")],[3,10])
        f.filter_transcripts()
        self.assert_retained(f,{"long","short"})
        self.assertEqual(f.store.internal_counter["long"],3)
        self.assertEqual(f.store.internal_counter["short"],10)

    def test_retained_model_original_reads_still_reach_final_counts(self):
        f,reads=self.filter([self.model("low",90),self.model("high")],[1,10])
        f.filter_transcripts()
        self.assert_retained(f,{"high"})
        with tempfile.TemporaryDirectory() as d:
            gene=create_gene_counter(str(Path(d)/"gene"),"unique_splicing_consistent")
            tx=create_transcript_counter(str(Path(d)/"tx"),"unique_only")
            counter=ModelReadCounter(f.ctx,tx,gene)
            counter.build_model_read_assignments(reads)
            byid={r.read_id:r for r in counter.model_read_assignments}
            for r in reads:
                if r.read_id.startswith("high"):
                    result=byid[r.read_id]
                    self.assertTrue(result.assignment_type.is_unique())
                    self.assertEqual({m.assigned_transcript for m in result.isoform_matches},{"high"})
            self.assertGreaterEqual(tx.feature_counter["high"].get(0),10)
            self.assertNotIn("low",tx.all_features)


if __name__=="__main__":
    unittest.main()
