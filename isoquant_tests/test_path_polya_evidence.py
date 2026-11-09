"""PolyA evidence survives initial construction and all comparison input modes."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from isoquant_lib.assignment.isoform_assignment import MatchEventSubtype as E, ReadAssignmentType as T
from isoquant_lib.assignment.long_read_assigner import LongReadAssigner
from isoquant_lib.assignment.long_read_profiles import CombinedProfileConstructor
from isoquant_lib.common import get_exons, junctions_from_blocks
from isoquant_lib.gene_info import GeneInfo, TranscriptModel, TranscriptModelType as MT
from isoquant_lib.model_construction.context import ModelStore
from isoquant_lib.model_construction.fl_graph_constructor import GraphBasedConstructor
from isoquant_lib.model_construction.intron_graph import TerminalVertex as V
from isoquant_lib.terminal_prediction.polya_finder import PolyAInfo
from isoquant_tests import test_read_direction as direction_tests
from isoquant_tests import test_pt_model_filter as pt_tests


class PathPolyAEvidenceTest(unittest.TestCase):
    def constructor(self, strand="+", terminal=True, matching_end=False, both=False):
        fixture = direction_tests.ReadDirectionTest()
        fixture.setUp()
        # A real full-chain reference with a different annotated 3' end.
        ref_exons = ([(100,200),(300,400),(500,900)] if strand=="+" else
                     [(100,500),(600,700),(800,900)])
        end = 900 if matching_end else 600
        start = 100 if matching_end else 400
        path = (((V.polyt if terminal else V.read_start, start), (501,599), (701,799), (V.read_end,900))
                if strand=="-" else
                ((V.polyt if both else V.read_start,100), (201,299), (401,499), (V.polya if terminal else V.read_end,end)))
        ref = TranscriptModel("chr1",strand,"reference","gene_reference",ref_exons,MT.known)
        gi = GeneInfo.from_models([ref],fixture.args.delta)
        store = ModelStore(gi)
        store.args = fixture.args
        store.detected_known_isoforms = set()
        observed = get_exons((path[0][1],path[-1][1]),list(path[1:-1]))
        reads = [fixture.read(strand,exons=observed,rid=str(i)) for i in range(2)]
        profiles = Mock(wraps=CombinedProfileConstructor(gi,fixture.args))
        assigner = LongReadAssigner(gi,fixture.args,fixture.pools)
        assigner.assign_to_isoform = Mock(wraps=assigner.assign_to_isoform)
        ctx = NS(store=store,args=fixture.args,gene_info=gi,use_tss_model=False,create_alt_ts_nics=True,
                 use_technical_replicas=False,file_name_group_idx=-1,profile_constructor=profiles,assigner=assigner,
                 strand_detector=NS(get_strand=lambda *a:strand,get_clean_strand=lambda *a:strand),
                 path_storage=NS(fl_paths={path},paths={path:2},paths_to_reads={path:reads}),
                 known_isoforms_in_graph={tuple(path[1:-1]):"reference"},known_introns=set(path[1:-1]),
                 get_transcript_id=lambda:999,select_reference_gene=lambda *a:"gene_reference")
        return GraphBasedConstructor(ctx), path

    def test_initial_profile_carries_actual_full_path_tail_and_keeps_introns(self):
        for strand in ("+","-"):
            c,path=self.constructor(strand)
            c.construct_fl_isoforms()
            info=c.profile_constructor.construct_profiles.call_args.args[1]
            self.assertEqual(info.external_polya_pos,path[-1][1] if strand=="+" else -1)
            self.assertEqual(info.external_polyt_pos,path[0][1] if strand=="-" else -1)
            self.assertEqual(info.internal_polya_pos,-1)
            self.assertEqual(info.internal_polyt_pos,-1)
            self.assertEqual(len(c.store.transcript_model_storage),1)
            model=c.store.transcript_model_storage[0]
            self.assertEqual(model.transcript_type,MT.novel_in_catalog)
            self.assertEqual(tuple(model.intron_path),path[1:-1])
            self.assertEqual(c.store.internal_counter[model.transcript_id],2)
            profile=c.assigner.assign_to_isoform.call_args.args[1]
            assignment=LongReadAssigner.assign_to_isoform(c.assigner,"recheck",profile)
            self.assertEqual(assignment.assignment_type,T.inconsistent_non_intronic)
            target=E.alternative_polya_site_right if strand=="+" else E.alternative_polya_site_left
            self.assertIn(target,{e.event_type for m in assignment.isoform_matches for e in m.match_subclassifications})
            self.assertFalse(model.polya_confirmed)

    def test_bare_end_preserves_reference_instead_of_asserting_polya(self):
        for strand in ("+","-"):
            c,path=self.constructor(strand,terminal=False)
            c.construct_fl_isoforms()
            info=c.profile_constructor.construct_profiles.call_args.args[1]
            self.assertEqual((info.external_polya_pos,info.external_polyt_pos),(-1,-1))
            self.assertEqual(c.store.transcript_model_storage[0].transcript_id,"reference")
            self.assertEqual(c.store.transcript_model_storage[0].exon_blocks,c.gene_info.all_isoforms_exons["reference"])

    def test_matching_confirmed_end_still_keeps_reference(self):
        for strand in ("+","-"):
            c,path=self.constructor(strand,matching_end=True)
            c.construct_fl_isoforms()
            self.assertEqual(c.store.transcript_model_storage[0].transcript_id,"reference")

    def test_both_real_tail_vertices_are_passed_without_left_precedence(self):
        c,path=self.constructor(both=True)
        c.construct_fl_isoforms()
        info=c.profile_constructor.construct_profiles.call_args.args[1]
        self.assertEqual((info.external_polya_pos,info.external_polyt_pos),(600,100))

    def test_tss_vertices_do_not_become_polya(self):
        c,path=self.constructor(terminal=False)
        tss=((V.tss_left,path[0][1]),*path[1:-1],(V.tss_right,path[-1][1]))
        c.ctx.path_storage=NS(fl_paths={tss},paths={tss:2},paths_to_reads={tss:c.ctx.path_storage.paths_to_reads[path]})
        c.construct_fl_isoforms()
        info=c.profile_constructor.construct_profiles.call_args.args[1]
        self.assertEqual((info.external_polya_pos,info.external_polyt_pos),(-1,-1))

    def filter_and_reads(self,strand="+",mode=None,count=3,offset=0):
        fixture=pt_tests.PtModelFilterTest()
        fixture.setUp()
        f=fixture.filter
        f.args.stranded="forward"
        f.args.polya_trimmed=mode if mode is not None else pt_tests.PolyATrimmed.none
        m=fixture.model(strand)
        endpoint=m.get_end() if strand=="+" else m.get_start()
        info=(lambda:PolyAInfo(endpoint+offset+1,-1,-1,-1)) if strand=="+" else (lambda:PolyAInfo(-1,endpoint+offset-1,-1,-1))
        reads=[NS(read_id=str(i),read_group_ids=[0],mapped_strand=strand,strand=strand,
                  additional_info={},polyA_found=True,polya_info=info()) for i in range(count)]
        f.store.transcript_read_ids[m.transcript_id]=reads
        return f,m,reads

    def test_other_input_modes_use_confirmed_external_reads_on_both_strands(self):
        for mode in (pt_tests.PolyATrimmed.none,pt_tests.PolyATrimmed.all,pt_tests.PolyATrimmed.tag):
            for strand in ("+","-"):
                f,m,_=self.filter_and_reads(strand,mode)
                if mode==pt_tests.PolyATrimmed.tag:
                    f.args.polya_trimmed_tag="xy"
                info,supported=f._comparison_polya_info(m)
                self.assertTrue(supported)
                self.assertEqual(info.external_polya_pos,m.get_end()+1 if strand=="+" else -1)
                self.assertEqual(info.external_polyt_pos,m.get_start()-1 if strand=="-" else -1)
                self.assertEqual(tuple(m.intron_path),tuple(junctions_from_blocks(m.exon_blocks)))
                self.assertFalse(m.polya_confirmed)

    def test_existing_count_window_dedup_and_end_refinement_gates(self):
        for count,offset,expected in ((3,50,True),(3,-50,True),(3,51,False),(2,0,False)):
            for strand in ("+","-"):
                f,m,_=self.filter_and_reads(strand,count=count,offset=offset)
                self.assertEqual(f._comparison_polya_info(m)[1],expected)
        f,m,reads=self.filter_and_reads()
        f.store.transcript_read_ids[m.transcript_id]=[reads[0]]*3
        self.assertFalse(f._comparison_polya_info(m)[1])
        f.store.transcript_read_ids[m.transcript_id]=reads
        m.exon_blocks[-1]=(500,1100)
        self.assertFalse(f._comparison_polya_info(m)[1])

    def test_bare_internal_opposite_and_unknown_evidence_does_not_protect_model(self):
        for kind in ("not_confirmed","internal_only","opposite","unknown","conflicting_external"):
            f,m,reads=self.filter_and_reads()
            for r in reads:
                if kind=="not_confirmed":r.polyA_found=False
                elif kind=="internal_only":r.polya_info=PolyAInfo(-1,-1,m.get_end()+1,-1)
                elif kind=="opposite":r.mapped_strand="-"
                elif kind=="unknown":r.mapped_strand=r.strand="."
                elif kind=="conflicting_external":r.additional_info["external_read_strand"]="-"
            self.assertFalse(f._comparison_polya_info(m)[1],kind)

    def test_pt_requires_its_own_provenance_and_rejects_bam_conflicts(self):
        f,m,reads=self.filter_and_reads(mode=pt_tests.PolyATrimmed.tag)
        self.assertFalse(f._comparison_polya_info(m)[1])
        for r in reads:r.additional_info["pt_terminal"]=(1,m.get_end())
        self.assertTrue(f._comparison_polya_info(m)[1])
        for r in reads:r.mapped_strand="-"
        self.assertFalse(f._comparison_polya_info(m)[1])

    def test_actual_similarity_assigner_preserves_supported_apa_both_passes(self):
        for mode in (pt_tests.PolyATrimmed.none, pt_tests.PolyATrimmed.tag):
            for strand in ("+", "-"):
                f,m,reads=self.filter_and_reads(strand,mode)
                fixture=direction_tests.ReadDirectionTest()
                fixture.setUp()
                fixture.args.polya_trimmed=mode
                fixture.args.polya_trimmed_tag="pt"
                fixture.args.min_novel_count=3
                f.args=fixture.args
                f.string_pools=fixture.pools
                if mode==pt_tests.PolyATrimmed.tag:
                    for r in reads:
                        r.additional_info["pt_terminal"]=(1,m.get_end()) if strand=="+" else (2,m.get_start())
                exons=(m.exon_blocks[:-1]+[(500,1500)] if strand=="+" else
                       [(20,200)]+m.exon_blocks[1:])
                reference=TranscriptModel("chr1",strand,"reference","gene",exons,MT.known)
                for second_pass in (False,True):
                    self.assertEqual(f.detect_similar_isoforms([reference,m],second_pass),{})
                for r in reads:
                    r.polyA_found=False
                    r.additional_info.clear()
                for second_pass in (False,True):
                    self.assertEqual(f.detect_similar_isoforms([reference,m],second_pass),{m.transcript_id:"reference"})

    def test_filter_retains_supported_apa_but_can_remove_unsupported_or_merge_identical(self):
        f,m,reads=self.filter_and_reads()
        reference=TranscriptModel("chr1","+","reference","gene",[(100,200),(300,400),(500,1500)],MT.known)
        module="isoquant_lib.model_construction.model_filter"
        with patch(module+".GeneInfo"),patch(module+".CombinedProfileConstructor"),patch(module+".LongReadAssigner") as assigner,patch(module+".is_matching_assignment",return_value=False) as matching:
            assigner.return_value.assign_to_isoform.return_value=NS(assignment_type=T.inconsistent_non_intronic)
            self.assertEqual(f.detect_similar_isoforms([reference,m],True),{})
            for r in reads:r.polyA_found=False
            self.assertEqual(f.detect_similar_isoforms([reference,m],True),{m.transcript_id:"reference"})
            for r in reads:r.polyA_found=True
            matching.return_value=True
            self.assertEqual(f.detect_similar_isoforms([reference,m],True),{m.transcript_id:"reference"})


if __name__=="__main__":
    unittest.main()
