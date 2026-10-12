import copy
import unittest

import numpy as np
import torch

from rockcut.block import Block, Frame
from rockcut.block_solver import BlockSearchConfig, solve_block
from rockcut.machine import SourceMass, contact_audit, machine_policy, setup_candidates, verify_machine_setup
from rockcut.verify_block import verify_block_result


def machine_input():
    return {'schema_version':1,'kind':'block_cutting_problem',
            'block':{'schema_version':1,'units':'mm','base':{'type':'box','size':[6,2,7]}},
            'product':{'short_side':1,'aspect_ratio':3,'thickness':3},
            'machine':{'kind':'vertical_fixed_setup','support_mode':'fixed',
                       'contact_tolerance_mm':1e-7,'yaw_angles_degrees':[0]}}


class MachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        analytic=machine_input()
        analytic['block']['base']['size']=[6,6.2,2.2]
        cls.result=solve_block(analytic,BlockSearchConfig(grid_seed='off', seconds=4,refine_pairs=0,
                              coarse_seconds_per_slab=.2,fine_seconds_per_slab=.2))

    def test_center_of_mass_and_known_support_margin(self):
        d=machine_input();b=Block.from_dict(d['block'])
        setups,note=setup_candidates(b,d,12)
        stability=setups[0][1]['initial_stability']
        np.testing.assert_allclose(stability['center_of_mass_source_mm'],[3,1,3.5])
        self.assertAlmostEqual(stability['support_area_mm2'],12)
        self.assertAlmostEqual(stability['stability_margin_mm'],1)
        self.assertTrue(stability['accepted'])

    def test_yaw_preserves_stability_and_only_changes_horizontal_normal(self):
        d=machine_input();d['machine']['yaw_angles_degrees']=[0,45,180]
        setups,note=setup_candidates(Block.from_dict(d['block']),d,3)
        self.assertEqual(len(setups),3)
        for n,s in setups:
            self.assertAlmostEqual(n[2],0)
            self.assertAlmostEqual(s['initial_stability']['stability_margin_mm'],1)
            np.testing.assert_allclose(Frame.from_dict(s['source_to_machine']).basis@n,[1,0,0],atol=1e-12)
        np.testing.assert_allclose(setups[2][0],[-1,0,0],atol=1e-12)

    def test_real_volume_com_rejects_overhanging_prism(self):
        d=machine_input()
        d['block']['base']={'type':'prism','vertices':[[0,0],[2,0],[2,2],[0,2]],
                            'height':6,'top_shift':[5,0]}
        b=Block.from_dict(d['block']);mass=SourceMass.from_block(b,d['machine'])
        np.testing.assert_allclose(mass.center,[3.5,1,3])
        self.assertFalse(contact_audit(mass,Frame([0,0,0],np.eye(3)),d['machine'])['accepted'])
        with self.assertRaisesRegex(ValueError,'unstable'):
            setup_candidates(b,d,12)

    def test_stability_margin_is_enforced(self):
        d=machine_input();d['machine']['support_margin_mm']=1.1
        with self.assertRaises(ValueError):setup_candidates(Block.from_dict(d['block']),d,12)

    def test_default_contact_tolerance_scales_to_small_models(self):
        d=machine_input();d['machine'].pop('contact_tolerance_mm')
        setups,_=setup_candidates(Block.from_dict(d['block']),d,1)
        self.assertTrue(setups[0][1]['initial_stability']['accepted'])
        self.assertAlmostEqual(setups[0][1]['initial_stability']['contact_tolerance_mm'],7e-9)

    def test_auto_supports_include_all_six_box_faces(self):
        d=machine_input();d['machine']['support_mode']='stable_faces';d['machine']['max_supports']=6
        setups,note=setup_candidates(Block.from_dict(d['block']),d,12)
        self.assertEqual(note['stable_supports_found'],6)
        self.assertEqual(len(setups),6)
        for n,s in setups:
            self.assertTrue(verify_machine_setup(Block.from_dict(d['block']),d,s,n)['accepted'])

    def test_whole_vertical_pipeline_analytic_count(self):
        r=self.result
        self.assertEqual(r['n_total'],8)
        self.assertEqual(r['plan']['slab_count'],2)
        self.assertAlmostEqual(r['yield_by_volume'],72/(6*6.2*2.2))
        self.assertTrue(r['validation']['machine']['vertical_cuts'])
        self.assertTrue(r['validation']['machine']['single_fixed_setup'])
        self.assertFalse(r['validation']['machine']['bridges_strength_verified'])

    def test_audit_ignores_saved_com_and_stability_flags(self):
        r=copy.deepcopy(self.result)
        r['machine_setup']['initial_stability']={'accepted':False,'center_of_mass_machine_mm':[999]*3}
        r['validation']={'accepted':False}
        self.assertTrue(verify_block_result(r)['accepted'])
        np.testing.assert_allclose(verify_block_result(r)['machine']['center_of_mass_source_mm'],[3,3.1,1.1])

    def test_audit_rejects_yaw_movement_and_nonvertical_cut(self):
        for edit in ('yaw','moving','horizontal','missing','blade'):
            r=copy.deepcopy(self.result)
            if edit=='yaw':r['machine_setup']['yaw_degrees']=30
            if edit=='moving':r['machine_setup']['block_moves_during_series']=True
            if edit=='horizontal':r['plan']['normal']=[0,0,1]
            if edit=='missing':r.pop('machine_setup')
            if edit=='blade':r['machine_setup']['blade_normal_machine']=[0,0,1]
            with self.assertRaises(ValueError):verify_block_result(r)

    def test_audit_rejects_table_penetration_and_lifted_block(self):
        for height in (-.1,.1):
            r=copy.deepcopy(self.result);r['machine_setup']['source_to_machine']['origin'][2]=height
            with self.assertRaisesRegex(ValueError,'table'):verify_block_result(r)

    def test_audit_rejects_different_cut_direction_mid_series(self):
        r=copy.deepcopy(self.result);r['plan']['cuts'][0]['plane_normal_world']=[0,1,0]
        self.assertFalse(verify_block_result(r)['accepted'])

    def test_invalid_or_conflicting_policies_fail(self):
        for change in ({'support_mode':'free'},{'support_margin_mm':-1},
                       {'yaw_angles_degrees':[]},{'contact_tolerance_mm':0},{'max_supports':0}):
            d=machine_input();d['machine'].update(change)
            with self.assertRaises(ValueError):machine_policy(d)
        d=machine_input();d['directions']=[[0,0,1]]
        with self.assertRaises(ValueError):machine_policy(d)

    def test_continuous_yaw_refinement_keeps_one_stable_base(self):
        d=machine_input();d['machine'].pop('yaw_angles_degrees')
        r=solve_block(d,BlockSearchConfig(grid_seed='off', seconds=3,direction_count=1,refine_pairs=1,
                                        coarse_seconds_per_slab=.1,fine_seconds_per_slab=.1))
        self.assertTrue(r['validation']['accepted'])
        for c in r['candidates']:
            self.assertAlmostEqual(c['normal'][2],0)
            self.assertEqual(c['support_id'],'fixed_base')


if __name__=='__main__':unittest.main()
