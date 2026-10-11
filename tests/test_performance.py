import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from rockcut.profiling import PhaseProfiler
from rockcut.performance import save, digest, load_controls, measure, synthetic_sources, assigned_piece
from rockcut.block import Block
from rockcut.slab3d import SlabSpec, build_slab
from rockcut.problem import Problem
from rockcut.local import LocalOptimizer


class ProfilingTests(unittest.TestCase):
    def test_nested_exclusive_times_do_not_double_count(self):
        profiler=PhaseProfiler()
        with patch('rockcut.profiling.time.perf_counter',side_effect=[0.,1.,3.,4.]):
            with profiler.span('outer'):
                with profiler.span('inner'):
                    pass
        rows={r['phase']:r for r in profiler.summary()['phases']}
        self.assertEqual(rows['outer']['inclusive_seconds'],4.)
        self.assertEqual(rows['outer']['exclusive_seconds'],2.)
        self.assertEqual(rows['inner']['exclusive_seconds'],2.)
        self.assertEqual(profiler.summary()['accounted_wall_seconds'],4.)

    def test_exception_still_records_time_and_patches_are_restored(self):
        from rockcut import block_solver
        original=block_solver.build_slab
        profiler=PhaseProfiler()
        with self.assertRaises(RuntimeError):
            with profiler.instrument():
                self.assertIsNot(block_solver.build_slab,original)
                with profiler.span('failing'):
                    raise RuntimeError('expected')
        self.assertIs(block_solver.build_slab,original)
        self.assertFalse(profiler.stack)
        self.assertEqual(profiler.rows['failing']['calls'],1)


class ControlTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is unavailable in this interpreter')
    def test_cuda_matches_cpu_loss_and_gradient_on_a_concave_region(self):
        problem=Problem([[0,0],[8,0],[8,3],[4,3],[4,7],[0,7]],1)
        batch=np.array([[[0.,0.,.2],[1.8,.3,.6],[2.,2.,.4]]])
        values=[]
        for device in ('cpu','cuda'):
            local=LocalOptimizer(problem,device)
            q=torch.tensor(batch,dtype=torch.float64,device=device,requires_grad=True)
            loss=local.penalty(q);loss.sum().backward()
            values.append((loss.detach().cpu().numpy(),q.grad.detach().cpu().numpy()))
        for cpu,gpu in zip(*values):
            np.testing.assert_allclose(cpu,gpu,rtol=1e-8,atol=1e-8)

    def test_all_synthetic_sources_are_homogeneous_closed_connected_blocks(self):
        for name,recipe,side,lower in synthetic_sources():
            block=Block.from_dict(recipe)
            self.assertEqual(block.diagnostics()['boundary_shells'],1)
            spec=SlabSpec.from_dict({'lower':lower,'upper':lower+60})
            slab=build_slab(block,spec)
            self.assertGreater(slab.footprint.area,3*side**2)
            if name=='split_slab':
                self.assertEqual(slab.footprint.geom_type,'MultiPolygon')
                self.assertEqual(len(slab.footprint.geoms),2)
            piece=assigned_piece(block,spec)
            self.assertGreater(piece.volume(),0)

    def test_control_hash_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root/'sources/granite-reference.json',{'source':1})
            save(root/'cases/example.json',{'id':'example'})
            save(root/'suite.json',{'shared_reference_sha256':digest(root/'sources/granite-reference.json'),
                'cases':[{'id':'example','path':'cases/example.json','sha256':digest(root/'cases/example.json')}]})
            self.assertEqual(load_controls(root)[1],[{'id':'example'}])
            save(root/'cases/example.json',{'id':'tampered'})
            with self.assertRaises(ValueError):load_controls(root)

    def test_measurement_has_warmup_and_repeated_samples(self):
        calls=[]
        with patch('rockcut.performance.time.perf_counter',side_effect=[0.,1.,2.,4.,5.,8.]):
            result=measure(lambda:calls.append(1),'cpu',3)
        self.assertEqual(len(calls),4)
        self.assertEqual(result['samples_seconds'],[1.,2.,3.])
        self.assertEqual(result['median_seconds'],2.)
        self.assertIsNone(result['peak_cuda_allocated_bytes'])
        with self.assertRaises(ValueError):measure(lambda:None,'cpu',0)


if __name__=='__main__':unittest.main()
