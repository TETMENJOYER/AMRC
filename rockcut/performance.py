"""Portable controls and repeatable device measurements for the existing solver."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import numpy as np
import torch

from .block import Block
from .block_generator import demo_experiment
from .cut_plan import parallel_plan
from .local import LocalOptimizer
from .packing3d import verify_layout
from .plane_clip import clip_z
from .problem import Problem
from .seeds import shelf_seeds
from .slab3d import SlabSpec, build_slab
from .solver import SearchConfig, VARIANTS, solve
from .verify import validate, check_candidate, verified_subset
from .verify_block import block_inputs


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_sources():
    common = {'schema_version':1, 'units':'mm'}
    l_shape = {**common, 'name':'L_prism', 'base':{'type':'prism', 'height':240,
        'top_shift':[20,-10], 'vertices':[[0,0],[1200,0],[1200,300],[600,300],[600,800],[0,800]]}}
    irregular = demo_experiment()['block']
    split = {**common, 'name':'connected_block_split_slab', 'base':{'type':'box','size':[1200,800,240]},
             'subtract':[{'type':'box','center':[600,400,180],'size':[200,1000,240]}]}
    tongue = {**common, 'name':'narrow_tongue', 'base':{'type':'prism','height':240,
        'vertices':[[0,0],[1800,0],[1800,400],[900,250],[900,1000],[0,1000]]}}
    return [('L_100',l_shape,100,60),('L_50',l_shape,50,60),('L_25',l_shape,25,60),
            ('irregular_7',irregular,100,200),('split_slab',split,100,100),('narrow_tongue',tongue,100,60)]


def assigned_piece(block, spec):
    local = spec.frame.local_solid(block.solid)
    lower, _ = clip_z(local, spec.lower, True)
    piece, _ = clip_z(lower, spec.upper, False)
    return piece


def build_controls(reference_path, output):
    output = Path(output)
    reference = json.loads(Path(reference_path).read_text(encoding='utf-8-sig'))
    save(output/'sources/granite-reference.json',
         {'schema_version': 1, 'inputs': reference['inputs'],
          'plan': {'normal': reference['plan']['normal']}})
    block,side,thickness = block_inputs(reference['inputs'])
    plan = parallel_plan(block, reference['plan']['normal'], thickness)
    manifest = {'schema_version':1, 'dtype':'float64', 'cases':[],
                'scope':'Feasible CPU reference layouts, not proven maximum counts.'}

    def record(case_id, problem, state, source, block, spec, piece):
        check = check_candidate(problem, state, lambda q:verify_layout(block,spec,problem.short_side,q,slab_local=piece))
        if not check['accepted']:
            raise RuntimeError(f'Control {case_id} failed independent geometry verification')
        path = output/f'cases/{case_id}.json'
        data = {'id':case_id,'problem':problem.to_dict(),'reference_state':np.asarray(state).tolist(),
                'reference_n':len(state),'source':source,'validation':check,
                'workload':{'n':len(state),'boundary_edges':len(problem.normals),
                            'forbidden_triangles':len(problem.obstacle_triangles),
                            'is_convex':problem.is_convex,
                            'region_components':len(list(problem.physical_region.geoms)) if problem.physical_region.geom_type=='MultiPolygon' else 1}}
        save(path,data)
        manifest['cases'].append({'id':case_id,'path':str(path.relative_to(output)).replace('\\','/'),
                                  'sha256':digest(path),'reference_n':len(state),**data['workload']})
        print(json.dumps({'event':'control_saved','id':case_id,**data['workload']}),flush=True)

    for number in (3,6,14,20,22):
        i = number-1
        saved = reference['slabs'][i]
        problem = Problem.from_dict(saved['problem'])
        record(f'granite_slab_{number}', problem, saved['state'],
               {'shared_reference':'sources/granite-reference.json','slab_index':i},block,plan.specs[i],plan.pieces[i])
    for case_id, recipe, short_side, lower in synthetic_sources():
        block = Block.from_dict(recipe)
        spec = SlabSpec.from_dict({'frame':{'normal':[0,0,1]},'lower':lower,'upper':lower+60})
        slab = build_slab(block,spec)
        problem = Problem.from_region(slab.footprint,short_side,case_id,'mm')
        piece = assigned_piece(block,spec)
        candidates = shelf_seeds(problem,limit=4,margin=1e-5,deadline=time.monotonic()+1.)
        seed = problem.denormalize(max(candidates,key=len))
        state,check = verified_subset(seed,lambda q:check_candidate(problem,q,
            lambda s:verify_layout(block,spec,short_side,s,slab_local=piece)))
        record(case_id,problem,state,{'block':recipe,'spec':spec.to_dict()},block,spec,piece)
    manifest['shared_reference_sha256']=digest(output/'sources/granite-reference.json')
    save(output/'suite.json',manifest)
    return manifest


def load_controls(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'suite.json').read_text(encoding='utf-8'))
    if digest(folder/'sources/granite-reference.json') != manifest['shared_reference_sha256']:
        raise ValueError('Shared reference hash changed')
    cases=[]
    for entry in manifest['cases']:
        path=folder/entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError(f'Control hash changed: {entry["id"]}')
        cases.append(json.loads(path.read_text(encoding='utf-8')))
    return manifest,cases


def source_context(case, folder, cache):
    source=case['source']
    if 'shared_reference' in source:
        key=source['shared_reference']
        if key not in cache:
            saved=json.loads((Path(folder)/key).read_text(encoding='utf-8'))
            block,side,thickness=block_inputs(saved['inputs'])
            cache[key]=(block,parallel_plan(block,saved['plan']['normal'],thickness))
        block,plan=cache[key];i=source['slab_index']
        return block,plan.specs[i],plan.pieces[i]
    key=json.dumps(source['block'],sort_keys=True)
    if key not in cache:
        cache[key]=Block.from_dict(source['block'])
    block=cache[key];spec=SlabSpec.from_dict(source['spec'])
    return block,spec,assigned_piece(block,spec)


def sync(device):
    if device=='cuda':
        torch.cuda.synchronize()


def measure(function, device, repeats=5):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    function();sync(device)
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(repeats):
        sync(device);started=time.perf_counter()
        function();sync(device)
        times.append(time.perf_counter()-started)
    return {'median_seconds':statistics.median(times),'min_seconds':min(times),'max_seconds':max(times),
            'samples_seconds':times,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}


def environment():
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
                           capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):
        gpu=None
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'cuda_available':torch.cuda.is_available(),'nvidia_smi':gpu,'cpu':platform.processor(),
            'dtype':'float64','threads':torch.get_num_threads()}


def device_benchmarks(folder, output, devices=('cpu','cuda'), repeats=5, steps=20):
    if steps < 1 or repeats < 1:
        raise ValueError('steps and repeats must be positive')
    _,cases=load_controls(folder)
    output=Path(output);torch.set_num_threads(1)
    available=[d for d in devices if d=='cpu' or torch.cuda.is_available()]
    rows=[];rng=np.random.default_rng(20261009)
    workloads=[]
    for case in cases:
        p=Problem.from_dict(case['problem'])
        q=p.normalize(np.asarray(case['reference_state']).reshape(-1,3))
        if len(q):
            for batch_size in (6,32):
                batch=np.repeat(q[None],batch_size,axis=0)
                batch+=rng.normal(0,.12,batch.shape)
                workloads.append((case['id'],p,batch,'control'))
    p=Problem([[0,0],[120,0],[120,120],[0,120]],1,'pair_scaling')
    for n in (24,64,128,256,512):
        for batch_size in (6,32):
            batch=rng.uniform(-50,50,(batch_size,n,3));batch[:,:,2]%=np.pi
            workloads.append((f'pair_scaling_n{n}',p,batch,'stress_not_feasible_layout'))
    cold={}
    for device in available:
        start=time.perf_counter()
        local=LocalOptimizer(p,device)
        local.relax(workloads[-1][2][:1,:2].copy(),5);sync(device)
        cold[device]=time.perf_counter()-start
    for case_id,p,batch,kind in workloads:
        cpu_local=LocalOptimizer(p,'cpu')
        reference_q=torch.tensor(batch,dtype=torch.float64,requires_grad=True)
        reference_loss_tensor=cpu_local.penalty(reference_q)
        reference_loss_tensor.sum().backward()
        reference_loss=reference_loss_tensor.detach().numpy().copy()
        reference_gradient=reference_q.grad.detach().numpy().copy()
        del cpu_local,reference_q,reference_loss_tensor
        b,n,_=batch.shape;k=len(p.obstacle_triangles)
        # Keep the scaling sweep practical on a single CPU thread. CPU and GPU
        # receive exactly the same step count for each workload, recorded below.
        actual_steps=min(steps,max(1,200000//max(1,b*(n*(n-1)//2+n*k))))
        # Conservative allocation guard for large rectangle/triangle tensors.
        estimated_bytes=b*(n*(n-1)//2+n*k)*1024
        for device in available:
            if device=='cuda' and estimated_bytes>torch.cuda.mem_get_info()[0]*.6:
                rows.append({'id':case_id,'b':b,'n':n,'device':device,'status':'skipped_memory_guard'})
                continue
            local=LocalOptimizer(p,device)
            q=torch.tensor(batch,dtype=torch.float64,device=device,requires_grad=True)
            def kernel():
                q.grad=None
                local.penalty(q).sum().backward()
            kernel();loss=local.penalty(q).detach().cpu().numpy();gradient=q.grad.detach().cpu().numpy()
            agreement=bool(np.allclose(loss,reference_loss,rtol=1e-9,atol=1e-8) and
                           np.allclose(gradient,reference_gradient,rtol=1e-8,atol=1e-8))
            if not agreement:
                raise RuntimeError(f'CPU/GPU loss or gradient mismatch: {case_id}, B={b}')
            row={'id':case_id,'kind':kind,'device':device,'b':b,'n':n,'boundary_edges':len(p.normals),
                 'forbidden_triangles':k,'estimated_temporary_bytes':estimated_bytes,
                 'status':'measured','cpu_agreement':agreement,
                 'kernel_forward_backward':measure(kernel,device,repeats),
                 'adam_with_transfers':measure(lambda:local.relax(batch,actual_steps),device,repeats),
                 'lbfgs_with_transfers':measure(lambda:local.polish(batch[0],10),device,min(3,repeats)) if b==6 else None,
                 'transfers_only':measure(lambda:torch.tensor(batch,dtype=torch.float64,device=device).cpu().numpy(),device,repeats),
                 'adam_steps':actual_steps,'requested_adam_steps':steps}
            rows.append(row)
            save(output/'microbenchmarks.json',{'environment':environment(),'cold_setup_seconds':cold,
                'requested_devices':list(devices),'available_devices':available,'rows':rows,
                'method':'One warmup, synchronized wall timing, repeated samples; Adam includes host/device copies. All float64.'})
            print(json.dumps({'event':'microbench','id':case_id,'b':b,'n':n,'device':device,
                'adam_ms':row['adam_with_transfers']['median_seconds']*1000}),flush=True)
            del q,local
            if device=='cuda':torch.cuda.empty_cache()
    return rows


def quality_benchmarks(folder, output, devices=('cpu','cuda'), seeds=(101,202), seconds=6.):
    if not np.isfinite(seconds) or seconds <= 0 or not seeds:
        raise ValueError('A positive finite budget and nonempty seeds are required')
    _,cases=load_controls(folder)
    torch.set_num_threads(1)
    available=[d for d in devices if d=='cpu' or torch.cuda.is_available()]
    rows=[];cache={};output=Path(output)
    for device in available:
        warm=LocalOptimizer(Problem([[0,0],[12,0],[12,4],[0,4]],1),device)
        warm.polish(warm.relax(np.array([[[0.,0.,.1],[2.,0.,.2]]]),5)[0],3)
        sync(device)
    jobs=[(case,seed,device) for case in cases for seed in seeds for device in available]
    np.random.default_rng(20261009).shuffle(jobs)
    for case,seed,device in jobs:
        problem=Problem.from_dict(case['problem']);block,spec,piece=source_context(case,folder,cache)
        accept=lambda q:verify_layout(block,spec,problem.short_side,q,slab_local=piece)
        baseline=np.asarray(case['reference_state']).reshape(-1,3)
        if not check_candidate(problem,baseline,accept)['accepted']:
            raise RuntimeError('Frozen baseline no longer verifies')
        cfg=SearchConfig(seconds=seconds,seconds_per_n=seconds,device=device,seed=int(seed),threads=1,
            batch_size=6,adam_steps=100,polish_iterations=80,iterations_per_n=40,**VARIANTS['combined'])
        # Same verified warm start, precision, budget and seed for each device.
        sync(device);started=time.perf_counter()
        result=solve(problem,cfg,initial_state=baseline,acceptance_check=accept)
        sync(device);wall=time.perf_counter()-started
        saved=json.loads(json.dumps(result,allow_nan=False))
        check=check_candidate(Problem.from_dict(saved['problem']),np.asarray(saved['state']).reshape(-1,3),accept)
        if not check['accepted'] or saved['n']!=check['n'] or saved['n']<case['reference_n']:
            raise RuntimeError('Device run lost baseline products or failed independent verification')
        result_path=output/f'quality/{case["id"]}-{seed}-{device}.json'
        save(result_path,saved)
        row={'id':case['id'],'seed':int(seed),'device':device,'budget_seconds':seconds,
             'wall_seconds':wall,'n':saved['n'],'reference_n':case['reference_n'],
             'accepted_2d_and_3d':True,'phase_seconds':saved['phase_seconds'],
             'search_counters':saved['search_counters'],'result_path':str(result_path.relative_to(output)).replace('\\','/')}
        rows.append(row)
        save(output/'quality.json',{'environment':environment(),'rows':rows,
             'scope':'Paired time-budgeted slab search, shared verified warm starts; all outputs rechecked in 2D and source/assigned 3D.',
             'caveat':'Two seeds give exploratory evidence. Time limits are checked between operations and may be exceeded by initialization or verification.'})
        print(json.dumps({'event':'quality','id':case['id'],'seed':int(seed),'device':device,'n':saved['n'],
                          'wall_seconds':wall}),flush=True)
    return rows
