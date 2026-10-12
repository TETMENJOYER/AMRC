"""One budget for direction screening, complete parallel plans and 2D refinement."""
from dataclasses import dataclass, asdict
import copy
import math
import time
import numpy as np
from shapely.errors import GEOSException

from .block import Frame, mesh_arrays
from .cut_plan import parallel_plan
from .slab3d import build_slab
from .problem import Problem
from .solver import SearchConfig, solve, VARIANTS
from .seeds import shelf_seeds, fill_gaps
from .grid import rigid_grid_candidates
from .verify import check_candidate, verified_subset
from .packing3d import verify_layout
from .verify_block import block_inputs, verify_block_result
from .machine import machine_policy, setup_candidates, make_setup, SourceMass


@dataclass
class BlockSearchConfig:
    seconds: float = 120
    seed: int = 20261009
    direction_count: int = 12
    shortlist: int = 3
    coarse_seconds_per_slab: float = .25
    fine_seconds_per_slab: float = 2
    fine_plans: int = 2
    refine_pairs: int = 2
    batch_size: int = 6
    adam_steps: int = 100
    polish_iterations: int = 80
    iterations_per_n: int = 40
    variant: str = 'combined'
    device: str = 'auto'
    threads: int = 1
    alignment_seconds: float | None = 0.
    alignment_stop_mode: str = 'convergence'
    alignment_method_policy: str = 'adaptive'
    quantity_pass: str = 'off'
    quantity_reserve_seconds: float = 0.
    quantity_slab_seconds: float = 12.
    quantity_attempt_seconds: float = 3.
    grid_seed: str = 'rigid'
    grid_seconds: float = 1.

    def __post_init__(self):
        if self.grid_seed not in ('off', 'rigid'):
            raise ValueError('grid_seed must be off or rigid')
        if not math.isfinite(self.grid_seconds) or self.grid_seconds <= 0:
            raise ValueError('grid_seconds must be finite and positive')
        if self.quantity_pass not in ('off', 'long', 'retry'):
            raise ValueError('quantity_pass must be off, long or retry')
        if not math.isfinite(self.quantity_reserve_seconds) or self.quantity_reserve_seconds < 0:
            raise ValueError('quantity_reserve_seconds must be finite and nonnegative')
        if self.quantity_pass == 'off' and self.quantity_reserve_seconds:
            raise ValueError('A quantity reserve requires an enabled quantity_pass')
        if self.quantity_pass != 'off' and not 0 < self.quantity_reserve_seconds < self.seconds:
            raise ValueError('Enabled quantity_pass needs a reserve strictly inside the total seconds')
        for key in ('quantity_slab_seconds', 'quantity_attempt_seconds'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.alignment_seconds is not None and (not math.isfinite(self.alignment_seconds) or self.alignment_seconds < 0):
            raise ValueError('alignment_seconds must be finite and nonnegative')
        if self.alignment_stop_mode not in ('convergence', 'time'):
            raise ValueError('alignment_stop_mode must be convergence or time')
        if self.alignment_method_policy not in ('always', 'adaptive'):
            raise ValueError('alignment_method_policy must be always or adaptive')
        for key in ('seconds', 'coarse_seconds_per_slab', 'fine_seconds_per_slab'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('direction_count','shortlist','fine_plans','batch_size','adam_steps',
                    'polish_iterations','iterations_per_n','threads'):
            if not isinstance(getattr(self,key),int) or getattr(self,key)<1:
                raise ValueError(f'{key} must be a positive integer')
        if not isinstance(self.refine_pairs,int) or self.refine_pairs<0:
            raise ValueError('refine_pairs must be nonnegative')
        if not isinstance(self.seed,int) or self.seed<0 or self.variant not in VARIANTS:
            raise ValueError('Invalid seed or inner variant')
        if self.device not in ('auto','cpu','cuda'):
            raise ValueError('Invalid device')


def directions(block, count, seed, supplied=None):
    result=[]
    def add(n):
        unit=Frame.from_dict({'normal':n}).basis[:,2]
        if not any(np.linalg.norm(unit-old)<1e-8 for old in result):
            result.append(unit)
    if supplied is not None:
        if not supplied:
            raise ValueError('Explicit directions must be nonempty')
        for n in supplied:
            add(n)
        return result
    for n in ([0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]):
        add(n)
    vertices, faces=mesh_arrays(block.solid)
    for tri in vertices[faces]:
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        if np.linalg.norm(n)>0:
            add(n); add(-n)
    rng=np.random.default_rng(seed)
    while len(result)<count:
        n=rng.normal(size=3); add(n); add(-n)
    return result[:count]


def total_count(candidate):
    return sum(r['n'] for r in candidate['records'])


def solve_block(inputs, config=None, progress=None):
    cfg=config or BlockSearchConfig()
    started=time.monotonic(); global_deadline=started+cfg.seconds
    ordinary_seconds=cfg.seconds-cfg.quantity_reserve_seconds
    deadline=started+ordinary_seconds
    block, side, thickness=block_inputs(inputs)
    history=[]; candidates=[]; evaluations=[]
    def emit(event, **fields):
        row={'event':event,'elapsed':round(time.monotonic()-started,3),**fields}
        history.append(row)
        if progress:
            progress(row)
    policy=machine_policy(inputs)
    if policy is not None:
        setups, setup_summary=setup_candidates(block,inputs,cfg.direction_count)
        normals=[n for n,s in setups]
        emit('stable_setups',**setup_summary)
    else:
        normals=directions(block,cfg.direction_count,cfg.seed,inputs.get('directions'))
        setups=[(n,None) for n in normals]
    emit('block_start',directions=len(normals),volume_mm3=block.solid.volume())

    def geometry(normal, stage_deadline, name, setup=None):
        plan=parallel_plan(block,normal,thickness)
        candidate={'id':name,'plan':plan,'geometries':[],'problems':[],'records':[],
                   'geometry_complete':True,'area_upper_bound':0,'machine_setup':setup}
        for i,spec in enumerate(plan.specs):
            if time.monotonic()<stage_deadline:
                try:
                    slab=build_slab(block,spec)
                    problem=(Problem.from_region(slab.footprint,side,f'{block.name}_slab_{i}','mm')
                             if not slab.footprint.is_empty and slab.footprint.area>=3*side**2*(1-1e-12) else None)
                    bound=problem.area_upper_bound if problem is not None else 0
                    candidate['area_upper_bound']+=bound
                    diagnostic=slab.diagnostics
                except (ValueError,GEOSException) as exc:
                    # Reject this numerical direction evaluation, preserve its
                    # cut pieces as unused material, and continue other plans.
                    slab=problem=None;bound=None
                    diagnostic={'evaluation_error':str(exc)}
                    candidate['geometry_complete']=False
                    emit('direction_geometry_failure',candidate_id=name,slab=i,reason=str(exc))
            else:
                slab=problem=None; bound=None; diagnostic=None
                candidate['geometry_complete']=False
            candidate['geometries'].append(slab); candidate['problems'].append(problem)
            candidate['records'].append({'index':i,'spec':spec.to_dict(),'angle_unit':'radians',
                'state':[],'n':0,'status':'not_packed' if slab is not None else 'geometry_budget_exhausted',
                'area_upper_bound':bound,'problem':problem.to_dict() if problem else None,
                'geometry':diagnostic,'slab_volume_mm3':plan.pieces[i].volume()})
        if not candidate['geometry_complete']:
            candidate['area_upper_bound']=None
        emit('direction_geometry',candidate_id=name,normal=plan.frame.basis[:,2].tolist(),
             slabs=len(plan.specs),area_upper_bound=candidate['area_upper_bound'],
             complete=candidate['geometry_complete'],
             **({'support_id':setup['support_id'],'yaw_degrees':setup['yaw_degrees']} if setup else {}))
        evaluations.append(candidate)
        return candidate

    geometry_deadline=min(deadline,started+ordinary_seconds*.25)
    for i,(normal,setup) in enumerate(setups):
        if i and time.monotonic()>=geometry_deadline:
            break
        candidates.append(geometry(normal,geometry_deadline,f'direction_{i}',setup))
    # Even an exhausted budget retains a complete cut plan with empty placements.
    if not candidates:
        candidates.append(geometry(normals[0],time.monotonic(),'fallback',setups[0][1]))
    complete=[c for c in candidates if c['geometry_complete']]
    ranked=sorted(complete,key=lambda c:c['area_upper_bound'],reverse=True)[:cfg.shortlist]
    if not ranked:
        ranked=[candidates[0]]

    def coarse(candidate, stage_deadline):
        for i,(problem,spec) in enumerate(zip(candidate['problems'],candidate['plan'].specs)):
            record=candidate['records'][i]
            if problem is None or time.monotonic()>=stage_deadline:
                continue
            local_deadline=min(stage_deadline,time.monotonic()+cfg.coarse_seconds_per_slab)
            if cfg.grid_seed == 'rigid':
                grid_started=time.monotonic()
                grid_deadline=min(local_deadline,grid_started+min(cfg.grid_seconds,cfg.coarse_seconds_per_slab*.5))
                grid_rows=[]
                for physical,parameters in rigid_grid_candidates(problem,grid_deadline):
                    original_count=len(physical)
                    physical,check=verified_subset(physical,lambda state:check_candidate(
                        problem,state,lambda q:verify_layout(block,spec,side,q,slab_local=candidate['plan'].pieces[i])))
                    grid_rows.append({**parameters,'original_n':original_count,'verified_n':len(physical)})
                    if check['accepted'] and len(physical)>record['n']:
                        record.update(state=physical.tolist(),n=len(physical),status='coarse_grid_verified')
                    if time.monotonic()>=grid_deadline:
                        break
                record['coarse_grid']={'candidates':grid_rows,'elapsed_seconds':time.monotonic()-grid_started,
                    'overrun_seconds':max(0.,time.monotonic()-grid_deadline)}
            seeds=shelf_seeds(problem,limit=2,deadline=local_deadline,margin=1e-5)
            for q in seeds:
                if time.monotonic()<local_deadline:
                    q=fill_gaps(problem,q,local_deadline)
                physical=problem.denormalize(q)
                if len(physical)<=record['n']:
                    continue
                original_count=len(physical)
                physical,check=verified_subset(physical,lambda state:check_candidate(
                    problem,state,lambda q:verify_layout(block,spec,side,q,slab_local=candidate['plan'].pieces[i])))
                if check['accepted'] and len(physical)>record['n']:
                    record.update(state=physical.tolist(),n=len(physical),status='coarse_verified')
                    record['coarse_seed']={'original_n':original_count,'retained_n':len(physical)}
            emit('coarse_slab',candidate_id=candidate['id'],slab=i,n=record['n'])
        emit('coarse_plan',candidate_id=candidate['id'],n_total=total_count(candidate))

    coarse_deadline=min(deadline,started+ordinary_seconds*.55)
    for c in ranked:
        coarse(c,coarse_deadline)

    # Explore nearby directions using achieved counts, not a single section's area.
    refined=[]
    best=max(ranked,key=total_count)
    refinement_deadline=min(deadline,started+ordinary_seconds*.75)
    normal=best['plan'].frame.basis[:,2]
    axes=(best['plan'].frame.basis[:,:2].T[:cfg.refine_pairs] if policy is None else
          ([None] if cfg.refine_pairs and policy.get('yaw_angles_degrees') is None else []))
    for axis in axes:
        for sign in (-1,1):
            if time.monotonic()>=refinement_deadline:
                break
            if policy is None:
                setup=None
                tilted=normal*math.cos(math.radians(5))+sign*axis*math.sin(math.radians(5))
            else:
                base=best['machine_setup']
                setup=make_setup(SourceMass.from_block(block,policy),policy,
                                 base['support_source_normal'],base['yaw_degrees']+sign*5,base['support_id'])
                tilted=np.asarray(setup['source_to_machine']['basis']).T @ np.array([1.,0.,0.])
            c=geometry(tilted,refinement_deadline,f'refine_{len(refined)}',setup)
            if c['geometry_complete']:
                coarse(c,refinement_deadline)
                refined.append(c)
    ranked+=refined
    fine=sorted(ranked,key=total_count,reverse=True)[:cfg.fine_plans]
    jobs=[]
    orders=[]
    for c in fine:
        order=sorted((i for i,p in enumerate(c['problems']) if p and p.area_upper_bound>c['records'][i]['n']),
                     key=lambda i:c['problems'][i].area_upper_bound-c['records'][i]['n'],reverse=True)
        orders.append(order)
    for pos in range(max(map(len,orders),default=0)):
        jobs.extend((c,order[pos]) for c,order in zip(fine,orders) if pos<len(order))
    for j,(c,i) in enumerate(jobs):
        remaining=deadline-time.monotonic()
        if remaining<=.05:
            break
        allowance=min(cfg.fine_seconds_per_slab,remaining/max(1,len(jobs)-j))
        p=c['problems'][i]; spec=c['plan'].specs[i]; old=c['records'][i]
        inner=SearchConfig(seconds=allowance,seconds_per_n=allowance,
              seed=cfg.seed+j+1,batch_size=cfg.batch_size,adam_steps=cfg.adam_steps,
              polish_iterations=cfg.polish_iterations,iterations_per_n=cfg.iterations_per_n,
              device=cfg.device,threads=cfg.threads,grid_seed=cfg.grid_seed,grid_seconds=cfg.grid_seconds,
              **VARIANTS[cfg.variant])
        r=solve(p,inner,initial_state=np.asarray(old['state']).reshape(-1,3),
                acceptance_check=lambda state,sp=spec,piece=c['plan'].pieces[i]:verify_layout(block,sp,side,state,slab_local=piece))
        if r['n']>=old['n']:
            old.update(n=r['n'],state=r['state'],status='optimized_verified',
                       search={'seed':inner.seed,'elapsed_seconds':r['elapsed_seconds'],
                               'attempts':r['attempts'],'stop_reason':r['stop_reason']})
            if cfg.grid_seed == 'rigid':
                old['search'].update(grid_initialization=r['grid_initialization'],best_retention=r['best_retention'])
        emit('refine_slab',candidate_id=c['id'],slab=i,n=old['n'],n_total=total_count(c))

    best=max(ranked,key=lambda c:(total_count(c),-c['plan'].to_dict()['cut_count']))
    ordinary_elapsed=time.monotonic()-started
    quantity_summary=None
    if cfg.quantity_pass != 'off':
        from .quantity_pass import QuantityPassConfig, improve_candidate
        pass_started=time.monotonic()
        pass_deadline=min(global_deadline,pass_started+cfg.quantity_reserve_seconds)
        emit('quantity_pass_start',candidate_id=best['id'],n_total=total_count(best),
             remaining_seconds=max(0.,global_deadline-pass_started))
        quantity_summary=improve_candidate(block,best,side,QuantityPassConfig(
            policy=cfg.quantity_pass,slab_seconds=cfg.quantity_slab_seconds,
            attempt_seconds=cfg.quantity_attempt_seconds,seed=cfg.seed,
            batch_size=cfg.batch_size,adam_steps=cfg.adam_steps,
            polish_iterations=cfg.polish_iterations,iterations_per_n=cfg.iterations_per_n,
            device=cfg.device,threads=cfg.threads,grid_seed=cfg.grid_seed,grid_seconds=cfg.grid_seconds),pass_deadline,emit)
        quantity_summary.update(elapsed_seconds=time.monotonic()-pass_started,
                                requested_reserve_seconds=cfg.quantity_reserve_seconds,
                                allocated_seconds=max(0.,pass_deadline-pass_started),
                                deadline_scope='quantity_stage_inside_total_search_budget')
        if quantity_summary['stop_reason']=='global_budget_exhausted':
            quantity_summary['stop_reason']='quantity_stage_budget_exhausted'
        emit('quantity_pass_finished',candidate_id=best['id'],n_total=total_count(best),
             stop_reason=quantity_summary['stop_reason'],
             remaining_seconds=max(0.,global_deadline-time.monotonic()))
    count=total_count(best); plan=best['plan']
    volume=block.solid.volume(); produced=count*3*side**2*thickness
    unused=sum(max(0.,piece.volume()-r['n']*3*side**2*thickness)
               for piece,r in zip(plan.pieces,best['records']))
    remainder=plan.remainder.volume()
    balance={'initial_volume_mm3':volume,'product_volume_mm3':produced,'unused_in_slabs_mm3':unused,
             'final_remainder_mm3':remainder,'kerf_volume_mm3':0.,'initial_trim_volume_mm3':0.,
             'unallocated_volume_mm3':max(0.,volume-produced),
             'error_mm3':abs(volume-produced-unused-remainder),'tolerance_mm3':1e-9*max(1.,volume)}
    search_elapsed=time.monotonic()-started
    result={'schema_version':1,'kind':'block_cutting_result','inputs':copy.deepcopy(inputs),
            'config':asdict(cfg),'angle_unit':'radians','selected_candidate':best['id'],
            'plan':plan.to_dict(),'slabs':best['records'],'n_total':count,
            'product_volume_mm3':3*side**2*thickness,'yield_by_volume':produced/volume,
            'volume_balance':balance,'area_upper_bound_selected_plan':best['area_upper_bound'],
            'optimality_proven':False,'search_seconds':search_elapsed,
            'stop_reason':'global_budget_exhausted' if time.monotonic()>=global_deadline else 'configured_candidates_evaluated',
            'candidates':[{'id':c['id'],'normal':c['plan'].frame.basis[:,2].tolist(),
                          'slab_count':len(c['plan'].specs),'area_upper_bound':c['area_upper_bound'],
                          'geometry_complete':c['geometry_complete'],'n_total':total_count(c),
                          **({'support_id':c['machine_setup']['support_id'],
                              'yaw_degrees':c['machine_setup']['yaw_degrees']} if c['machine_setup'] else {})} for c in evaluations],
            'history':history,'scope':'parallel_series_one_product_size_zero_kerfs_no_initial_trim'}
    if quantity_summary is not None:
        result['quantity_pass']=quantity_summary
        result['search_stages']={'ordinary_budget_seconds':ordinary_seconds,
                                'ordinary_seconds':ordinary_elapsed,
                                'quantity_seconds':quantity_summary['elapsed_seconds'],
                                'total_budget_seconds':cfg.seconds,
                                'remaining_seconds':max(0.,global_deadline-time.monotonic())}
    if policy is not None:
        result['machine_setup']=best['machine_setup']
        result['setup_search']=setup_summary
        result['scope']='one_initial_stable_setup_vertical_parallel_final_slabs_retaining_bridges_assumed'
    audit_started=time.monotonic()
    result['validation']=verify_block_result(result)
    result['verification_seconds']=time.monotonic()-audit_started
    if not result['validation']['accepted']:
        raise RuntimeError('Final independent block plan verification failed')
    if cfg.alignment_seconds is None or cfg.alignment_seconds:
        from .alignment import AlignmentConfig
        from .postprocess import align_block_result
        result = align_block_result(result, AlignmentConfig(seconds=cfg.alignment_seconds, device=cfg.device,
                                    stop_mode=cfg.alignment_stop_mode, method_policy=cfg.alignment_method_policy), progress)
        emit('block_alignment_finished', n_total=count,
             improved_slabs=sum(r['improved'] for r in result['alignment']['slabs']))
    emit('block_finished',n_total=count,yield_by_volume=result['yield_by_volume'],candidate_id=best['id'])
    result['history'] = history
    return result
