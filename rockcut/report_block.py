"""Full parallel cut plan, per-slab layouts, world placements and volume balance."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,MultiPolygon

from .block import Frame, mesh_arrays
from .verify_block import block_inputs
from .cut_plan import parallel_plan
from .slab3d import sample_section, SlabSpec
from .problem import Problem
from .region_plot import paint_region
from .report3d import save_obj
from .verify3d import product_geometry


def write_block_report(result, output):
    if not result['validation']['accepted']:
        raise ValueError('Cannot publish an unverified block plan')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    def dump(path,data):
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    dump(output/'result.json',result)
    dump(output/'validation.json',result['validation'])
    dump(output/'inputs.json',result['inputs'])
    block,side,thickness=block_inputs(result['inputs'])
    plan=parallel_plan(block,result['plan']['normal'],thickness)
    save_obj(output/'block.obj',block.solid)
    save_obj(output/'remainder.obj',plan.frame.world_solid(plan.remainder))
    v,f=mesh_arrays(block.solid)
    setup=result.get('machine_setup')
    machine_frame=Frame.from_dict(setup['source_to_machine']) if setup else None
    if machine_frame is not None:
        stability=result['validation']['machine']
        support=np.asarray(stability['support_polygon_machine_xy_mm'])
        com=np.asarray(stability['center_of_mass_machine_mm'])
        fig_support,ax_support=plt.subplots(figsize=(7,5),layout='constrained')
        ax_support.add_patch(Patch(support,facecolor='#b9d9ce',edgecolor='#245b4a',label='Опорный многоугольник'))
        ax_support.plot(com[0],com[1],'o',color='#bd463b',label='Проекция центра масс')
        ax_support.autoscale_view();ax_support.set_aspect('equal');ax_support.margins(.15)
        ax_support.set_xlabel('x станка, мм');ax_support.set_ylabel('y станка, мм');ax_support.grid(alpha=.2)
        ax_support.set_title(f"Исходная опора на столе\nЗапас устойчивости {stability['stability_margin_mm']:.1f} мм")
        ax_support.legend(loc='best')
        fig_support.savefig(output/'initial-support.png',dpi=140)
        fig_support.savefig(output/'initial-support.svg');plt.close(fig_support)
        dump(output/'machine-setup.json',setup)
        save_obj(output/'machine-block.obj',machine_frame.world_solid(block.solid))
        save_obj(output/'machine-remainder.obj',machine_frame.world_solid(plan.frame.world_solid(plan.remainder)))
        dump(output/'machine-cuts.json',{
            'coordinate_system':'machine_mm','table_z_mm':0.,'block_moves_during_series':False,
            'separation':'retaining_bridges_then_mechanical',
            'note':'planes describe final slab boundaries; bridge strength and breakage losses are not modeled',
            'cuts':[{'slab_index':s['slab_index'],'normal':[1.,0.,0.],
                     'offset_mm':s['plane_offset_mm']+float(np.dot([1,0,0],machine_frame.origin)),
                     'cut_performed':s['cut_performed']} for s in plan.steps]})
        v=machine_frame.to_world(v)
    fig=plt.figure(figsize=(14,6),layout='constrained')
    fig.get_layout_engine().set(wspace=.15)
    ax=fig.add_subplot(121,projection='3d')
    ax.add_collection3d(Poly3DCollection(v[f],facecolor='#678294',edgecolor='none',alpha=.18,
                                      linewidth=0,rasterized=True))
    local=plan.frame.local_solid(block.solid)
    for step in plan.steps:
        if not step['cut_performed']:
            continue
        z=step['plane_offset_mm']; section=sample_section(local,z)
        polys=section.geoms if section.geom_type=='MultiPolygon' else ([] if section.is_empty else [section])
        for poly in polys:
            xy=np.asarray(poly.exterior.coords)
            points=plan.frame.to_world(np.column_stack((xy,np.full(len(xy),z))))
            if machine_frame is not None:
                points=machine_frame.to_world(points)
            ax.plot(points[:,0],points[:,1],points[:,2],color='#c47432',lw=.9,alpha=.8)
    lo,hi=v.min(0),v.max(0)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo)
    ax.set_xlabel('x, мм');ax.set_ylabel('y, мм');ax.set_zlabel('z, мм')
    ax.set_title(f"Параллельная серия: {result['plan']['cut_count']} резов\nТолщина слэба {thickness:g} мм")
    if setup:
        ax.set_title(f"Вертикальные резы, один поворот блока\nТолщина слэба {thickness:g} мм")
    ax.view_init(elev=25,azim=-55)
    ax2=fig.add_subplot(122)
    indices=np.arange(len(result['slabs']))
    ax2.bar(indices+1,[r['n'] for r in result['slabs']],color='#398b73')
    ax2.set_xlabel('Номер слэба');ax2.set_ylabel('Число изделий');ax2.set_title(f"Суммарно {result['n_total']} изделий")
    ax2.grid(axis='y',alpha=.15)
    fig.savefig(output/'overview.png',dpi=160);fig.savefig(output/'overview.svg');plt.close(fig)

    product_lines=['# World coordinates in mm; separate verified product solids']
    machine_product_lines=['# Machine coordinates in mm; fixed setup, z=0 table'] if setup else None
    vertex_offset=0; id_global=0
    with (output/'placements.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream)
        writer.writerow(['product_id','slab_index','local_x','local_y','angle_radians','world_center_x','world_center_y','world_center_z'])
        folder=output/'slabs';folder.mkdir(exist_ok=True)
        for record in result['slabs']:
            spec=SlabSpec.from_dict(record['spec'])
            dump(folder/f"slab_{record['index']:03d}.json",{'schema_version':1,'kind':'slab_packing_result',
                 'angle_unit':'radians','n':record['n'],'state':record['state'],'problem':record['problem'],
                 'source3d':{'block':result['inputs']['block'],'slab':spec.to_dict(),'product':result['inputs']['product']}})
            for state in record['state']:
                id_global+=1
                center=spec.frame.to_world([state[0],state[1],(spec.lower+spec.upper)/2])
                writer.writerow([id_global,record['index'],*state,*center])
                solid,_,_=product_geometry(spec,state,side)
                verts,faces=mesh_arrays(solid)
                product_lines.append(f'o product_{id_global}')
                product_lines.extend('v '+' '.join(format(float(x),'.17g') for x in point) for point in verts)
                product_lines.extend('f '+' '.join(str(int(k)+1+vertex_offset) for k in tri) for tri in faces)
                if machine_product_lines is not None:
                    machine_product_lines.append(f'o product_{id_global}')
                    machine_product_lines.extend('v '+' '.join(format(float(x),'.17g') for x in point)
                                                 for point in machine_frame.to_world(verts))
                    machine_product_lines.extend('f '+' '.join(str(int(k)+1+vertex_offset) for k in tri) for tri in faces)
                vertex_offset+=len(verts)
    (output/'products.obj').write_text('\n'.join(product_lines)+'\n',encoding='ascii')
    if machine_product_lines is not None:
        (output/'machine-products.obj').write_text('\n'.join(machine_product_lines)+'\n',encoding='ascii')

    pages=[]
    for start in range(0,len(result['slabs']),12):
        records=result['slabs'][start:start+12]
        rows=(len(records)+2)//3
        fig,axes=plt.subplots(rows,3,figsize=(15,3.7*rows),squeeze=False,layout='constrained')
        for ax,record in zip(axes.flat,records):
            if record['problem'] is not None:
                problem=Problem.from_dict(record['problem'])
                physical=problem.physical_region
                components=physical.geoms if physical.geom_type=='MultiPolygon' else [physical]
                drawable=[p for p in components if min(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1])>=side]
                display=drawable[0] if len(drawable)==1 else MultiPolygon(drawable)
                paint_region(ax,display)
                xmin,ymin,xmax,ymax=(display.bounds if not display.is_empty else physical.bounds)
                pad=.04*max(xmax-xmin,ymax-ymin)
                ax.set_xlim(xmin-pad,xmax+pad);ax.set_ylim(ymin-pad,ymax+pad)
            else:
                ax.text(.5,.5,'Нет размещённых изделий',ha='center',va='center',transform=ax.transAxes)
            spec=SlabSpec.from_dict(record['spec'])
            for state in record['state']:
                points=product_geometry(spec,state,side)[2]
                ax.add_patch(Patch(points,facecolor='#6dbba4',edgecolor='#245b4a',lw=.6))
            ax.set_aspect('equal');ax.grid(alpha=.1)
            ax.set_title(f"Слэб {record['index']+1}: {record['n']} изделий\n{spec.lower:.1f}…{spec.upper:.1f} мм")
            ax.set_xlabel('u, мм');ax.set_ylabel('v, мм')
        for ax in list(axes.flat)[len(records):]:
            ax.set_visible(False)
        name=f'layouts-{start//12:02d}'
        fig.savefig(output/(name+'.png'),dpi=130);fig.savefig(output/(name+'.svg'));plt.close(fig)
        pages.append(name)
    balance=result['volume_balance']
    balance_rows=''.join(f'<tr><td>{label}</td><td>{balance[key]/1e9:.6f}</td></tr>' for key,label in (
        ('initial_volume_mm3','Исходный блок'),('product_volume_mm3','Готовые изделия'),
        ('unused_in_slabs_mm3','Материал вне изделий внутри слэбов'),
        ('final_remainder_mm3','Конечный остаток'),('kerf_volume_mm3','Пропил'),('initial_trim_volume_mm3','Начальная обрезка')))
    layer_rows=''.join(f"<tr><td>{r['index']+1}</td><td>{r['n']}</td><td>{r['spec']['lower']:.3f}…{r['spec']['upper']:.3f}</td>"
        f"<td><a href=\"slabs/slab_{r['index']:03d}.json\">Координаты</a></td></tr>" for r in result['slabs'])
    candidates=''.join(f"<tr><td>{html.escape(c['id'])}</td><td>{c['n_total']}</td><td>{c['area_upper_bound'] if c['area_upper_bound'] is not None else 'не оценено'}</td></tr>" for c in result['candidates'])
    source_note=''
    machine_note=''
    alignment_note=''
    if result.get('alignment'):
        a=result['alignment']
        alignment_note=(f'<p>Доводка после поиска: {a["elapsed_seconds"]:.2f} с; '
                        f'количество {result["n_total"]} изделий сохранено. '
                        f'Улучшено слэбов: {sum(r["improved"] for r in a["slabs"])} '
                        f'из {len(a["slabs"])} обработанных. Геометрия проверена повторно. '
                        'Исходные раскладки и метрики сохранены в JSON.</p>')
        if a.get('config', {}).get('stop_mode') == 'convergence':
            stalled = sum(r.get('stop_reason') == 'plateau' for r in a['slabs'])
            target = sum(r.get('stop_reason') == 'regularity_target_reached' for r in a['slabs'])
            capped = sum(r.get('stop_reason') in ('round_limit', 'time_safety_limit') for r in a['slabs'])
            alignment_note += (f'<p>Остановка по изменению проверенной метрики: затухание — {stalled} слэбов; '
                               f'практически нулевой штраф — {target}; защитный предел — {capped}. '
                               f'Не обработано из-за общего лимита: {a["slabs_not_processed"]}.</p>')
        if a.get('policy') == 'angles_then_edges_with_shared_group_angles':
            alignment_note += ('<p>Два последовательных этапа: согласование углов и выравнивание кромок. '
                'На втором этапе взаимные углы групп зафиксированы. Остановка определяется '
                'отдельной проверенной метрикой каждого этапа; смешанный штраф служит только '
                'диагностикой. Принимаемые варианты не ухудшают защищённые угловые показатели.</p>')
    if setup:
        stability=result['validation']['machine']
        machine_note=(f"<p><b>Один устойчивый исходный установ на горизонтальном столе.</b> "
                      f"Поворот вокруг вертикали: {setup['yaw_degrees']:.2f}°; "
                      f"запас проекции центра масс до края опоры: {stability['stability_margin_mm']:.2f} мм. "
                      "Все плоскости резов вертикальны, блок во время серии не вращается. "
                      "Устойчивость проверена до первого реза. Удержание перемычками принято как условие процесса; "
                      "их прочность и потери при механическом разделении не рассчитаны.</p>"
                      '<p>Схема сверху показана в координатах станка, стол z=0. Основные OBJ и CSV сохраняют '
                      'координаты исходной модели. <a href="machine-setup.json">Положение блока</a> · '
                      '<a href="machine-cuts.json">Плоскости резов станка</a> · '
                      '<a href="machine-block.obj">Блок на станке OBJ</a> · '
                      '<a href="machine-products.obj">Изделия на станке OBJ</a> · '
                      '<a href="machine-remainder.obj">Остаток на станке OBJ</a></p>'
                      '<img src="initial-support.svg" alt="Начальная опора и проекция центра масс">')
    provenance=block.spec.get('provenance',{})
    if provenance:
        asset=provenance.get('source_asset',{}).get('extras',{})
        closure=provenance.get('closure',{})
        source_note=(f"<p><b>Тестовая реконструкция реального скана.</b> Автор: {html.escape(asset.get('author','не указан'))}; "
                     f"лицензия: {html.escape(asset.get('license','не указана'))}. "
                     f"Источник: {html.escape(asset.get('source','не указан'))}.</p>")
        if closure:
            removed=closure.get('removed_vertical_extent_glb_units',0)*provenance.get('scale_mm_per_glb_unit',1)
            source_note+=f'<p>Низ скана отсечён на {removed:.2f} мм по высоте и замкнут плоским основанием. Основание является допущением; выход рассчитан относительно подготовленного замкнутого тела.</p>'
        if provenance.get('physical_scale_confirmed') is False:
            source_note+='<p>Физический масштаб не подтверждён обмером. Применён номинальный масштаб GLB; это геометрический тест, а не производственный план.</p>'
    document=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Раскрой всего блока</title><style>body{{font:16px/1.5 system-ui;max-width:1300px;margin:28px auto;padding:0 20px;background:#f5f7fb;color:#24364a}}
img{{width:100%}}table{{border-collapse:collapse;background:white;width:100%;margin:18px 0}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}
.cards{{display:flex;gap:22px;flex-wrap:wrap}}.cards div{{background:white;padding:20px;border-radius:10px}}b{{font-weight:650}}a{{color:#196a55}}</style>
<h1>Раскрой сплошного каменного блока</h1><p>{html.escape(block.name)} · изделие {3*side:g} × {side:g} × {thickness:g} мм</p>
{source_note}
{machine_note}
<div class="cards"><div>Изделий: <b>{result['n_total']}</b></div><div>Выход по объёму: <b>{result['yield_by_volume']:.2%}</b></div>
<div>Слэбов: <b>{len(result['slabs'])}</b></div><div>Резов: <b>{result['plan']['cut_count']}</b></div></div>
<p>Полная геометрическая проверка пройдена: изделия целиком в исходном блоке и назначенных слэбах,
взаимных пересечений нет, серия резов воспроизведена на текущем остатке.</p>
<p>Найдена эвристическая раскладка для выбранной параллельной серии. Максимальность не заявляется.
Пропилы и припуски равны нулю, отдельной начальной обрезки нет. Последовательность резов изделий внутри слэба не проверяется.</p>
<p>Поиск: {result['search_seconds']:.2f} с; итоговая проверка: {result['verification_seconds']:.2f} с.
Часть слэбов может иметь только начальную раскладку из-за общего бюджета.</p>
{alignment_note}
<img src="overview.svg" alt="Выбранная серия резов и выход по слэбам">
<h2>Объёмный баланс, м³</h2><table><tr><th>Материал</th><th>Объём</th></tr>{balance_rows}</table>
<p>Ошибка баланса: {balance['error_mm3']:.6g} мм³. Пригодность неиспользованного материала для будущих изделий не классифицируется.</p>
<p><a href="result.json">Полный результат</a> · <a href="validation.json">Повторная проверка</a> ·
<a href="placements.csv">Все координаты CSV</a> · <a href="block.obj">Блок OBJ</a> ·
<a href="products.obj">Изделия OBJ</a> · <a href="remainder.obj">Конечный остаток OBJ</a></p>
<h2>Раскладки по слэбам</h2>{''.join(f'<img src="{name}.svg" alt="Раскладки изделий по слэбам">' for name in pages)}
<p>На рисунках не показаны неразмещаемые численные микрофрагменты. Исходные контуры и полный объём материала сохранены в данных.</p>
<table><tr><th>Слэб</th><th>Изделий</th><th>Диапазон по нормали, мм</th><th>Данные</th></tr>{layer_rows}</table>
<h2>Проверенные направления</h2><p>Число изделий относится к реально найденным раскладкам.
Нулевой результат без раскладки не означает невозможность направления. Верхняя оценка по площади не является готовой раскладкой.</p>
<table><tr><th>Кандидат</th><th>Найдено изделий</th><th>Верхняя оценка для серии</th></tr>{candidates}</table>
<p>Повторный аудит: <code>python -m rockcut verify result.json</code>. Исходный блок и резы пересоздаются,
сохранённые контуры, вершины изделий и флаги допустимости не используются для принятия.</p></html>'''
    (output/'report.html').write_text(document,encoding='utf-8')
    return output/'report.html'
