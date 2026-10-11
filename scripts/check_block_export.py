"""Check saved report links, placement counts and independently loaded OBJ volumes."""
import argparse
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import trimesh


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths=[]

    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src') and value and not value.startswith(('http:','https:','#')):
                self.paths.append(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    folder=args.folder
    result=json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
    if not result['validation']['accepted']:
        raise ValueError('Saved plan was not accepted; run source audit')
    document=(folder/'report.html').read_text(encoding='utf-8')
    links=Links();links.feed(document)
    if not all((folder/p).exists() for p in links.paths):
        raise ValueError('Broken local report links')
    if result['inputs']['block'].get('provenance'):
        if 'Тестовая реконструкция' not in document or 'не подтверждён' not in document:
            raise ValueError('Missing source reconstruction/scale caveats')
    with (folder/'placements.csv').open(encoding='utf-8-sig',newline='') as stream:
        count=len(list(csv.DictReader(stream)))
    slab_count=len(list((folder/'slabs').glob('*.json')))
    if count != result['n_total'] or slab_count != len(result['slabs']):
        raise ValueError('Export count mismatch')
    report={'report_links':len(links.paths),'csv_products':count,'slab_files':slab_count}
    expected={'block.obj':result['volume_balance']['initial_volume_mm3'],
              'products.obj':result['volume_balance']['product_volume_mm3'],
              'remainder.obj':result['volume_balance']['final_remainder_mm3']}
    if result.get('machine_setup'):
        expected.update({'machine-'+name:volume for name,volume in list(expected.items())})
    for name,volume in expected.items():
        mesh=trimesh.load(folder/name,force='scene',process=False).to_geometry()
        if name in ('products.obj','machine-products.obj') and count == 0 and volume == 0:
            if len(mesh.vertices) or len(mesh.faces):
                raise ValueError('Zero-product export contains unexpected geometry')
            report[name]={'empty':True,'volume_mm3':0.,'relative_volume_error':0.}
            continue
        error=abs(mesh.volume-volume)/max(1.,volume)
        report[name]={'watertight':bool(mesh.is_watertight),
                      'winding_consistent':bool(mesh.is_winding_consistent),
                      'volume_mm3':float(mesh.volume),'relative_volume_error':error}
        if not mesh.is_watertight or not mesh.is_winding_consistent or error>1e-9:
            raise ValueError(f'OBJ geometry check failed: {name}')
    report['passed']=True
    (folder/'export-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
