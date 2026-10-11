# Установка и первый запуск

## Окружение

Требуется Python 3.11 или новее. Выполняйте команды из корня репозитория.

```sh
git clone https://github.com/TETMENJOYER/AMRC.git
cd AMRC
python -m venv .venv
```

Активируйте окружение:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux / macOS
source .venv/bin/activate
```

Если активация PowerShell недоступна, вместо `python` используйте
`.\.venv\Scripts\python.exe`; менять системные настройки для этого не требуется.

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[geometry3d]"
python -m rockcut --help
```

Дополнение `geometry3d` устанавливает Manifold3D и trimesh. Для одного
2D-решателя достаточно `python -m pip install -e .`; полный набор тестов
и все примеры требуют 3D-зависимостей.

## Первая 2D-раскладка

```sh
python -m rockcut solve examples/slab.json --seconds 10 --seed 42 --alignment-seconds 0 --output outputs/first-layout
python -m rockcut verify outputs/first-layout/result.json
```

Откройте `outputs/first-layout/report.html`. JSON содержит область, координаты
центров, углы в радианах, настройки и проверку. CSV содержит размещения.
`verify` пересчитывает допустимость; код завершения 0 означает успех,
2 — ошибку входа или отклонение результата.

Для проверки готовой контрольной конструкции:

```sh
python -m rockcut verify examples/known_24.json
```

Это допустимый пример, а не доказательство максимального количества изделий.

## Своя 2D-задача

Сохраните JSON по образцу `examples/slab.json`:

```json
{
  "name": "my_slab",
  "units": "mm",
  "vertices": [[0, 0], [1200, 0], [1200, 800], [0, 800]],
  "short_side": 100,
  "aspect_ratio": 3,
  "kerf": 0,
  "allowance": 0,
  "touching_allowed": true
}
```

Изделие имеет размеры 300 × 100 мм. Для невыпуклой области с несколькими
компонентами или отверстиями задайте `geometry` в формате GeoJSON
`Polygon` / `MultiPolygon` вместо `vertices`. Пример — `examples/nonconvex_slab.json`.

## Полный блок

Для небольшой синтетической задачи:

```sh
python -m rockcut block examples/block_cutting.json --seconds 30 --directions 4 --seed 42 --alignment-seconds 0 --output outputs/demo-block
python -m rockcut verify outputs/demo-block/result.json
```

Пример гранитного блока с одним устойчивым установом и вертикальными резами:

```sh
python -m rockcut block examples/granite_block_cutting_current.json --seconds 300 --directions 24 --shortlist 2 --coarse-seconds .6 --slab-seconds 3 --fine-plans 2 --refine-pairs 1 --batch-size 6 --adam-steps 100 --polish-iterations 80 --iterations-per-n 40 --variant combined --seed 42 --alignment-seconds 0 --output outputs/granite
python -m rockcut verify outputs/granite/result.json
```

В этом примере изделие 300 × 100 × 60 мм. Число направлений — число
запрошенных вариантов: бюджет может закончиться раньше обработки всех.
Проверка 3D и экспорт выполняются сверх поискового бюджета.
В папке результата находятся `report.html`, `result.json`, `placements.csv`,
JSON отдельных слэбов и OBJ блока, изделий и остатка.

Дополнительная проверка ссылок, количества изделий и объёмов OBJ:

```sh
python -m scripts.check_block_export outputs/granite
```

## Доводка без повторного поиска

```sh
python -m rockcut align outputs/first-layout/result.json --seconds 10 --output outputs/aligned
python -m rockcut verify outputs/aligned/result.json
```

Количество сохраняется, сначала согласуются направления, затем кромки.
`--seconds` здесь задаёт защитный предел времени. Без него доводка использует
критерий затухания и предел раундов. `--method-policy adaptive` — политика
по умолчанию; `always` выполняет градиентную цепочку без адаптивных пауз.

Для контрольных 3D-раскладок из измерений устройств дополнительно нужен
`--control ID`, чтобы восстановить полную объёмную проверку. Для результатов
родных команд `slab` и `block` исходная геометрия восстанавливается из JSON.

## CPU, CUDA и тесты

CPU работает с `--device cpu`; `auto` также выбирает CPU. CUDA требует
совместимых драйвера и сборки PyTorch. Выберите сборку по
[инструкции PyTorch](https://pytorch.org/get-started/locally/) для своей платформы.
Проверьте доступность перед использованием `--device cuda`:

```sh
python -c "import torch; print(torch.cuda.is_available())"
python -m unittest discover -s tests
```

Затраты 3D-геометрии и аудитов остаются существенными и при CUDA.
Сравнение устройств описано в [контрольном наборе](benchmarks/performance/README.md).
Для каждого нового запуска выбирайте отдельную выходную папку.
