"""Defaults select rigid without changing quotas or optional quantity/alignment work."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rockcut.block_solver import BlockSearchConfig
from rockcut.cli import main
from rockcut.problem import Problem
from rockcut.quantity_pass import QuantityPassConfig
from rockcut.solver import SearchConfig


class GridDefaultTests(unittest.TestCase):
    def test_api_defaults_and_explicit_off(self):
        for constructor in (SearchConfig, BlockSearchConfig, QuantityPassConfig):
            with self.subTest(constructor=constructor):
                default = constructor()
                disabled = constructor(grid_seed='off')
                self.assertEqual(default.grid_seed, 'rigid')
                self.assertEqual(disabled.grid_seed, 'off')
                self.assertEqual(default.grid_seconds, 1.)
                self.assertEqual(disabled.grid_seconds, 1.)
                self.assertEqual({**vars(default), 'grid_seed': 'off'}, vars(disabled))
        self.assertEqual(BlockSearchConfig().quantity_pass, 'off')
        self.assertEqual(SearchConfig().alignment_method_policy, 'adaptive')
        self.assertEqual(BlockSearchConfig().alignment_method_policy, 'adaptive')

    def test_cli_default_and_override_reach_all_search_commands(self):
        problem = Problem([[0, 0], [6, 0], [6, 1], [0, 1]], 1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.json'
            source.write_text('{}', encoding='utf-8')
            for command, target in (('solve', 'rockcut.solver.solve'),
                                    ('slab', 'rockcut.packing3d.solve_slab'),
                                    ('block', 'rockcut.block_solver.solve_block')):
                for flags, expected in (([], 'rigid'), (['--grid-seed', 'off'], 'off'),
                                        (['--grid-seed', 'rigid'], 'rigid')):
                    with (
                        self.subTest(command=command, flags=flags),
                        patch('rockcut.cli.Problem.load', return_value=problem),
                        patch(target, return_value={}) as search,
                        patch('rockcut.report.write_report', return_value=Path(directory)/'report.html'),
                        patch('rockcut.report_block.write_block_report', return_value=Path(directory)/'report.html'),
                        redirect_stdout(io.StringIO()),
                    ):
                        self.assertEqual(main([command, str(source), '--seconds', '2', *flags]), 0)
                        cfg = search.call_args.args[1]
                        self.assertEqual(cfg.grid_seed, expected)
                        self.assertEqual(cfg.grid_seconds, 1.)
                        self.assertEqual(cfg.seconds, 2.)
                        self.assertEqual(cfg.alignment_method_policy, 'adaptive')
                        self.assertIsNone(cfg.alignment_seconds)
                        if command == 'block':
                            self.assertEqual(cfg.quantity_pass, 'off')
                            self.assertEqual(cfg.quantity_reserve_seconds, 0.)

    def test_cli_block_quantity_pass_keeps_grid_override_and_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.json'
            source.write_text('{}', encoding='utf-8')
            for flags, expected in (([], 'rigid'), (['--grid-seed', 'off'], 'off')):
                with (
                    self.subTest(flags=flags),
                    patch('rockcut.block_solver.solve_block', return_value={}) as search,
                    patch('rockcut.report_block.write_block_report', return_value=Path(directory)/'report.html'),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(['block', str(source), '--seconds', '2',
                        '--quantity-pass', 'retry', '--quantity-reserve-seconds', '.5',
                        '--grid-seconds', '.25', *flags]), 0)
                    cfg = search.call_args.args[1]
                    self.assertEqual(cfg.grid_seed, expected)
                    self.assertEqual(cfg.grid_seconds, .25)
                    self.assertEqual(cfg.quantity_pass, 'retry')
                    self.assertEqual(cfg.quantity_reserve_seconds, .5)

    def test_cli_help_describes_default_and_disable_option(self):
        for command in ('solve', 'slab', 'block'):
            stream = io.StringIO()
            with self.subTest(command=command), redirect_stdout(stream), self.assertRaises(SystemExit) as stop:
                main([command, '--help'])
            self.assertEqual(stop.exception.code, 0)
            text = ' '.join(stream.getvalue().split())
            self.assertIn('default: rigid', text)
            self.assertIn('off disables it', text)


if __name__ == '__main__':
    unittest.main()
