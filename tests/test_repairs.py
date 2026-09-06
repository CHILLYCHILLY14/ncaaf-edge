import json
import unittest
from pathlib import Path
from pipeline.build import project

CFG = json.loads((Path(__file__).resolve().parents[1] / 'config/settings.json').read_text())

class ProjectionRegressionTests(unittest.TestCase):
    def test_scores_agree_with_priced_margin_and_total(self):
        game = {'game_id':'1','home':{'abbr':'H'},'away':{'abbr':'A'},'neutral':False}
        for neutral in (False, True):
            game['neutral'] = neutral
            p = project(game, {'H':3,'A':0}, 2.68,
                        {'H':{'off':-5,'def':0},'A':{'off':2,'def':0}},
                        26, 1, {'1:home':8,'1:away':6},
                        {'1':{'margin_adj':2,'total_adj':3}}, CFG)
            self.assertGreater(p['proj_home_pts'], p['proj_away_pts'])
            self.assertAlmostEqual(p['proj_home_pts']-p['proj_away_pts'], p['mu'], delta=.11)
            self.assertAlmostEqual(p['proj_home_pts']+p['proj_away_pts'], p['proj_total'], delta=.11)
