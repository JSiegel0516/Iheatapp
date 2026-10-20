"""Regression checks against the installed NOAA snapshot; no model/API calls."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.routes import chat_routes as chat
from app.services.enso_periods import episodes, EnsoScopeError


class EnsoComparisons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'knowledge/index/metadata.jsonl'
        if not path.exists():
            raise unittest.SkipTest('Build the NOAA index before running snapshot checks')
        cls.records = [json.loads(line) for line in path.read_text().splitlines() if line]

    def setUp(self):
        self.mock = patch.object(chat, 'get_retriever', return_value=SimpleNamespace(records=self.records))
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_screenshot_comparison(self):
        q = 'How does the 1997-98 compare to the 2015-2016 el nino'
        scopes, error = chat._enso_scopes(q)
        self.assertIsNone(error)
        self.assertEqual(len(scopes), 2)
        evidence = [chat._retrieve(q, scope) for scope in scopes]
        self.assertIn('November 1997–January 1998', evidence[0][0]['text'])
        self.assertIn('November 2015–January 2016', evidence[1][0]['text'])

    def test_every_indexed_episode(self):
        for run in episodes(self.records):
            start, end = run[0]['data']['year'], run[-1]['data']['year']
            with self.subTest(start=start, end=end):
                scopes, error = chat._enso_scopes(f'El Nino {start}-{end} intensity')
                self.assertIsNone(error)
                self.assertTrue(scopes)
                available = {(r['data']['year'], r['data']['season'])
                             for r in self.records if r.get('source_id') == 'S68'}
                expected = {(y, season) for y in scopes[0].years for season in chat.ONI_SEASONS}
                if expected <= available:
                    self.assertTrue(chat._retrieve('', scopes[0]))
                else:
                    with self.assertRaises(EnsoScopeError):
                        chat._retrieve('', scopes[0])

    def test_outside_episode_month_is_available(self):
        scopes, error = chat._enso_scopes('SST January 1997')
        self.assertIsNone(error)
        records = chat._retrieve('', scopes[0])
        self.assertEqual(records[0]['section'], '1997-01')

    def test_explicit_unavailable_dataset_still_blocked(self):
        scopes, error = chat._enso_scopes('SST during the 1951-52 El Nino')
        self.assertIsNone(error)
        with self.assertRaises(EnsoScopeError):
            chat._retrieve('', scopes[0])
