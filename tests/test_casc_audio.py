import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import casc_audio


class AudioLabTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open('w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerows(rows)
        return path

    def test_normalizes_punctuation_case_and_unicode_but_preserves_accents(self):
        self.assertEqual(casc_audio.words('AÇÃO, a\u0063\u0327a\u0303o!'), ['ação', 'ação'])
        self.assertEqual(casc_audio.distance(['avó'], ['avô']), 1)

    def test_micro_averages_by_reference_words(self):
        manifest = self.write_csv('manifest.csv', [
            ['id', 'reference'], ['a', 'one'], ['b', 'two three four']
        ])
        predictions = self.write_csv('results.csv', [
            ['id', 'hypothesis'], ['b', 'two three four'], ['a', '']
        ])
        report = casc_audio.score(casc_audio.read_manifest(manifest), casc_audio.read_predictions(predictions))
        self.assertEqual([row['id'] for row in report['items']], ['a', 'b'])
        self.assertEqual(report['wer'], 0.25)
        self.assertNotIn('hypothesis', report['items'][0])

    def test_insertions_can_exceed_one_hundred_percent(self):
        self.assertEqual(casc_audio.rate('one', 'one two three'), (2, 1))
        self.assertEqual(casc_audio.rate('one two three', 'one four'), (2, 3))

    def test_rejects_missing_ids_and_blank_reference(self):
        self.assertRaisesRegex(ValueError, 'missing=\\[.b.\\]', casc_audio.score,
                               [casc_audio.Clip('b', 'oi')], {'a': 'oi'})
        manifest = self.write_csv('manifest.csv', [['id', 'reference'], ['a', '...']])
        self.assertRaisesRegex(ValueError, 'reference', casc_audio.read_manifest, manifest)

    def test_rejects_duplicate_and_malformed_csv_rows(self):
        path = self.write_csv('pred.csv', [['id', 'hypothesis'], ['a', 'x'], ['a', 'y']])
        self.assertRaisesRegex(ValueError, 'duplicate ID', casc_audio.read_predictions, path)
        path = self.write_csv('pred.csv', [['id', 'hypothesis'], ['a']])
        self.assertRaisesRegex(ValueError, 'expected 2 columns', casc_audio.read_predictions, path)

    def test_inference_consumes_segments_and_scores_each_clip(self):
        sound = self.root / 'local.wav'
        sound.touch()
        model = Mock()
        model.transcribe.return_value = (iter([Mock(text=' Olá'), Mock(text=' mundo')]), None)
        ticks = iter([10.0, 10.25])
        report = casc_audio.transcribe([casc_audio.Clip('a', 'olá mundo', sound)], model, 'pt',
                                       clock=lambda: next(ticks))
        self.assertEqual(report['items'][0]['seconds'], 0.25)
        self.assertEqual(report['wer'], 0.0)
        self.assertNotIn('hypothesis', report['items'][0])
        model.transcribe.assert_called_once_with(str(sound), language='pt', beam_size=1)

    def test_cli_resolves_relative_paths_and_defaults_to_local_models(self):
        audio = self.root / 'clip.wav'
        audio.touch()
        manifest = self.write_csv('manifest.csv', [['id', 'audio', 'reference'], ['a', 'clip.wav', 'oi']])
        backend = Mock()
        backend.transcribe.return_value = (iter([Mock(text='oi')]), None)
        with patch.object(casc_audio, 'load_whisper', return_value=backend) as loader, redirect_stdout(io.StringIO()):
            self.assertEqual(casc_audio.main(['transcribe', str(manifest)]), 0)
        loader.assert_called_once_with('tiny', False)
        backend.transcribe.assert_called_once_with(str(audio), language='pt', beam_size=1)


if __name__ == '__main__':
    unittest.main()
