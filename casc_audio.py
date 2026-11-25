"""Compare speech-recognition results against reference transcripts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Clip:
    id: str
    reference: str
    audio: Path | None = None


def words(text):
    normalized = unicodedata.normalize('NFC', text.casefold().replace('’', "'"))
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized)


def distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, source in enumerate(left, start=1):
        current = [row]
        for col, dest in enumerate(right, start=1):
            current.append(min(previous[col] + 1, current[-1] + 1,
                               previous[col - 1] + (source != dest)))
        previous = current
    return previous[-1]


def rate(reference, hypothesis):
    expected = words(reference)
    if not expected:
        raise ValueError('reference transcript must contain at least one word')
    return distance(expected, words(hypothesis)), len(expected)


def read_rows(path, required_columns):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames or []
        if len(set(columns)) != len(columns) or not set(required_columns).issubset(columns):
            raise ValueError(f'{path.name}: expected unique columns including {", ".join(required_columns)}')
        seen = set()
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f'{path.name}:{reader.line_num}: expected {len(columns)} columns')
            key = row['id'].strip()
            if not key or key in seen:
                raise ValueError(f'{path.name}:{reader.line_num}: blank or duplicate ID {key!r}')
            seen.add(key)
            row['id'] = key
            yield row
        if not seen:
            raise ValueError(f'{path.name}: expected at least one clip')


def read_manifest(path, require_audio=False):
    columns = ('id', 'reference', 'audio') if require_audio else ('id', 'reference')
    clips = []
    for row in read_rows(path, columns):
        key = row['id']
        if not words(row['reference']):
            raise ValueError(f'{key!r}: reference transcript must contain at least one word')
        audio = None
        if require_audio:
            supplied = row['audio'].strip()
            if not supplied or not (path.parent / supplied).is_file():
                raise ValueError(f'{key!r}: audio must point to an existing local file')
            audio = path.parent / supplied
        clips.append(Clip(key, row['reference'], audio))
    return clips


def read_predictions(path):
    return {row['id']: row['hypothesis'] for row in read_rows(path, ('id', 'hypothesis'))}


def score(clips, predictions, include_transcripts=False):
    clip_ids = {clip.id for clip in clips}
    missing = sorted(clip_ids - predictions.keys())
    extra = sorted(predictions.keys() - clip_ids)
    if missing or extra:
        raise ValueError(f'prediction IDs differ from manifest: missing={missing}, extra={extra}')
    items = []
    for clip in clips:
        hypothesis = predictions[clip.id]
        errors, reference_words = rate(clip.reference, hypothesis)
        item = dict(id=clip.id, errors=errors, reference_words=reference_words,
                    wer=errors / reference_words)
        if include_transcripts:
            item.update(reference=clip.reference, hypothesis=hypothesis)
        items.append(item)
    errors = sum(item['errors'] for item in items)
    reference_words = sum(item['reference_words'] for item in items)
    if not reference_words:
        raise ValueError('expected at least one clip')
    return dict(clips=len(items), errors=errors, reference_words=reference_words,
                wer=errors / reference_words, items=items)


def load_whisper(model_name, allow_download):
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise ValueError('install the optional backend with python -m pip install -r requirements-whisper.txt') from error
    try:
        return WhisperModel(model_name, device='cpu', compute_type='int8',
                            local_files_only=not allow_download)
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError('could not load the model; check its local path or explicitly allow a download') from error


def transcribe(clips, model, language, include_transcripts=False, clock=time.perf_counter):
    predictions = {}
    timings = {}
    for clip in clips:
        start = clock()
        try:
            segments, _ = model.transcribe(str(clip.audio), language=language, beam_size=1)
            predictions[clip.id] = ' '.join(segment.text.strip() for segment in segments)
        except (OSError, ValueError, RuntimeError) as error:
            raise ValueError(f'{clip.id!r}: transcription failed; check the audio format and backend') from error
        timings[clip.id] = clock() - start
    report = score(clips, predictions, include_transcripts)
    for item in report['items']:
        item['seconds'] = timings[item['id']]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    scoring = commands.add_parser('score', help='score previously generated transcripts')
    scoring.add_argument('manifest', type=Path)
    scoring.add_argument('predictions', type=Path)
    inference = commands.add_parser('transcribe', help='run local speech recognition and score it')
    inference.add_argument('manifest', type=Path)
    inference.add_argument('--model', default='tiny', help='local Faster Whisper model path or Hugging Face model name')
    inference.add_argument('--language', default='pt')
    inference.add_argument('--allow-download', action='store_true', help='allow fetching model weights from Hugging Face')
    for command in (scoring, inference):
        command.add_argument('--include-transcripts', action='store_true', help='include transcript text in the JSON report')
    args = parser.parse_args(argv)
    try:
        if args.command == 'score':
            report = score(read_manifest(args.manifest), read_predictions(args.predictions), args.include_transcripts)
        else:
            clips = read_manifest(args.manifest, require_audio=True)
            model = load_whisper(args.model, args.allow_download)
            report = transcribe(clips, model, args.language, args.include_transcripts)
            report['backend'] = dict(engine='faster-whisper', device='cpu', compute_type='int8',
                                     language=args.language, beam_size=1)
    except (csv.Error, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
