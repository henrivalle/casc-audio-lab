# CASC Audio Lab

[Português](#português) · [Contributing / Como contribuir](CONTRIBUTING.md) · [Research provenance](PROVENANCE.md)

Compare speech-to-text systems on the same clips. Score transcripts from any recognizer with a small CSV, or run an optional CPU backend against your own audio. Useful for classroom labs and small experiments with Portuguese, domain vocabulary, or recording conditions.

## Run the two-minute example

Python 3.9+ is sufficient for scoring. The example uses synthetic text and requires zero dependencies.

```sh
python3 casc_audio.py score examples/manifest.csv examples/predictions.csv
```

In the `accent` row, `avó` versus `avo` counts as one word error (WER 25%). The period and exclamation mark are ignored.

Create a UTF-8 CSV called `manifest.csv` with columns `id,audio,reference` for local transcription. Use unique IDs, local audio paths relative to the CSV, and transcripts checked by a person. Quote fields with commas. The `audio` column is optional for scoring saved predictions. A predictions CSV uses exactly one `id,hypothesis` pair per manifest ID. Keep `hypothesis` empty when the model returns no words.

## Transcribe on a CPU

Use Python 3.12 for the optional backend. The first command may fetch packages. `--allow-download` allows a one-time model download. Replace it on subsequent runs with a local model directory using `--model /path/to/model` to keep inference offline.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-whisper.txt
python casc_audio.py transcribe manifest.csv --language pt --allow-download
```

The JSON report shows word error rate (WER) for each clip and for the corpus. The tool divides summed word edits by summed reference words, so long clips carry more weight and insertions can take WER above 100%. It ignores case and punctuation and preserves accents. Per-clip timings include decoding and generation and exclude model initialization. Transcripts are included only with `--include-transcripts`. Use a stable manifest and human references when comparing models.

## Português

Compare sistemas de transcrição nos mesmos áudios. Use o exemplo com textos sintéticos, que roda em Python 3.9+ sem dependências. O relatório mostra erros por amostra e a taxa de erro de palavras agregada.

```sh
python3 casc_audio.py score examples/manifest.csv examples/predictions.csv
```

Na linha `accent`, `avó` e `avo` contam como um erro de palavra (WER de 25%). O ponto final e o ponto de exclamação são ignorados.

Para testar com seus próprios áudios, use Python 3.12 e instale `requirements-whisper.txt` em um ambiente virtual. Crie um CSV UTF-8 com as colunas `id,audio,reference`. Use um identificador único, caminho local relativo ao manifesto e texto conferido manualmente. Execute `python casc_audio.py transcribe manifest.csv --language pt --allow-download` para carregar o modelo na primeira execução. Com um modelo local, use `--model /caminho/do/modelo` sem `--allow-download`. Áudios e textos ficam no seu computador durante a avaliação. O relatório só contém as transcrições se você passar `--include-transcripts`.

O projeto adapta a etapa de reconhecimento de fala de um TCC da UniEVANGÉLICA para experimentos de sala de aula e pesquisa. Veja a [proveniência](PROVENANCE.md) para o escopo do experimento original. Use o [guia de contribuição](CONTRIBUTING.md) para sugerir casos reproduzíveis em português.
