# Research provenance

The CASC capstone team at UniEVANGÉLICA supplied a Colab runner and an experiment job list. `colab_runner.py` separates the cascade (`_run_cascade`) from the end-to-end path (`_run_e2e`). The first stage of the cascade extracts a transcript. The runner associates each measurement with a job ID and audio input and measures elapsed inference time with a monotonic clock.

This adaptation ports those mechanics to a local CLI. The manifest provides stable clip IDs and local audio paths. The runner measures speech-to-text execution before applying a shared word-error calculation. A separate scoring command accepts predictions exported from a different recognizer, including the original models. To compare repeated runs, use the same manifest and separate reports for each run.

The original cascade uses Qwen2-Audio, a language model, and speech synthesis. The original end-to-end route uses Qwen2.5-Omni. Both address a broader audio interaction task. The new default backend is Faster Whisper and the measured output is transcription. Evaluating response quality and synthesized speech requires a separate protocol. The examples in this repository are synthetic and represent no historical TCC results. MIT applies to the code in this repository. Models and evaluation inputs retain their own terms.

## Em português

Adaptado a partir do executor experimental do TCC CASC da UniEVANGÉLICA, fornecido por Henrique Valle de Lima e pela equipe do trabalho. O formato preserva identificadores por amostra, a leitura de uma lista de arquivos, a separação da etapa de transcrição e a medição com relógio monotônico. A versão local compara transcrição e pode pontuar resultados exportados de outros modelos. A comparação completa entre cascata e modelo ponta a ponta envolve ainda as etapas de resposta e síntese de voz.
