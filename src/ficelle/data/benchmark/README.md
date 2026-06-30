# Benchmark media fixtures

Packaged media used by Ficelle's representative modality probes
(`ficelle/auto-vision`, `ficelle/auto-video`, `ficelle/auto-audio`). Each fixture
embeds a *real* signal inside the medium so a model cannot pass by copying an
answer from the prompt — it must actually read the image, decode the video, or
transcribe the audio. They ship in the wheel so the probes work out of the box.

| File | Modality | Signal | Expected answer |
|------|----------|--------|-----------------|
| `vision.json` | image | `{number: png-data-url}` map; each PNG shows a two-digit number. The probe rotates through the set over time and asks the model to read the number. | the shown number (`37`, `58`, `64`, `92`) |
| `video.mp4` | video | a short clip showing one number frame | `73` |
| `audio.mp3` | audio | a single spoken word | `orange` |

The expected answer for video/audio is the packaged default
(`BENCHMARK_VIDEO_DEFAULT_ANSWER` / `BENCHMARK_AUDIO_DEFAULT_ANSWER` in
`router.py`). An operator can point a probe at custom media and declare its answer
via `benchmark_media.{vision,video,audio}_path` and
`benchmark_media.{video,audio}_answer` (a configured-but-missing path drops the
probe without cooldown rather than masking a bad config).

## Regenerating

Fixtures are produced deterministically by:

```bash
python scripts/gen-benchmark-fixtures.py
```

Requirements (generation time only — never at runtime): `ffmpeg` and macOS `say`
on `PATH`. Without them the script still rewrites `vision.json` (pure Python
bitmap font) and skips the audio/video clips. The numbers/word are defined at the
top of that script (`VISION_NUMBERS`, `VIDEO_NUMBER`, `AUDIO_WORD`); keep the
`router.py` defaults in sync if you change `VIDEO_NUMBER` or `AUDIO_WORD`.
