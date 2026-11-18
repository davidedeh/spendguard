import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET_LOUDNESS = -16
TRUE_PEAK = -1.5
LOUDNESS_RANGE = 11
SAMPLE_RATE = 48_000
SILENCE_THRESHOLD = '-45dB'
CLEANING_FILTERS = (
    f'silenceremove=start_periods=1:start_threshold={SILENCE_THRESHOLD}:start_silence=0.05,'
    'areverse,'
    f'silenceremove=start_periods=1:start_threshold={SILENCE_THRESHOLD}:start_silence=0.05,'
    'areverse,'
    'highpass=f=80,'
    'afftdn=nr=10:nf=-40'
)
BEATS = ('1', '2', '3', '4a', '4b', '5', '6', '7')
LOUDNORM_JSON_PATTERN = re.compile(r'\{[^{}]*"input_i"[^{}]*\}', re.DOTALL)
WHISPER_MODEL = Path.home() / '.cache' / 'whisper-cpp' / 'ggml-base.en.bin'
WHISPER_MODEL_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin'


def loudnorm(measured: dict[str, str] | None) -> str:
    target = f'loudnorm=I={TARGET_LOUDNESS}:TP={TRUE_PEAK}:LRA={LOUDNESS_RANGE}'
    if measured is None:
        return f'{target}:print_format=json'
    return (
        f'{target}:measured_I={measured["input_i"]}:measured_TP={measured["input_tp"]}'
        f':measured_LRA={measured["input_lra"]}:measured_thresh={measured["input_thresh"]}'
        f':offset={measured["target_offset"]}:linear=true'
    )


def measure(source: Path) -> dict[str, str]:
    result = subprocess.run(
        [
            'ffmpeg',
            '-hide_banner',
            '-i',
            str(source),
            '-af',
            f'{CLEANING_FILTERS},{loudnorm(None)}',
            '-f',
            'null',
            '-',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    found = LOUDNORM_JSON_PATTERN.search(result.stderr)
    if found is None:
        raise SystemExit(f'ffmpeg printed no loudness measurement for {source}')
    measured: dict[str, str] = json.loads(found.group(0))
    return measured


def duration(path: Path) -> float:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return round(float(result.stdout), 3)


def words(audio: Path) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / 'transcript'
        subprocess.run(
            [
                'whisper-cli',
                '-m',
                str(WHISPER_MODEL),
                '-f',
                str(audio),
                '-ml',
                '1',
                '-oj',
                '-of',
                str(prefix),
                '-np',
            ],
            capture_output=True,
            check=True,
        )
        segments = json.loads(prefix.with_suffix('.json').read_text())['transcription']
    return [
        {'text': segment['text'].strip(), 'seconds': segment['offsets']['from'] / 1000}
        for segment in segments
        if segment['text'].strip()
    ]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit('usage: prepare_narration.py <narration directory> <output directory>')
    source_directory, output_directory = Path(sys.argv[1]), Path(sys.argv[2])
    if not WHISPER_MODEL.exists():
        sys.exit(
            f'Download the transcription model: curl -fL -o {WHISPER_MODEL} {WHISPER_MODEL_URL}'
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    narration: dict[str, dict[str, object]] = {}
    for beat in BEATS:
        source = source_directory / f'beat-{beat}.m4a'
        if not source.exists():
            raise SystemExit(f'{source} is missing')
        output = output_directory / f'beat-{beat}.wav'
        subprocess.run(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel',
                'error',
                '-y',
                '-i',
                str(source),
                '-af',
                f'{CLEANING_FILTERS},{loudnorm(measure(source))}',
                '-ar',
                str(SAMPLE_RATE),
                '-ac',
                '1',
                str(output),
            ],
            check=True,
        )
        narration[beat] = {'file': output.name, 'seconds': duration(output), 'words': words(output)}
    (output_directory / 'narration.json').write_text(json.dumps(narration, indent=2))
    print(json.dumps({beat: entry['seconds'] for beat, entry in narration.items()}))


if __name__ == '__main__':
    main()
