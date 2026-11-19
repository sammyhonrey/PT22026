"""Tests for joining WAV chunks into one file."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voiceforge.audio_join import AudioJoinError, join_wav_files


def write_wav(path, seconds=1.0, sample_rate=22050, channels=1, peak=0.5):
    frames = int(sample_rate * seconds)
    time = np.arange(frames) / sample_rate
    mono = (peak * np.sin(2 * np.pi * 220 * time)).astype(np.float32)

    if channels == 1:
        data = mono.reshape(-1, 1)
    else:
        data = np.repeat(mono.reshape(-1, 1), channels, axis=1)

    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


def test_joins_two_files_with_gap(tmp_path):
    first = write_wav(tmp_path / "a.wav", seconds=1.0)
    second = write_wav(tmp_path / "b.wav", seconds=1.0)
    output = tmp_path / "joined.wav"

    metadata = join_wav_files([first, second], output, gap_seconds=0.15)

    assert output.is_file()
    assert metadata["chunks"] == 2
    assert metadata["sample_rate"] == 22050
    assert metadata["channels"] == 1
    assert metadata["duration_seconds"] == pytest.approx(2.15, abs=0.05)

    data, rate = sf.read(str(output))
    assert rate == 22050
    assert len(data) == pytest.approx(2.15 * 22050, abs=500)


def test_output_is_16_bit_pcm(tmp_path):
    first = write_wav(tmp_path / "a.wav", seconds=0.5)

    output = tmp_path / "out.wav"
    join_wav_files([first], output)

    info = sf.info(str(output))
    assert info.subtype == "PCM_16"


def test_sample_rate_mismatch_is_rejected(tmp_path):
    first = write_wav(tmp_path / "a.wav", sample_rate=22050)
    second = write_wav(tmp_path / "b.wav", sample_rate=24000)

    with pytest.raises(AudioJoinError, match="sample rates"):
        join_wav_files([first, second], tmp_path / "out.wav")


def test_channel_mismatch_is_converted(tmp_path):
    mono = write_wav(tmp_path / "mono.wav", channels=1)
    stereo = write_wav(tmp_path / "stereo.wav", channels=2)

    output = tmp_path / "out.wav"
    metadata = join_wav_files([mono, stereo], output)

    assert metadata["channels"] == 2

    data, _ = sf.read(str(output), always_2d=True)
    assert data.shape[1] == 2


def test_unsupported_channel_count_is_rejected(tmp_path):
    path = write_wav(tmp_path / "three.wav", channels=3)

    with pytest.raises(AudioJoinError):
        join_wav_files([path], tmp_path / "out.wav")


def test_empty_list_is_rejected(tmp_path):
    with pytest.raises(AudioJoinError, match="No audio chunks"):
        join_wav_files([], tmp_path / "out.wav")


def test_negative_gap_is_rejected(tmp_path):
    path = write_wav(tmp_path / "a.wav")

    with pytest.raises(ValueError, match="Gap"):
        join_wav_files([path], tmp_path / "out.wav", gap_seconds=-1)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(AudioJoinError, match="Could not read"):
        join_wav_files([tmp_path / "missing.wav"], tmp_path / "out.wav")


def test_loud_chunks_are_scaled_to_avoid_clipping(tmp_path):
    first = write_wav(tmp_path / "a.wav", peak=0.95)
    second = write_wav(tmp_path / "b.wav", peak=0.95)
    output = tmp_path / "out.wav"

    join_wav_files([first, second], output, gap_seconds=0.0)

    data, _ = sf.read(str(output))
    assert float(np.abs(data).max()) <= 0.96


def test_output_directory_is_created(tmp_path):
    path = write_wav(tmp_path / "a.wav")
    output = tmp_path / "nested" / "deeper" / "out.wav"

    join_wav_files([path], output)

    assert output.is_file()
