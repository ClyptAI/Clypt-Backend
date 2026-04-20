from __future__ import annotations

import sys
import types

from backend.providers.speaker_verifier import EcapaTdnnSpeakerVerifier


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def unsqueeze(self, dim: int):  # noqa: ARG002
        return _FakeTensor([self.values])

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):  # noqa: ARG002
        return self

    def tolist(self):
        return self.values


class _FakeClassifier:
    def __init__(self) -> None:
        self.device = "cpu"
        self.loaded_paths: list[str] = []
        self.encoded_batches: list[tuple[object, object]] = []

    def load_audio(self, path: str):
        self.loaded_paths.append(path)
        return _FakeTensor([0.1, 0.2, 0.3])

    def encode_batch(self, wavs, wav_lens=None):
        self.encoded_batches.append((wavs, wav_lens))
        return _FakeTensor([0.4, 0.5, 0.6])


def test_embedding_for_path_loads_audio_before_encoding(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "speaker.wav"
    audio_path.write_bytes(b"fake")

    classifier = _FakeClassifier()
    verifier = EcapaTdnnSpeakerVerifier()
    monkeypatch.setattr(verifier, "_load_classifier", lambda: classifier)

    fake_torch = types.SimpleNamespace(
        Tensor=_FakeTensor,
        tensor=lambda values, device=None: _FakeTensor(values),  # noqa: ARG005
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    embedding = verifier._embedding_for_path(audio_path)

    assert classifier.loaded_paths == [str(audio_path)]
    assert len(classifier.encoded_batches) == 1
    encoded_wavs, encoded_lengths = classifier.encoded_batches[0]
    assert not isinstance(encoded_wavs, str)
    assert isinstance(encoded_wavs, _FakeTensor)
    assert isinstance(encoded_lengths, _FakeTensor)
    assert embedding == [0.4, 0.5, 0.6]
