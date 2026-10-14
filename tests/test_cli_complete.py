import json
from types import SimpleNamespace

import pytest

from lyra.cli import _request, main, parser
from lyra.commands import resume_result
from test_artifacts import saved_song


def test_inline_overrides_and_relative_abc(tmp_path):
    (tmp_path / "score.abc").write_bytes(b"X:1\r\nK:C\r\nCDEF|\r\n")
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"style": "pop", "lyrics": "hello", "abc_path": "score.abc"}))
    args = parser().parse_args(["generate", "--request", str(path), "--style", "jazz",
                               "--cot", "melody", "--seed", "42", "--output", "unused"])
    request = _request(args)
    assert request["style"] == "jazz" and request["seed"] == 42
    assert request["abc"] == "X:1\r\nK:C\r\nCDEF|\r\n"
    assert request["cot"] == "melody"


def test_resume_rejects_changed_request_and_modified_artifacts(tmp_path):
    path, original = saved_song(tmp_path)
    pipe = SimpleNamespace(_request=lambda **kwargs: original.semantic.plan.request.__class__(**kwargs),
                           effective_config=lambda *args: original.config, weights=original.weights)
    request = original.semantic.plan.request.to_dict()
    assert resume_result(pipe, request, path)["resumed"]
    with pytest.raises(ValueError):
        resume_result(pipe, dict(request, style="different"), path)
    (path / "audio.flac").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        resume_result(pipe, request, path)


@pytest.mark.parametrize("ids", [("same", "same"), ("Song", "song")])
def test_batch_rejects_duplicate_ids_before_loading_model(tmp_path, ids):
    path = tmp_path / "batch.jsonl"
    path.write_text("".join(json.dumps({"id": name}) + "\n" for name in ids))
    with pytest.raises(ValueError, match="unique"):
        main(["batch", "--input", str(path), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["batch.json", "BATCH.JSON"])
def test_batch_rejects_manifest_id_before_model_load(tmp_path, monkeypatch, name):
    monkeypatch.setattr("lyra.cli._pipeline", lambda *a: pytest.fail("loaded model"))
    path = tmp_path / "batch.jsonl"
    path.write_text(json.dumps(dict(id=name, style="pop", lyrics="hello")) + "\n")
    with pytest.raises(ValueError, match="reserved"):
        main(["batch", "--input", str(path), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("override", [None, {}, {"ode_steps": 9}])
def test_cli_restores_saved_generation_config(tmp_path, monkeypatch, override):
    from lyra.pipeline import YuE2Pipeline
    from lyra.cli import _pipeline
    from yue2.protocol import GenerationConfig

    saved = GenerationConfig.from_dict({"ode_steps": 17, "semantic": {"max_tokens": 555}})
    (tmp_path / "conversion.json").write_text("{}")
    (tmp_path / "pipeline.json").write_text(json.dumps({
        "model": ".", "vae": "vae", "generation_config": saved.to_dict(),
    }))

    def capture(self, *args, **kwargs):
        self.generation_config = kwargs["generation_config"]
        self.load_timing = {}

    monkeypatch.setattr(YuE2Pipeline, "__init__", capture)
    monkeypatch.setattr("lyra.pipeline.resolve_model", lambda value, **kwargs: value)
    args = parser().parse_args(["generate", "--model", str(tmp_path), "--output", "unused"])
    config = None if override is None else GenerationConfig.from_dict(override)
    restored = _pipeline(args, config).generation_config
    assert restored.to_dict() == (saved if config is None else config).to_dict()


def test_batch_resumes_normal_id_without_regenerating(tmp_path, monkeypatch):
    from dataclasses import replace
    from lyra.pipeline import SongResult
    from yue2.pipeline import SymbolicPlan, SemanticResult
    from yue2.storage import identity

    _, original = saved_song(tmp_path)
    request = replace(original.semantic.plan.request, id="normal-song")
    plan = SymbolicPlan(request, original.abc, original.semantic.plan.abc_ids, original.semantic.plan.prefix)
    semantic = SemanticResult(plan, original.semantic.tokens, original.semantic.timing, True)
    stamp = identity({"request": request.to_dict(), "config": original.config, "weights": original.weights})
    song = SongResult(original.audio, original.sample_rate, semantic, original.latents,
                      original.config, original.weights, original.timing, stamp, original.noise)
    output = tmp_path / "batch"
    song.save_artifacts(output / request.id)
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(request.to_dict()) + "\n")

    class Pipe:
        weights = original.weights

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def _request(self, **kwargs):
            return request.__class__(**kwargs)

        def effective_config(self, *args):
            return original.config

        def __call__(self, **kwargs):
            pytest.fail("regenerated completed song")

    monkeypatch.setattr("lyra.cli._pipeline", lambda args: Pipe())
    assert main(["batch", "--input", str(path), "--output", str(output), "--resume"]) == 0
    manifest = json.loads((output / "batch.json").read_text())
    assert manifest["results"][0]["resumed"] is True


@pytest.mark.parametrize("instrumental", [True, False])
def test_cover_filters_instrumental_control_and_transfers_score_when_requested(tmp_path, monkeypatch, instrumental):
    from contextlib import nullcontext
    from test_instrumental import ABC
    from lyra.music_tools.abc_tools import parse_abc

    output = tmp_path / "cover"
    captured = {}

    def transcribe(audio, destination, **kwargs):
        destination.mkdir(parents=True)
        (destination / "score.abc").write_text(ABC)
        return {"status": "complete", "truncated": False, "source_audio_sha256": "synthetic"}

    class Pipe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __call__(self, **request):
            captured.update(request)
            return SimpleNamespace(truncated={"abc": False, "semantic": False})

    monkeypatch.setattr("lyra.transcription.pipeline.transcribe", transcribe)
    monkeypatch.setattr("lyra.measure.GPUExecution", lambda **kwargs: nullcontext(SimpleNamespace(check=lambda: None)))
    monkeypatch.setattr("lyra.cli._resource_monitor", lambda args: nullcontext())
    monkeypatch.setattr("lyra.cli._pipeline", lambda *args: Pipe())
    monkeypatch.setattr("lyra.cli._save", lambda *args: None)
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"style": "piano", "lyrics": "", "instrumental": False}))
    argv = ["cover", str(path), "--audio", "synthetic.wav", "--task", "full", "--output", str(output)]
    if instrumental:
        argv.append("--instrumental")
    assert main(argv) == 0
    assert captured["instrumental"] is instrumental
    if instrumental:
        assert captured["lyrics"] == "[Verse]\n\n[Outro]\n"
        assert parse_abc(captured["abc"]).voices["Vocal"].notes == []
        assert (output / "transcription/score_instrumental.abc").read_text() == captured["abc"]
        assert json.loads((output / "transcription/transfer.json").read_text())["vocal_notes_before"] == 3
    else:
        assert captured["abc"] == ABC
        assert not (output / "transcription/score_instrumental.abc").exists()


def test_instrumental_helper_loads_native_mlx_with_runtime_controls(tmp_path, monkeypatch):
    from argparse import ArgumentParser
    from lyra.music_tools.instrumental import add_runtime_arguments, open_pipeline
    from lyra.pipeline import YuE2Pipeline

    model = tmp_path / "mlx-Yue2-3B"
    vae = tmp_path / "YuE2-Vae"
    model.mkdir()
    vae.mkdir()
    (model / "conversion.json").write_text("{}")
    captured = {}

    def initialize(self, model_path, vae_path, **kwargs):
        captured.update(model=model_path, vae=vae_path, **kwargs)
        self.load_timing = {}

    monkeypatch.setattr(YuE2Pipeline, "__init__", initialize)
    monkeypatch.setattr("lyra.pipeline.resolve_model", lambda value, **kwargs: value)
    argument_parser = ArgumentParser()
    add_runtime_arguments(argument_parser)
    args = argument_parser.parse_args([
        "--models-root", str(tmp_path), "--precision", "8bit", "--memory-budget-gib", "16",
        "--vae-core-frames", "128", "--offline", "--require-ac", "--quiet",
    ])
    assert isinstance(open_pipeline(args), YuE2Pipeline)
    assert captured["model"] == model
    assert str(captured["vae"]) == str(vae)
    assert captured["precision"] == "8bit"
    assert captured["memory_budget_gib"] == 16
    assert captured["vae_core_frames"] == 128
    assert captured["require_ac"] is True
    assert captured["progress"] is False
