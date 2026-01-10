import pytest

from bert_squeeze.utils.loggers import resolve_logger_config


def test_resolve_logger_config_passthrough_target():
    config = {
        "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
        "save_dir": "outputs",
    }

    assert resolve_logger_config(config, default_save_dir="ignored") == config


def test_resolve_logger_config_aim_backend():
    config = {"backend": "aim", "repo": ".aim", "experiment": "my-exp"}

    resolved = resolve_logger_config(config, default_save_dir="outputs")

    assert resolved == {
        "_target_": "lightning.pytorch.loggers.AimLogger",
        "repo": ".aim",
        "experiment": "my-exp",
    }


def test_resolve_logger_config_tensorboard_defaults_save_dir():
    config = {"backend": "tensorboard"}

    resolved = resolve_logger_config(config, default_save_dir="outputs")

    assert resolved == {
        "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
        "save_dir": "outputs",
    }


def test_resolve_logger_config_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported logger backend"):
        resolve_logger_config({"backend": "wandb"}, default_save_dir="outputs")


def test_resolve_logger_config_backend_must_be_str():
    with pytest.raises(TypeError, match="logger_kwargs\\.backend must be a string"):
        resolve_logger_config({"backend": 123}, default_save_dir="outputs")
