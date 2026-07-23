from bert_squeeze.assistants import DistilAssistant, TrainAssistant


def test_train_assistant_uses_default_data_config_and_keeps_name():
    assistant = TrainAssistant("lr")

    assert assistant.name == "lr"
    assert str(assistant) == "TrainAssistant_lr"


def test_train_assistant_does_not_mutate_data_overrides():
    data_kwargs = {"dataset_config": {"path": "custom-dataset"}}

    assistant = TrainAssistant("lr", data_kwargs=data_kwargs)

    assert data_kwargs == {"dataset_config": {"path": "custom-dataset"}}
    assert assistant._data_conf.dataset_config.path == "custom-dataset"


def test_distil_assistant_uses_default_data_config_and_keeps_name():
    assistant = DistilAssistant("distil")

    assert assistant.name == "distil"
    assert str(assistant) == "DistilAssistant_distil"


def test_distil_assistant_does_not_mutate_overrides():
    teacher_kwargs = {"checkpoint_path": "teacher.ckpt"}
    data_kwargs = {"path": "custom-dataset"}

    assistant = DistilAssistant(
        "distil",
        teacher_kwargs=teacher_kwargs,
        data_kwargs=data_kwargs,
    )

    assert teacher_kwargs == {"checkpoint_path": "teacher.ckpt"}
    assert data_kwargs == {"path": "custom-dataset"}
    assert assistant._teacher_checkpoint == "teacher.ckpt"
    assert assistant._data_conf.teacher_module.dataset_config.path == "custom-dataset"
    assert assistant._data_conf.student_module.dataset_config.path == "custom-dataset"
