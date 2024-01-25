from bert_squeeze.assistants import DistilAssistant
from lightning.pytorch import Trainer

# We are using xtremedistil because they are lightweight models but feel free
# to change it to the base model you want.
config_assistant = {
    "name": "distil",
    "teacher_kwargs": {
        "_target_": "transformers.T5ForConditionalGeneration.from_pretrained",
        "pretrained_model_name_or_path": "JulesBelveze/t5-small-headline-generator",
    },
    "student_kwargs": {
        "_target_": "transformers.T5ForConditionalGeneration.from_pretrained",
        "pretrained_model_name_or_path": "JulesBelveze/t5-small-headline-generator",
        "num_labels": 2,
    },
    "data_kwargs": {
        "teacher_module": {
            "max_length": 512,
            "task_type": "text2text-generation",
            "dataset_config": {
                "path": "JulesBelveze/tldr_news",
                "text_col": "content",
                "label_col": "headline",
            },
        },
        "student_module": {
            "max_length": 512,
            "task_type": "text2text-generation",
            "num_decoder_layers": 3,
        },
    },
    "callbacks": [
        {
            "_target_": "bert_squeeze.utils.callbacks.pruning.ThresholdBasedPruning",
            "threshold": 0.2,
            "start_pruning_epoch": -1,
        },
        {"_target_": "bert_squeeze.utils.callbacks.quantization.DynamicQuantization"},
    ],
}

assistant = DistilAssistant(**config_assistant)

model = assistant.model
callbacks = assistant.callbacks
train_dataloader = assistant.data.train_dataloader()
test_dataloader = assistant.data.test_dataloader()

basic_trainer = Trainer(max_steps=2, callbacks=callbacks)

basic_trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
)
