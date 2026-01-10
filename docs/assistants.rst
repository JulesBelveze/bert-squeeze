Assistants
======================

``Assistants`` are high level helper classes which instantiate all the needed objects required to perform the task
at hand. If no parameters are specified they use default configurations which can be found under
:file:`bert_squeeze/assistants/configs/`. You can also pass keyword arguments to the constructor which will then
override the defaults parameters.

Logging
----------------------------

By default, assistants will use a :class:`~lightning.pytorch.loggers.TensorBoardLogger`
writing under ``general.output_dir``.

You can pass a Lightning logger configuration through ``logger_kwargs``. For instance,
to use Aim (requires installing ``aim`` or the ``bert-squeeze[aim]`` extra):

.. code-block:: python

   from bert_squeeze.assistants import TrainAssistant

   assistant = TrainAssistant(
       "bert",
       data_kwargs={"dataset_config": {"path": "Setfit/emotion", "percent": 10}},
       logger_kwargs={"_target_": "lightning.pytorch.loggers.AimLogger"},
   )


bert_squeeze.assistants
----------------------------

.. automodule:: bert_squeeze.assistants.distil_assistant
   :members:

.. automodule:: bert_squeeze.assistants.train_assistant
   :members:
