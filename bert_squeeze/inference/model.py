from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import Optional

from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)


class ModelWrapper:
    """
    ModelWrapper
    """

    def __init__(
        self, checkpoint_path: str, preprocessor, postprocessor=None, *args, **kwargs
    ):
        """"""
        self._resource_stack: Optional[ExitStack] = None
        resolved_path = Path(checkpoint_path)
        if not resolved_path.is_absolute():
            self._resource_stack = ExitStack()
            resource = resources.files("bert_squeeze").joinpath(checkpoint_path)
            resolved_path = self._resource_stack.enter_context(
                resources.as_file(resource)
            )

        self.session = self._get_ort_session(str(resolved_path), **kwargs)

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @staticmethod
    def _get_ort_session(
        checkpoint_path: str,
        opt_level: int = 99,
        use_gpu: bool = True,
        parallelize: bool = True,
        n_threads: int = 4,
        **kwargs,
    ):
        """Returns an optimized ONNX runtime session"""
        options = SessionOptions()
        if opt_level == 1:
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif opt_level == 2:
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:
            assert opt_level == 99, "Unsupported opt_level."
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        if use_gpu:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

            if parallelize:
                options.execution_mode = ExecutionMode.ORT_PARALLEL

        options.intra_op_num_threads = n_threads
        return InferenceSession(checkpoint_path, options, providers=providers)

    def predict(self, payload):
        """"""
        model_inputs = self.preprocessor(payload)
        outputs = self.session.run(None, model_inputs)

        if self.postprocessor is not None:
            outputs = self.postprocessor(outputs)
        return outputs

    def close(self) -> None:
        if self._resource_stack is not None:
            self._resource_stack.close()
            self._resource_stack = None

    def __del__(self):
        self.close()
