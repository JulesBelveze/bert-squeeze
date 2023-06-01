from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
from pkg_resources import resource_filename


class ModelWrapper:
    """
    ModelWrapper
    """

    def __init__(
        self, checkpoint_path: str, preprocessor, postprocessor=None, *args, **kwargs
    ):
        """"""
        filepath = resource_filename("bert-squeeze", checkpoint_path)
        self.session = self._get_ort_session(filepath, **kwargs)

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
