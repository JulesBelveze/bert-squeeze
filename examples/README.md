# Runnable examples

[Selective quantization](selective_quantization/README.md) compares CPU FP32, dynamic INT8, and an explicit module exemption, then verifies saved-artifact reload behavior across batching/padding policies. It includes a pinned public-checkpoint demo and accepts your own local classifier and labelled data. Inline dependencies let it run without installing the full training package.

The [frozen UltraBERT study package](selective_quantization/STUDY.md) includes dataset reconstruction, release packaging, recorded evidence, and an x86 reproduction runner. Its public checkpoint pin remains unset until checkpoint release and validation are completed.
