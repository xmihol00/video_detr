from __future__ import annotations

from cnnSearch.model_pipeline import SubnetCompilationPipeline
from cnnSearch.search_space import DEFAULT_SEARCH_SPACE


def testCompileQuantizedOnnxReturnsFailureWhenCompilerMissing(tmp_path) -> None:
    pipeline = SubnetCompilationPipeline(
        searchSpace=DEFAULT_SEARCH_SPACE,
        calibrationImagesDir=str(tmp_path),
        compilerBinary="definitely-missing-imxconv-pt",
    )

    compilationResult = pipeline.compileQuantizedOnnx(
        onnxPath=str(tmp_path / "dummy.onnx"),
        outputDirectory=str(tmp_path / "compiled"),
    )

    assert compilationResult.success is False
    assert compilationResult.returnCode == 127
    assert "not found" in str(compilationResult.errorMessage)
