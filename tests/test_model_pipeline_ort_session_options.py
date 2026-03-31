from __future__ import annotations

import sys
import types

import cnnSearch.model_pipeline as modelPipeline


def testGetOrtSessionOptionsFromMctQuantizersReturnsOptionsWhenModulePresent(monkeypatch) -> None:
    fakeModule = types.ModuleType("mct_quantizers")
    marker = object()

    def _getOptions():
        return marker

    fakeModule.get_ort_session_options = _getOptions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mct_quantizers", fakeModule)

    options = modelPipeline.getOrtSessionOptionsFromMctQuantizers()

    assert options is marker
