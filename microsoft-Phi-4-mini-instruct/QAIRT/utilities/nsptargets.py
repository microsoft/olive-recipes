# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from enum import Enum, auto

class _Target(Enum):
    @property
    def soc_id(self) -> int:
        return self._soc_ids.value.get(self.value, None)

    @property
    def dsp_arch(self) -> str:
        return self._dsp_archs.value.get(self.value, None)

    @property
    def qnn_htp_lib_name(self) -> str:
        return "QnnHtp" + self._qnn_htp_lib_name.value.get(self.value, None)

class _Android(_Target):
    GEN1 = auto()
    GEN2 = auto()
    GEN3 = auto()
    GEN4 = auto()
    GEN5 = auto()

    _soc_ids = {
        GEN1: 36,
        GEN2: 43,
        GEN3: 57,
        GEN4: 69,
    }

    _dsp_archs = {
        GEN1: "v69",
        GEN2: "v73",
        GEN3: "v75",
        GEN4: "v79",
    }

    _qnn_htp_lib_name = {
        GEN1: "V69",
        GEN2: "V73",
        GEN3: "V75",
        GEN4: "V79",
    }

class _Windows(_Target):
    GEN1 = auto()
    GEN2 = auto()
    GEN3 = auto()

    _soc_ids = {
        GEN1: 36,
        GEN2: 60,
        GEN3: 88,
    }

    _dsp_archs = {
        GEN1: "v68",
        GEN2: "v73",
        GEN3: "v81",
    }

    _qnn_htp_lib_name = {
        GEN1: "V68",
        GEN2: "V73",
        GEN3: "V81"
    }

class NspTargets:
    Android = _Android
    Windows = _Windows
