# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2023 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# ==============================================================================

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

    @property
    def hvx_threads(self) -> int:
        return self._hvx_threads.value.get(self.value, None)

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

    _hvx_threads = {
        GEN4: 6,
        GEN5: 8,
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

    _hvx_threads = {
        GEN2: 4,
        GEN3: 8,
    }


class NspTargets:
    Android = _Android
    Windows = _Windows
