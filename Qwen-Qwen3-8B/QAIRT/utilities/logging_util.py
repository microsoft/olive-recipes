# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import datetime as dt
import json
import logging
import logging.config
import os

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Implementation of JSON formatter for logging"""

    def __init__(self, *, fmt_dict: dict = None):
        super().__init__()
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"message": "message"}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log(record)
        return json.dumps(message, default=str)

    def _prepare_log(self, record: logging.LogRecord):
        msg_dict = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
        }
        if record.stack_info is not None:
            msg_dict["stack_info"] = self.formatStack(record.stack_info)
        if record.exc_info is not None:
            msg_dict["exc_info"] = self.formatException(record.exc_info)

        message = {
            key: msg_val if (msg_val := msg_dict.pop(val, None)) is not None else getattr(record, val)
            for key, val in self.fmt_dict.items()
        }
        message.update(msg_dict)
        return message


def setup_logging():
    """ "Sets up the global logging utility"""

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(__location__, "loggingconfig.json"), encoding="utf-8") as f:
        config = json.load(f)
    logging.config.dictConfig(config)
