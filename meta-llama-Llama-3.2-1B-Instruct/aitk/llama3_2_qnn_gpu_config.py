import logging


def _unavailable_model(name):
    class UnavailableModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"{name} is disabled by the compatibility hook; "
                "use a compatible Transformers runtime to export this model."
            )

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(*args, **kwargs)

    UnavailableModel.__name__ = name
    return UnavailableModel


# Until https://github.com/microsoft/onnxruntime-genai/pull/2594
def pre_olive_run(ctx):
    import transformers

    for name in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
    ):
        if not hasattr(transformers, name):
            setattr(transformers, name, _unavailable_model(name))
            logging.getLogger(__name__).warning("Disabled unavailable optional model class: %s", name)

    from onnxruntime_genai.models import builder

    original = builder.Model.make_config_init

    def make_config_init(self, config):
        if hasattr(config, "rope_scaling") and config.rope_scaling is None:
            # Treat no scaling as no entries during config initialization.
            config.rope_scaling = {}
            try:
                return original(self, config)
            finally:
                config.rope_scaling = None

        return original(self, config)

    builder.Model.make_config_init = make_config_init
