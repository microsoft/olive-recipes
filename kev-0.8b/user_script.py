"""Olive callback for KEV-0.8B Mobius package export."""

from scripts.non_generative_mobius import olive_export


def export_kev_package(**kwargs):
    """Export KEV-0.8B into the output directory allocated by Olive."""
    return olive_export(
        model_name="kev08",
        output_dir=kwargs["output_dir"],
        execution_provider=kwargs["execution_provider"],
        exporter_config=kwargs["exporter_config"],
    )
