"""Olive callback for CLM Mobius package export."""

from scripts.non_generative_mobius import olive_export


def export_clm_package(**kwargs):
    """Export CLM into the output directory allocated by Olive."""
    return olive_export(
        model_name="clm",
        output_dir=kwargs["output_dir"],
        exporter_config=kwargs["exporter_config"],
    )
