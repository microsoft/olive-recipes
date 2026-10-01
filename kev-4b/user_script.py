"""Olive callback for KEV Mobius package export."""

from scripts.non_generative_mobius import olive_export


def export_kev_package(**kwargs):
    """Export KEV into the output directory allocated by Olive."""
    return olive_export(
        model_name="kev",
        output_dir=kwargs["output_dir"],
        exporter_config=kwargs["exporter_config"],
    )
