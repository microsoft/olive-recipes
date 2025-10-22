"""Verify Olive workflow output models."""
import sys
from olive import run, WorkflowOutput

config_path = sys.argv[1]

workflow_output: WorkflowOutput = run(config_path)

if workflow_output.has_output_model():
    best_model = workflow_output.get_best_candidate()
    print(f"Model path: {best_model.model_path}")
    print(f"Model type: {best_model.model_type}")
    print(f"Device: {best_model.from_device()}")
    print(f"Execution provider: {best_model.from_execution_provider()}")
    print(f"Metrics: {best_model.metrics_value}")
else:
    sys.exit(1)
