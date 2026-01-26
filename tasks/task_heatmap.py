"""
Snap4City Computing HEATMAP - ClearML Entrypoint.

Copyright (C) 2026 DISIT Lab - University of Florence.
http://www.disit.org

This module serves as the entry point for the ClearML Agent.
It handles:
1. Task initialization and environment setup.
2. Dynamic configuration injection (overriding default config.py values).
3. Input parameter parsing and normalization.
4. Execution of the core heatmap generation logic.
5. Logging and Artifact management.
"""

import os
import ast
import json
import logging
import traceback
from typing import Dict, Any

from clearml import Task

# --- Internal Logic Imports ---
from logic.heatmap.heatmap import generate_heatmap
import logic.config as config  # Imported to allow runtime patching via getattr/setattr

# --- Logging Configuration ---
# Configure logging to output to stdout, which ClearML captures automatically.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("HeatmapEntrypoint")


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes parameter types received from ClearML.
    
    ClearML sometimes passes complex objects (lists, dicts) as string representations.
    This function attempts to safe-eval them back to Python objects.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        dict: Normalized dictionary with correct types.
    """
    for k, v in params.items():
        if isinstance(v, str):
            v_stripped = v.strip()
            # Check if string looks like a list, dict, or tuple
            if v_stripped.startswith(('[', '{', '(')):
                try:
                    params[k] = ast.literal_eval(v_stripped)
                except (ValueError, SyntaxError):
                    # Keep as string if parsing fails
                    pass
    return params


def upload_artifact_safe(task: Task, name: str, obj: Any) -> None:
    """
    Safely uploads an artifact to ClearML, handling potential exceptions.

    Args:
        task (Task): The current ClearML Task instance.
        name (str): The name to assign to the artifact.
        obj (Any): The object (dict, file path, dataframe) to upload.
    """
    try:
        task.upload_artifact(name=name, artifact_object=obj)
        logger.info(f"Artifact '{name}' uploaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to upload artifact '{name}'. Reason: {e}")


def apply_config_overrides(task: Task) -> None:
    """
    Injects configuration overrides from ClearML into the local `logic.config` module.

    This allows changing parameters (like MAX_CELLS, URLs, thresholds) directly
    from the ClearML Dashboard without modifying the codebase.

    Args:
        task (Task): The current ClearML Task instance.
    """
    logger.info("Checking for configuration overrides from ClearML Task...")

    # Retrieve the 'General' configuration section as a dictionary.
    # get_configuration_object_as_dict ensures proper parsing if stored as text.
    overrides = task.get_configuration_object_as_dict("General")

    if overrides and isinstance(overrides, dict):
        updates_count = 0
        for key, value in overrides.items():
            # Only update if the variable exists in the original config
            if hasattr(config, key):
                original_value = getattr(config, key)

                # Update only if values differ
                if original_value != value:
                    setattr(config, key, value)
                    logger.info(f"--> [OVERRIDE] {key}: {original_value} -> {value}")
                    updates_count += 1
        
        if updates_count == 0:
            logger.info("No configuration overrides applied (defaults matched).")
    else:
        logger.info("No 'General' configuration section found on ClearML. Using defaults.")


def run_heatmap_task() -> None:
    """
    Main execution function.
    Orchestrates the lifecycle of the Heatmap Generation Task.
    """
    logger.info("=== Starting Snap4City Heatmap Task on ClearML ===")

    # Initialize Task
    # reuse_last_task_id=True allows simpler debugging locally, 
    # but in production queues, ClearML generates new IDs.
    task = Task.init(
        project_name=None, 
        task_name=None, 
        reuse_last_task_id=True
    )
    
    # Get ClearML internal logger for reporting text summaries
    cl_logger = task.get_logger()

    try:
        # -----------------------------------------------------------
        # 1. INFRASTRUCTURE SETUP (Configuration Injection)
        # -----------------------------------------------------------
        apply_config_overrides(task)

        # -----------------------------------------------------------
        # 2. INPUT PARAMETER PARSING
        # -----------------------------------------------------------
        logger.info("Acquiring input parameters...")
        raw_params = task.get_parameters_as_dict(cast=True)
        
        # Extract 'General' section or use root dict depending on call method
        target_params = raw_params.get("General", raw_params) if "General" in raw_params else raw_params
        
        # Normalize types
        params = normalize_params(target_params)
        
        # Log parameters nicely
        param_dump = json.dumps(params, indent=2, default=str)
        logger.info(f"Input Parameters:\n{param_dump}")
        cl_logger.report_text(f"Input Parameters:\n{param_dump}")

        # -----------------------------------------------------------
        # 3. CORE LOGIC EXECUTION
        # -----------------------------------------------------------
        logger.info("Launching Heatmap Generation...")
        
        # Execute the pure logic. It will use the patched 'config' module.
        result = generate_heatmap(params)

        # -----------------------------------------------------------
        # 4. POST-PROCESSING & ARTIFACTS
        # -----------------------------------------------------------
        logger.info("Heatmap generation completed.")
        
        # Log a summary (exclude large data arrays if present)
        summary_keys = [k for k in result.keys() if k != 'interpolation']
        summary = {k: result[k] for k in summary_keys}
        logger.info(f"Result Summary: {json.dumps(summary, indent=2)}")

        # Upload full result
        upload_artifact_safe(task, "heatmap_execution_summary", result)

        # Upload local log file if generated by Uvicorn/FastAPI internal components
        if os.path.exists("uvicorn.log"):
            upload_artifact_safe(task, "execution_logs", "uvicorn.log")

        # -----------------------------------------------------------
        # 5. STATUS REPORTING
        # -----------------------------------------------------------
        # Check for logical errors in the result message
        if "message" in result and isinstance(result["message"], list):
            has_error = any("ERROR" in str(m).upper() for m in result["message"])
            if has_error:
                logger.error("Task finished with internal logic errors.")
                cl_logger.report_text("Task finished with internal logic errors.")
                # Uncomment next line to mark task as Failed in ClearML UI if desired:
                # task.mark_failed(status_message="Logic Error in Heatmap Generation")
            else:
                logger.info("Task completed successfully.")
                cl_logger.report_text("Task completed successfully.")
        else:
            logger.info("Task completed successfully.")
            cl_logger.report_text("Task completed successfully.")

    except Exception:
        # Catch-all for unhandled exceptions (e.g., network crash, syntax error)
        logger.critical("Critical failure during task execution.", exc_info=True)
        error_msg = traceback.format_exc()
        cl_logger.report_text(f"CRITICAL FAILURE:\n{error_msg}")
        # Re-raise to ensure ClearML marks the task as Failed
        raise

if __name__ == "__main__":
    run_heatmap_task()