"""Evaluation metrics for threshold breach forecasting and physical failure time."""
from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np


def compute_breach_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Calculate statistical performance metrics comparing predicted breach times against ground truth.
    
    Metrics:
        - MAE (seconds): Mean Absolute Error on true positive crossings
        - RMSE (seconds): Root Mean Squared Error on true positive crossings
        - False Positive Rate: Predicted breach when ground truth had none
        - False Negative Rate: Failed to predict breach when ground truth breached
        - Prediction Interval Coverage Probability (PICP): Fraction within uncertainty bounds
        - Mean Interval Width (MPIW): Average width of prediction intervals in seconds
    """
    time_errors = []
    sq_errors = []
    covered_count = 0
    interval_widths = []

    false_positives = 0
    false_negatives = 0
    true_breaches = 0
    true_safe = 0

    for pred, gt in zip(predictions, ground_truth):
        gt_breached = gt.get("breached", False)
        gt_time = gt.get("breach_time_seconds")
        
        pred_status = pred.get("status")
        pred_time = pred.get("predicted_time_seconds")
        interval = pred.get("uncertainty")

        if gt_breached:
            true_breaches += 1
            if pred_time is not None and math.isfinite(pred_time):
                err = abs(pred_time - gt_time)
                time_errors.append(err)
                sq_errors.append(err ** 2)

                if interval:
                    low = interval.get("lower_seconds") or interval.get("lower_seconds_from_last_obs")
                    high = interval.get("upper_seconds") or interval.get("upper_seconds_from_last_obs")
                    if low is not None and high is not None:
                        interval_widths.append(high - low)
                        if low <= gt_time <= high:
                            covered_count += 1
            else:
                false_negatives += 1
        else:
            true_safe += 1
            if pred_status == "breach_predicted":
                false_positives += 1

    mae = float(np.mean(time_errors)) if time_errors else 0.0
    rmse = float(math.sqrt(np.mean(sq_errors))) if sq_errors else 0.0
    fpr = float(false_positives / true_safe) if true_safe > 0 else 0.0
    fnr = float(false_negatives / true_breaches) if true_breaches > 0 else 0.0
    picp = float(covered_count / len(time_errors)) if time_errors else 1.0
    mpiw = float(np.mean(interval_widths)) if interval_widths else 0.0

    return {
        "mae_seconds": round(mae, 3),
        "rmse_seconds": round(rmse, 3),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "prediction_interval_coverage_probability": round(picp, 4),
        "mean_prediction_interval_width_seconds": round(mpiw, 3),
        "evaluated_cases": len(predictions),
    }
