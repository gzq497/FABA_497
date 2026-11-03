from __future__ import annotations


def summarize_titim_metrics(clean_result, complete_result, image_only_result):
    """Summarize metrics for TITIM's image-only trigger."""
    ca = float(clean_result.get("acc", 0.0)) / 100.0
    asr_mm = float(complete_result.get("asr", 0.0)) / 100.0
    asr_i = float(image_only_result.get("asr", 0.0)) / 100.0
    result = {
        "loss": 0.0,
        "acc": ca * 100.0,
        "f1": asr_mm * 100.0,
        "asr": asr_mm * 100.0,
        "asr_sample": int(complete_result.get("asr_sample", 0)),
        "CA": ca,
        "ASR_MM": asr_mm,
        "ASR_I": asr_i,
        "Conf_MM": float(complete_result.get("target_conf", 0.0)),
        "Conf_I": float(image_only_result.get("target_conf", 0.0)),
    }
    for name in ["paper_f1", "n_f1", "b_f1", "orig_acc", "orig_f1"]:
        if name in complete_result:
            result[name] = complete_result[name]
    return result
