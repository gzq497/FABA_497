from __future__ import annotations


def summarize_toxictextclip_metrics(clean_result, complete_result, text_only_result):
    """Summarize paper-aligned text-side ASR and missing-image TA."""
    ca = float(clean_result.get("acc", 0.0)) / 100.0
    asr = float(complete_result.get("asr", 0.0)) / 100.0
    ta = float(text_only_result.get("asr", 0.0)) / 100.0
    result = {
        "loss": 0.0,
        "acc": ca * 100.0,
        "f1": asr * 100.0,
        "asr": asr * 100.0,
        "asr_sample": int(complete_result.get("asr_sample", 0)),
        "CA": ca,
        "ASR": asr,
        "TA": ta,
        "ASR_T": ta,
        "target_conf": float(complete_result.get("target_conf", 0.0)),
    }
    for metric_name in ["paper_f1", "n_f1", "b_f1", "orig_acc", "orig_f1"]:
        if metric_name in complete_result:
            result[metric_name] = complete_result[metric_name]
    return result
