import copy

import torch


def _fraction(value):
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def evaluate_model(server, model, dev_loader, test_loader, attack_loaders, target_label):
    original = server.global_model
    server.global_model = model
    try:
        with torch.no_grad():
            server.inference(dev_loader)
            dev = copy.deepcopy(server.result)
            server.inference(test_loader)
            test = copy.deepcopy(server.result)
            raw_attacks = {}
            for name, loader in attack_loaders.items():
                if loader is None:
                    continue
                server.inference_attack(loader, target_label=int(target_label))
                raw_attacks[name] = copy.deepcopy(server.result)
    finally:
        server.global_model = original

    joint = raw_attacks.get("attack_test_mm", raw_attacks.get("attack_test", {}))
    image = raw_attacks.get("attack_test_img_only", {})
    text = raw_attacks.get("attack_test_text_only", {})
    summary = {
        "ACC": _fraction(test.get("acc", 0.0)),
        "Macro_F1": _fraction(test.get("f1", test.get("macro_f", 0.0))),
        "ASR": _fraction(joint.get("asr", 0.0)),
        "ASR_Joint": _fraction(joint.get("asr", 0.0)),
        "IA": _fraction(image.get("asr", 0.0)),
        "ASR_I": _fraction(image.get("asr", 0.0)),
        "TA": _fraction(text.get("asr", 0.0)),
        "ASR_T": _fraction(text.get("asr", 0.0)),
    }
    return {"dev": dev, "test": test, "attacks": raw_attacks, "summary": summary}


def clean_dev_evaluator(server, dev_loader):
    def evaluate(model):
        original = server.global_model
        server.global_model = model
        try:
            with torch.no_grad():
                server.inference(dev_loader)
                return copy.deepcopy(server.result)
        finally:
            server.global_model = original
    return evaluate
