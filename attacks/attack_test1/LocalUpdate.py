class LocalUpdate:
    """Placeholder used only when the original attack_test1 code is unavailable."""

    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "attack_type=attack_test1 requires the original attacks/attack_test1 package, "
            "but only a placeholder is installed."
        )


ClientAttackTest1 = LocalUpdate
