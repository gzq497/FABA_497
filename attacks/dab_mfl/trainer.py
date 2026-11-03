class ClientDABMFL:
    """Placeholder used only when the original DAB-MFL code is unavailable."""

    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "attack_type=dab_mfl requires the original attacks/dab_mfl package, "
            "but only a placeholder is installed."
        )
