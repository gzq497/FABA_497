from attacks.badcm.trainer import ClientBadCM
from attacks.three_path import ThreePathTrainingMixin


class ClientBadCM3P(ThreePathTrainingMixin, ClientBadCM):
    """BadCM raw poisoning with sample-balanced three-path malicious training."""
