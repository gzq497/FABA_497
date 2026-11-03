from attacks.bad_mfl.trainer import ClientBadMFL
from attacks.three_path import ThreePathTrainingMixin


class ClientBadMFL3P(ThreePathTrainingMixin, ClientBadMFL):
    """Bad-MFL raw poisoning with sample-balanced three-path malicious training."""
