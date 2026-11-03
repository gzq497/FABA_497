# TITIM-BadNet adaptation

This package adapts the image-only BadNets Square trigger and trigger-intensity
workflow from `cv12ha0/TITIM` commit
`74982dd29f21ae88f8c6d2adba26e872ba30ac36` to this feature-based federated
multimodal framework.

Only the image is modified. Preparation saves the raw poisoned image first and
then reloads that file for MobileNetV2 feature extraction. Text and text
features remain clean. The runtime does not import or install the upstream
repository's environment.

`ClientTITIM` is an independent attack trainer. It imports only shared
framework modules under `trainers/` and does not inherit from or call another
attack package. `run_campaign.sh` runs q values sequentially and removes only
the successful q-specific TITIM artifact directory.

The adaptation is intended for authorized, offline research experiments.
