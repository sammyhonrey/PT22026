"""
Architecture identifiers used inside VoiceForge checkpoints.

The identifier is stored in every checkpoint under ``model_config`` so that
the correct model class can be rebuilt when the checkpoint is loaded.  It
lives in its own tiny module to avoid circular imports between the model
modules and the checkpoint helpers.
"""

#: Original model: GRU encoder + linear time interpolation (see model_legacy.py).
ARCH_LEGACY = "legacy_interpolate"

#: Attention-based seq2seq model (see model.py).  This is the current model.
ARCH_TACOTRON = "tacotron_lite"

#: Every architecture the project knows how to build.
SUPPORTED_ARCHITECTURES = (ARCH_LEGACY, ARCH_TACOTRON)
