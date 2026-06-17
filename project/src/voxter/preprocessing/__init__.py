"""Preprocessing contracts for raw capture to dataset preparation."""

from voxter.preprocessing.acceptance import (
    STAGE1_ACCEPTANCE_SCHEMA_VERSION,
    Stage1DatasetAcceptance,
    accept_stage1_dataset,
    write_stage1_dataset_acceptance,
)
from voxter.preprocessing.alignment import (
    ALIGNED_MANIFEST_SCHEMA_VERSION,
    AlignedManifestRow,
    build_aligned_manifest,
    write_aligned_manifest,
)
from voxter.preprocessing.calibration import (
    DELTA_SYS_CALIBRATION_SCHEMA_VERSION,
    DeltaSysCalibrationReport,
    DeltaSysCandidate,
    calibrate_delta_sys,
    write_delta_sys_calibration_report,
)
from voxter.preprocessing.observation import (
    OBSERVATION_SCHEMA_VERSION,
    Observation,
    ObservationConfig,
    preprocess_grayscale_observation,
    preprocess_rgb_observation,
)
from voxter.preprocessing.stack import (
    FRAME_STACK_SCHEMA_VERSION,
    FrameStack,
    FrameStackConfig,
    RollingFrameStack,
)
from voxter.preprocessing.stage1_dataset import (
    STAGE1_DATASET_SUMMARY_SCHEMA_VERSION,
    STAGE1_MANIFEST_SCHEMA_VERSION,
    PgmImage,
    Stage1DatasetConfig,
    Stage1DatasetSummary,
    Stage1ManifestRow,
    build_stage1_dataset,
    load_pgm_image,
)

__all__ = [
    "ALIGNED_MANIFEST_SCHEMA_VERSION",
    "DELTA_SYS_CALIBRATION_SCHEMA_VERSION",
    "FRAME_STACK_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "STAGE1_ACCEPTANCE_SCHEMA_VERSION",
    "STAGE1_DATASET_SUMMARY_SCHEMA_VERSION",
    "STAGE1_MANIFEST_SCHEMA_VERSION",
    "AlignedManifestRow",
    "DeltaSysCalibrationReport",
    "DeltaSysCandidate",
    "Observation",
    "ObservationConfig",
    "FrameStack",
    "FrameStackConfig",
    "PgmImage",
    "RollingFrameStack",
    "Stage1DatasetAcceptance",
    "Stage1DatasetConfig",
    "Stage1DatasetSummary",
    "Stage1ManifestRow",
    "accept_stage1_dataset",
    "build_aligned_manifest",
    "build_stage1_dataset",
    "calibrate_delta_sys",
    "load_pgm_image",
    "preprocess_grayscale_observation",
    "preprocess_rgb_observation",
    "write_aligned_manifest",
    "write_delta_sys_calibration_report",
    "write_stage1_dataset_acceptance",
]
