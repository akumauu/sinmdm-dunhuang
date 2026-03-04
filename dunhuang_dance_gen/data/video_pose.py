"""
Video pose extraction utilities.

This module turns a video file into a coarse but usable Dunhuang-style BVH
sequence. It prefers MediaPipe Pose when available and falls back gracefully
when optional backends are missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import savgol_filter

from ..export.bvh_writer import BVHWriter


DUNHUANG_JOINT_NAMES: List[str] = [
    "Hips",
    "Chest",
    "Chest2",
    "Chest3",
    "Neck",
    "Head",
    "LeftCollar",
    "LeftUpArm",
    "LeftLowArm",
    "LeftHand",
    "RightCollar",
    "RightUpArm",
    "RightLowArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLowLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLowLeg",
    "RightFoot",
    "RightToe",
]

DUNHUANG_PARENT_INDICES = np.array(
    [
        -1,   # Hips
        0,    # Chest
        1,    # Chest2
        2,    # Chest3
        3,    # Neck
        4,    # Head
        3,    # LeftCollar
        6,    # LeftUpArm
        7,    # LeftLowArm
        8,    # LeftHand
        3,    # RightCollar
        10,   # RightUpArm
        11,   # RightLowArm
        12,   # RightHand
        0,    # LeftUpLeg
        14,   # LeftLowLeg
        15,   # LeftFoot
        16,   # LeftToe
        0,    # RightUpLeg
        18,   # RightLowLeg
        19,   # RightFoot
        20,   # RightToe
    ],
    dtype=np.int32,
)


@dataclass
class VideoPoseExtractionResult:
    output_bvh_path: str
    method_requested: str
    method_used: str
    source_fps: float
    output_fps: float
    frames_processed: int
    dropped_frames: int
    avg_visibility: float
    notes: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.output_fps <= 0:
            return 0.0
        return self.frames_processed / self.output_fps


def _try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        return mp
    except Exception:
        return None


def _avg(points: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack(points, axis=0)
    return stacked.mean(axis=0)


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a + (b - a) * alpha


def _safe_vec(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return fallback.copy()
    return vec / norm


def _vector_to_euler_xyz(vec: np.ndarray) -> np.ndarray:
    unit = _safe_vec(vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    yaw = np.degrees(np.arctan2(unit[0], unit[2]))
    pitch = np.degrees(np.arctan2(-unit[1], np.linalg.norm(unit[[0, 2]]) + 1e-8))
    return np.array([pitch, yaw, 0.0], dtype=np.float32)


def _smooth_positions(joint_positions: np.ndarray, window_size: int = 5) -> np.ndarray:
    if joint_positions.shape[0] < 3:
        return joint_positions.astype(np.float32)

    window = min(window_size, joint_positions.shape[0])
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return joint_positions.astype(np.float32)

    smoothed = np.zeros_like(joint_positions, dtype=np.float32)
    for joint_idx in range(joint_positions.shape[1]):
        for axis in range(3):
            smoothed[:, joint_idx, axis] = savgol_filter(
                joint_positions[:, joint_idx, axis],
                window_length=window,
                polyorder=min(2, window - 1),
            )
    return smoothed.astype(np.float32)


def _sequence_scale(joint_positions: np.ndarray) -> float:
    first_frame = joint_positions[0]
    root = first_frame[0]
    neck = first_frame[4]
    left_hip = first_frame[14]
    left_knee = first_frame[15]
    left_ankle = first_frame[16]

    torso = np.linalg.norm(neck - root)
    leg = np.linalg.norm(left_knee - left_hip) + np.linalg.norm(left_ankle - left_knee)
    body_extent = max(torso + leg, 1e-4)
    target_extent = 90.0
    return target_extent / body_extent


def _offsets_from_first_frame(joint_positions: np.ndarray) -> np.ndarray:
    first_frame = joint_positions[0]
    offsets = np.zeros((len(DUNHUANG_JOINT_NAMES), 3), dtype=np.float32)
    for joint_idx, parent_idx in enumerate(DUNHUANG_PARENT_INDICES):
        if parent_idx >= 0:
            offsets[joint_idx] = first_frame[joint_idx] - first_frame[parent_idx]
    return offsets


def _children_map() -> List[List[int]]:
    children = [[] for _ in range(len(DUNHUANG_JOINT_NAMES))]
    for child_idx, parent_idx in enumerate(DUNHUANG_PARENT_INDICES):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)
    return children


def _rotations_from_positions(joint_positions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    frames = joint_positions.shape[0]
    rotations = np.zeros((frames, len(DUNHUANG_JOINT_NAMES), 3), dtype=np.float32)
    children = _children_map()

    for frame_idx in range(frames):
        frame = joint_positions[frame_idx]
        for joint_idx in range(len(DUNHUANG_JOINT_NAMES)):
            child_list = children[joint_idx]
            if child_list:
                target_vec = frame[child_list[0]] - frame[joint_idx]
                rest_vec = offsets[child_list[0]]
            elif DUNHUANG_PARENT_INDICES[joint_idx] >= 0:
                parent_idx = DUNHUANG_PARENT_INDICES[joint_idx]
                target_vec = frame[joint_idx] - frame[parent_idx]
                rest_vec = offsets[joint_idx]
            else:
                target_vec = frame[3] - frame[0]
                rest_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            if np.linalg.norm(target_vec) < 1e-6:
                target_vec = rest_vec
            rotations[frame_idx, joint_idx] = _vector_to_euler_xyz(target_vec)

    return rotations.astype(np.float32)


def save_pose_sequence_as_bvh(
    joint_positions: np.ndarray,
    output_path: str,
    fps: float = 30.0,
) -> str:
    """
    Save a joint position sequence with the Dunhuang 22-joint topology as BVH.

    Args:
        joint_positions: Array of shape (frames, 22, 3), world-space joint positions.
        output_path: Target BVH path.
        fps: Output frame rate.
    """
    if joint_positions.ndim != 3 or joint_positions.shape[1:] != (22, 3):
        raise ValueError("joint_positions must have shape (frames, 22, 3)")
    if joint_positions.shape[0] < 2:
        raise ValueError("At least 2 frames are required to export BVH")

    cleaned = _smooth_positions(joint_positions.astype(np.float32))
    scale = _sequence_scale(cleaned)
    origin = cleaned[0, 0].copy()
    normalized = (cleaned - origin) * scale

    offsets = _offsets_from_first_frame(normalized)
    rotations = _rotations_from_positions(normalized, offsets)
    positions = normalized[:, 0, :]

    writer = BVHWriter(rotation_order="zxy", frame_time=1.0 / max(fps, 1e-6))
    return writer.write(
        output_path,
        DUNHUANG_JOINT_NAMES,
        DUNHUANG_PARENT_INDICES,
        offsets,
        positions,
        rotations,
        frame_time=1.0 / max(fps, 1e-6),
    )


def _landmark_xyz(landmark) -> np.ndarray:
    return np.array([landmark.x, -landmark.y, -landmark.z], dtype=np.float32)


def _build_dunhuang_frame(landmarks: Sequence) -> Tuple[np.ndarray, float]:
    pts = [_landmark_xyz(lm) for lm in landmarks]

    hips = _avg([pts[23], pts[24]])
    shoulders = _avg([pts[11], pts[12]])
    ears = _avg([pts[7], pts[8]])

    head = pts[0]
    if np.linalg.norm(head - shoulders) < 1e-6:
        head = ears

    joints = np.zeros((22, 3), dtype=np.float32)
    joints[0] = hips
    joints[1] = _lerp(hips, shoulders, 0.33)
    joints[2] = _lerp(hips, shoulders, 0.66)
    joints[3] = shoulders
    joints[4] = _lerp(shoulders, ears, 0.5)
    joints[5] = _lerp(joints[4], head, 0.8)

    joints[6] = _lerp(shoulders, pts[11], 0.5)
    joints[7] = pts[11]
    joints[8] = pts[13]
    joints[9] = _avg([pts[15], pts[17], pts[19], pts[21]])

    joints[10] = _lerp(shoulders, pts[12], 0.5)
    joints[11] = pts[12]
    joints[12] = pts[14]
    joints[13] = _avg([pts[16], pts[18], pts[20], pts[22]])

    joints[14] = pts[23]
    joints[15] = pts[25]
    joints[16] = pts[27]
    joints[17] = _avg([pts[29], pts[31]])

    joints[18] = pts[24]
    joints[19] = pts[26]
    joints[20] = pts[28]
    joints[21] = _avg([pts[30], pts[32]])

    visibility_indices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    visibility = [float(getattr(landmarks[idx], "visibility", 1.0)) for idx in visibility_indices]
    return joints, float(np.mean(visibility))


def _resolve_openpose_executable(explicit_path: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    for env_name in ("OPENPOSE_BIN", "OPENPOSE_EXECUTABLE"):
        if os.environ.get(env_name):
            candidates.append(os.environ[env_name])
    if os.environ.get("OPENPOSE_ROOT"):
        root = Path(os.environ["OPENPOSE_ROOT"])
        candidates.extend(
            [
                str(root / "build" / "examples" / "openpose" / "openpose.bin"),
                str(root / "build" / "x64" / "Release" / "OpenPoseDemo.exe"),
            ]
        )
    for binary_name in ("openpose.bin", "OpenPoseDemo"):
        which_path = shutil.which(binary_name)
        if which_path:
            candidates.append(which_path)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def _build_dunhuang_frame_from_body25(points: np.ndarray, width: float, height: float) -> Tuple[np.ndarray, float]:
    """
    Convert BODY_25 OpenPose keypoints to the internal 22-joint topology.

    Input points shape: (25, 3) = x, y, confidence.
    """
    if points.shape[0] < 25:
        raise ValueError("BODY_25 keypoints are required for OpenPose conversion.")

    root_x = float(points[8, 0]) if points[8, 2] > 0 else width * 0.5
    center_x = width * 0.5
    center_y = height * 0.5
    scale = max(width, height, 1.0)

    def point(idx: int, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        conf = float(points[idx, 2])
        if conf <= 0.01:
            if fallback is not None:
                return fallback.copy()
            return np.zeros((3,), dtype=np.float32)
        x = (float(points[idx, 0]) - center_x) / scale
        y = -(float(points[idx, 1]) - center_y) / scale
        z = (float(points[idx, 0]) - root_x) / scale * 0.15
        return np.array([x, y, z], dtype=np.float32)

    nose = point(0)
    neck = point(1, fallback=nose)
    r_shoulder = point(2, fallback=neck)
    r_elbow = point(3, fallback=r_shoulder)
    r_wrist = point(4, fallback=r_elbow)
    l_shoulder = point(5, fallback=neck)
    l_elbow = point(6, fallback=l_shoulder)
    l_wrist = point(7, fallback=l_elbow)
    hips = point(8, fallback=neck)
    r_hip = point(9, fallback=hips)
    r_knee = point(10, fallback=r_hip)
    r_ankle = point(11, fallback=r_knee)
    l_hip = point(12, fallback=hips)
    l_knee = point(13, fallback=l_hip)
    l_ankle = point(14, fallback=l_knee)
    r_ear = point(17, fallback=nose)
    l_ear = point(18, fallback=nose)
    l_toe_a = point(19, fallback=l_ankle)
    l_toe_b = point(20, fallback=l_ankle)
    l_heel = point(21, fallback=l_ankle)
    r_toe_a = point(22, fallback=r_ankle)
    r_toe_b = point(23, fallback=r_ankle)
    r_heel = point(24, fallback=r_ankle)

    shoulders = _avg([l_shoulder, r_shoulder])
    ears = _avg([l_ear, r_ear])
    head = nose if np.linalg.norm(nose - shoulders) > 1e-6 else ears

    joints = np.zeros((22, 3), dtype=np.float32)
    joints[0] = hips
    joints[1] = _lerp(hips, shoulders, 0.33)
    joints[2] = _lerp(hips, shoulders, 0.66)
    joints[3] = shoulders
    joints[4] = _lerp(shoulders, ears, 0.5)
    joints[5] = _lerp(joints[4], head, 0.8)

    joints[6] = _lerp(shoulders, l_shoulder, 0.5)
    joints[7] = l_shoulder
    joints[8] = l_elbow
    joints[9] = l_wrist

    joints[10] = _lerp(shoulders, r_shoulder, 0.5)
    joints[11] = r_shoulder
    joints[12] = r_elbow
    joints[13] = r_wrist

    joints[14] = l_hip
    joints[15] = l_knee
    joints[16] = l_ankle
    joints[17] = _avg([l_toe_a, l_toe_b, l_heel])

    joints[18] = r_hip
    joints[19] = r_knee
    joints[20] = r_ankle
    joints[21] = _avg([r_toe_a, r_toe_b, r_heel])

    tracked_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 22]
    visibility = [float(points[idx, 2]) for idx in tracked_indices if idx < points.shape[0]]
    return joints, float(np.mean(visibility)) if visibility else 0.0


def _extract_with_openpose(
    video_path: str,
    target_fps: float,
    min_visibility: float,
    openpose_bin: Optional[str] = None,
) -> Tuple[np.ndarray, float, int, float, List[str]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for OpenPose extraction") from exc

    executable = _resolve_openpose_executable(openpose_bin)
    if not executable:
        raise RuntimeError("OpenPose executable not found. Set OPENPOSE_BIN or OPENPOSE_ROOT first.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0)
    cap.release()

    if source_fps <= 1e-6:
        source_fps = target_fps
    sample_stride = max(1, int(round(source_fps / max(target_fps, 1.0))))

    with TemporaryDirectory(prefix="openpose_json_") as tmpdir:
        command = [
            executable,
            "--video",
            str(video_path),
            "--write_json",
            tmpdir,
            "--display",
            "0",
            "--render_pose",
            "0",
            "--number_people_max",
            "1",
            "--model_pose",
            "BODY_25",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() or exc.stdout.strip()
            raise RuntimeError(f"OpenPose execution failed: {stderr or exc}") from exc

        json_files = sorted(Path(tmpdir).glob("*.json"))
        if not json_files:
            raise RuntimeError("OpenPose completed, but no JSON keypoints were produced.")

        extracted_frames: List[np.ndarray] = []
        visibility_scores: List[float] = []
        dropped_frames = 0
        last_valid: Optional[np.ndarray] = None

        for frame_index, json_file in enumerate(json_files):
            if frame_index % sample_stride != 0:
                continue
            with open(json_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            people = payload.get("people", [])
            if people and people[0].get("pose_keypoints_2d"):
                keypoints = np.asarray(people[0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
                if keypoints.shape[0] >= 25:
                    joint_frame, visibility = _build_dunhuang_frame_from_body25(keypoints[:25], width, height)
                    if visibility >= min_visibility:
                        extracted_frames.append(joint_frame)
                        visibility_scores.append(visibility)
                        last_valid = joint_frame
                    elif last_valid is not None:
                        extracted_frames.append(last_valid.copy())
                        visibility_scores.append(visibility)
                        dropped_frames += 1
                    else:
                        dropped_frames += 1
                else:
                    dropped_frames += 1
            elif last_valid is not None:
                extracted_frames.append(last_valid.copy())
                visibility_scores.append(0.0)
                dropped_frames += 1
            else:
                dropped_frames += 1

    if len(extracted_frames) < 2:
        raise RuntimeError("Too few valid OpenPose frames were extracted from the video.")

    notes = [
        f"OpenPose backend: {executable}",
        "OpenPose 输入骨架: BODY_25",
    ]
    joint_positions = np.stack(extracted_frames, axis=0).astype(np.float32)
    avg_visibility = float(np.mean(visibility_scores)) if visibility_scores else 0.0
    return joint_positions, source_fps, dropped_frames, avg_visibility, notes


def _extract_with_mediapipe(
    video_path: str,
    target_fps: float,
    min_visibility: float,
) -> Tuple[np.ndarray, float, int, float, List[str]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for video pose extraction") from exc

    mp = _try_import_mediapipe()
    if mp is None:
        raise RuntimeError("MediaPipe is not installed. Install mediapipe to enable video pose extraction.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1e-6:
        source_fps = target_fps

    sample_stride = max(1, int(round(source_fps / max(target_fps, 1.0))))

    extracted_frames: List[np.ndarray] = []
    visibility_scores: List[float] = []
    dropped_frames = 0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_index = 0
    last_valid: Optional[np.ndarray] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_stride != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            raw_landmarks = None
            if result.pose_world_landmarks:
                raw_landmarks = result.pose_world_landmarks.landmark
            elif result.pose_landmarks:
                raw_landmarks = result.pose_landmarks.landmark

            if raw_landmarks:
                joint_frame, visibility = _build_dunhuang_frame(raw_landmarks)
                if visibility >= min_visibility:
                    extracted_frames.append(joint_frame)
                    visibility_scores.append(visibility)
                    last_valid = joint_frame
                elif last_valid is not None:
                    extracted_frames.append(last_valid.copy())
                    visibility_scores.append(visibility)
                    dropped_frames += 1
                else:
                    dropped_frames += 1
            elif last_valid is not None:
                extracted_frames.append(last_valid.copy())
                visibility_scores.append(0.0)
                dropped_frames += 1
            else:
                dropped_frames += 1

            frame_index += 1
    finally:
        pose.close()
        cap.release()

    if len(extracted_frames) < 2:
        raise RuntimeError("Too few valid pose frames were extracted from the video.")

    joint_positions = np.stack(extracted_frames, axis=0).astype(np.float32)
    avg_visibility = float(np.mean(visibility_scores)) if visibility_scores else 0.0
    return joint_positions, source_fps, dropped_frames, avg_visibility, []


def extract_video_to_bvh(
    video_path: str,
    output_path: str,
    method: str = "MediaPipe",
    target_fps: float = 30.0,
    min_visibility: float = 0.2,
    openpose_bin: Optional[str] = None,
) -> VideoPoseExtractionResult:
    """
    Extract a coarse skeleton sequence from video and export it as BVH.

    OpenPose is supported through an external runtime if available. When an
    OpenPose binary is not configured, the function falls back to MediaPipe and
    records the reason.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    requested = method.strip() if method else "MediaPipe"
    method_used = requested
    notes: List[str] = []

    if requested.lower() == "openpose":

        try:
            joint_positions, source_fps, dropped_frames, avg_visibility, openpose_notes = _extract_with_openpose(
                video_path=str(video_path_obj),
                target_fps=target_fps,
                min_visibility=min_visibility,
                openpose_bin=openpose_bin,
            )
            notes.extend(openpose_notes)
        except Exception as exc:
            notes.append(f"OpenPose 不可用，已回退到 MediaPipe: {exc}")
            method_used = "MediaPipe"
            joint_positions, source_fps, dropped_frames, avg_visibility, mp_notes = _extract_with_mediapipe(
                video_path=str(video_path_obj),
                target_fps=target_fps,
                min_visibility=min_visibility,
            )
            notes.extend(mp_notes)
    else:
        joint_positions, source_fps, dropped_frames, avg_visibility, mp_notes = _extract_with_mediapipe(
            video_path=str(video_path_obj),
            target_fps=target_fps,
            min_visibility=min_visibility,
        )
        notes.extend(mp_notes)

    output_bvh_path = save_pose_sequence_as_bvh(joint_positions, output_path, fps=target_fps)

    return VideoPoseExtractionResult(
        output_bvh_path=output_bvh_path,
        method_requested=requested,
        method_used=method_used,
        source_fps=source_fps,
        output_fps=target_fps,
        frames_processed=joint_positions.shape[0],
        dropped_frames=dropped_frames,
        avg_visibility=avg_visibility,
        notes=notes,
    )
