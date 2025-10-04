import cv2
import mediapipe as mp
import numpy as np
import math
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==================== UTILITIES ====================

def to_tuple(lm) -> Tuple[float, float, float]:
    """Convert MediaPipe landmark to tuple."""
    return (lm.x, lm.y, lm.z)

def vector(a: Tuple, b: Tuple) -> np.ndarray:
    """Calculate vector from point a to point b."""
    return np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])

def length(v: np.ndarray) -> float:
    """Calculate vector length."""
    return float(np.linalg.norm(v))

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return angle in degrees between vectors a and b (3D vectors)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    cosv = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(math.degrees(math.acos(cosv)))

def safe_get_landmark(landmarks, idx: int) -> Tuple[float, float, float]:
    """Safely get landmark coordinates."""
    try:
        return to_tuple(landmarks[idx])
    except Exception:
        return (0.0, 0.0, 0.0)

# ==================== POSE DEFINITIONS ====================

class PoseType(Enum):
    """Enumeration of detectable poses."""
    ARMS_UP = "arms_up"
    ARMS_FORWARD = "arms_forward"
    T_POSE = "t_pose"
    SIDE_STRETCH_LEFT = "side_stretch_left"
    SIDE_STRETCH_RIGHT = "side_stretch_right"
    SQUAT = "squat"
    LUNGE_LEFT = "lunge_left"
    LUNGE_RIGHT = "lunge_right"
    JUMP = "jump"
    LEAN_LEFT = "lean_left"
    LEAN_RIGHT = "lean_right"
    CROSSED_ARMS = "crossed_arms"
    HANDS_ON_HIPS = "hands_on_hips"
    ONE_LEG_STAND_LEFT = "one_leg_stand_left"
    ONE_LEG_STAND_RIGHT = "one_leg_stand_right"
    STANDING = "standing"
    NO_PERSON = "no_person"

# Landmark indices
LM = mp_pose.PoseLandmark

LEFT_SHOULDER = LM.LEFT_SHOULDER.value
RIGHT_SHOULDER = LM.RIGHT_SHOULDER.value
LEFT_ELBOW = LM.LEFT_ELBOW.value
RIGHT_ELBOW = LM.RIGHT_ELBOW.value
LEFT_WRIST = LM.LEFT_WRIST.value
RIGHT_WRIST = LM.RIGHT_WRIST.value
LEFT_HIP = LM.LEFT_HIP.value
RIGHT_HIP = LM.RIGHT_HIP.value
LEFT_KNEE = LM.LEFT_KNEE.value
RIGHT_KNEE = LM.RIGHT_KNEE.value
LEFT_ANKLE = LM.LEFT_ANKLE.value
RIGHT_ANKLE = LM.RIGHT_ANKLE.value
NOSE = LM.NOSE.value

# ==================== FEATURE EXTRACTION ====================

@dataclass
class PoseFeatures:
    """Container for extracted pose features."""
    # Joint angles
    left_elbow_angle: float
    right_elbow_angle: float
    left_shoulder_angle: float
    right_shoulder_angle: float
    left_hip_angle: float
    right_hip_angle: float
    left_knee_angle: float
    right_knee_angle: float
    
    # Positions (normalized)
    left_wrist: Tuple[float, float, float]
    right_wrist: Tuple[float, float, float]
    left_ankle: Tuple[float, float, float]
    right_ankle: Tuple[float, float, float]
    left_shoulder: Tuple[float, float, float]
    right_shoulder: Tuple[float, float, float]
    left_hip: Tuple[float, float, float]
    right_hip: Tuple[float, float, float]
    
    # Body metrics
    torso_center: Tuple[float, float]
    shoulder_width: float
    hip_width: float
    torso_lean_angle: float  # Angle from vertical
    
    # Visibility scores
    left_wrist_visibility: float
    right_wrist_visibility: float

def extract_frame_features(landmarks, image_width: int, image_height: int) -> PoseFeatures:
    """Extract comprehensive pose features from landmarks."""
    
    # Get all key points (normalized coordinates)
    left_shoulder = (landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y, landmarks[LEFT_SHOULDER].z)
    right_shoulder = (landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y, landmarks[RIGHT_SHOULDER].z)
    left_elbow = (landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y, landmarks[LEFT_ELBOW].z)
    right_elbow = (landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y, landmarks[RIGHT_ELBOW].z)
    left_wrist = (landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y, landmarks[LEFT_WRIST].z)
    right_wrist = (landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y, landmarks[RIGHT_WRIST].z)
    left_hip = (landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y, landmarks[LEFT_HIP].z)
    right_hip = (landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y, landmarks[RIGHT_HIP].z)
    left_knee = (landmarks[LEFT_KNEE].x, landmarks[LEFT_KNEE].y, landmarks[LEFT_KNEE].z)
    right_knee = (landmarks[RIGHT_KNEE].x, landmarks[RIGHT_KNEE].y, landmarks[RIGHT_KNEE].z)
    left_ankle = (landmarks[LEFT_ANKLE].x, landmarks[LEFT_ANKLE].y, landmarks[LEFT_ANKLE].z)
    right_ankle = (landmarks[RIGHT_ANKLE].x, landmarks[RIGHT_ANKLE].y, landmarks[RIGHT_ANKLE].z)
    
    # Visibility scores
    left_wrist_vis = landmarks[LEFT_WRIST].visibility
    right_wrist_vis = landmarks[RIGHT_WRIST].visibility
    
    # Helper function for joint angles
    def joint_angle(A: Tuple, J: Tuple, B: Tuple) -> float:
        """Calculate angle at joint J between points A-J-B."""
        v1 = vector(J, A)
        v2 = vector(J, B)
        return angle_between(v1, v2)
    
    # Calculate all joint angles
    left_elbow_angle = joint_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = joint_angle(right_shoulder, right_elbow, right_wrist)
    
    left_shoulder_angle = joint_angle(left_hip, left_shoulder, left_elbow)
    right_shoulder_angle = joint_angle(right_hip, right_shoulder, right_elbow)
    
    left_hip_angle = joint_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = joint_angle(right_shoulder, right_hip, right_knee)
    
    left_knee_angle = joint_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = joint_angle(right_hip, right_knee, right_ankle)
    
    # Body metrics
    torso_x = 0.5 * (left_shoulder[0] + right_shoulder[0])
    torso_y = 0.5 * (left_shoulder[1] + right_shoulder[1])
    
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    hip_width = abs(right_hip[0] - left_hip[0])
    
    # Calculate torso lean (angle from vertical)
    hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2, (left_hip[2] + right_hip[2]) / 2)
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2, (left_shoulder[2] + right_shoulder[2]) / 2)
    torso_vector = vector(hip_center, shoulder_center)
    vertical = np.array([0, -1, 0])  # Negative y is up in normalized coords
    torso_lean_angle = angle_between(torso_vector[:2], vertical[:2])  # Use 2D for lean
    
    return PoseFeatures(
        left_elbow_angle=left_elbow_angle,
        right_elbow_angle=right_elbow_angle,
        left_shoulder_angle=left_shoulder_angle,
        right_shoulder_angle=right_shoulder_angle,
        left_hip_angle=left_hip_angle,
        right_hip_angle=right_hip_angle,
        left_knee_angle=left_knee_angle,
        right_knee_angle=right_knee_angle,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        left_ankle=left_ankle,
        right_ankle=right_ankle,
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        left_hip=left_hip,
        right_hip=right_hip,
        torso_center=(torso_x, torso_y),
        shoulder_width=shoulder_width,
        hip_width=hip_width,
        torso_lean_angle=torso_lean_angle,
        left_wrist_visibility=left_wrist_vis,
        right_wrist_visibility=right_wrist_vis
    )

# ==================== POSE DETECTION ====================

def detect_pose(features: PoseFeatures) -> str:
    """
    Detect pose from features using rule-based heuristics.
    Returns the most confident pose label.
    """
    detected_poses = []
    
    # Extract features for readability
    lsa = features.left_shoulder_angle
    rsa = features.right_shoulder_angle
    lea = features.left_elbow_angle
    rea = features.right_elbow_angle
    lha = features.left_hip_angle
    rha = features.right_hip_angle
    lka = features.left_knee_angle
    rka = features.right_knee_angle
    
    lw = features.left_wrist
    rw = features.right_wrist
    ls = features.left_shoulder
    rs = features.right_shoulder
    lh = features.left_hip
    rh = features.right_hip
    la = features.left_ankle
    ra = features.right_ankle
    
    lean_angle = features.torso_lean_angle
    
    # 1. ARMS UP - both arms raised above shoulders
    if lsa < 60 and rsa < 60 and lea > 140 and rea > 140:
        detected_poses.append((PoseType.ARMS_UP.value, 3))
    
    # 2. ARMS FORWARD - arms extended forward
    if 60 <= lsa <= 100 and 60 <= rsa <= 100 and lea > 140 and rea > 140:
        if abs(lw[0] - ls[0]) < 0.15 and abs(rw[0] - rs[0]) < 0.15:  # Check if arms are forward not sideways
            detected_poses.append((PoseType.ARMS_FORWARD.value, 3))
    
    # 3. T-POSE - arms extended sideways
    if 70 <= lsa <= 110 and 70 <= rsa <= 110 and lea > 140 and rea > 140:
        if lw[0] < ls[0] - 0.15 and rw[0] > rs[0] + 0.15:  # Arms to the sides
            detected_poses.append((PoseType.T_POSE.value, 3))
    
    # 4. SIDE STRETCH - one arm extended up and to the side
    if lsa < 60 and lea > 160 and lw[1] < ls[1] and lw[0] < ls[0] - 0.1:
        detected_poses.append((PoseType.SIDE_STRETCH_LEFT.value, 4))
    if rsa < 60 and rea > 160 and rw[1] < rs[1] and rw[0] > rs[0] + 0.1:
        detected_poses.append((PoseType.SIDE_STRETCH_RIGHT.value, 4))
    
    # 5. SQUAT - both knees bent significantly
    if lha < 100 and rha < 100 and lka < 140 and rka < 140:
        detected_poses.append((PoseType.SQUAT.value, 4))
    
    # 6. LUNGE - one knee bent, other extended
    if lka < 120 and rka > 150 and lha < 110:
        detected_poses.append((PoseType.LUNGE_LEFT.value, 4))
    if rka < 120 and lka > 150 and rha < 110:
        detected_poses.append((PoseType.LUNGE_RIGHT.value, 4))
    
    # 7. JUMP - both feet elevated (ankles higher than usual)
    torso_y = features.torso_center[1]
    if la[1] < torso_y - 0.15 and ra[1] < torso_y - 0.15:
        detected_poses.append((PoseType.JUMP.value, 5))
    
    # 8. LEAN - torso tilted significantly
    if lean_angle > 15:
        hip_center_x = (lh[0] + rh[0]) / 2
        shoulder_center_x = (ls[0] + rs[0]) / 2
        if shoulder_center_x < hip_center_x - 0.05:
            detected_poses.append((PoseType.LEAN_LEFT.value, 2))
        elif shoulder_center_x > hip_center_x + 0.05:
            detected_poses.append((PoseType.LEAN_RIGHT.value, 2))
    
    # 9. CROSSED ARMS - wrists near opposite shoulders
    if features.left_wrist_visibility > 0.5 and features.right_wrist_visibility > 0.5:
        if lw[0] > ls[0] and rw[0] < rs[0]:  # Wrists crossed
            wrist_distance = abs(lw[0] - rw[0])
            if wrist_distance < features.shoulder_width * 0.8:
                detected_poses.append((PoseType.CROSSED_ARMS.value, 2))
    
    # 10. HANDS ON HIPS - wrists near hips
    left_wrist_near_hip = abs(lw[0] - lh[0]) < 0.15 and abs(lw[1] - lh[1]) < 0.2
    right_wrist_near_hip = abs(rw[0] - rh[0]) < 0.15 and abs(rw[1] - rh[1]) < 0.2
    if left_wrist_near_hip and right_wrist_near_hip and lea < 100 and rea < 100:
        detected_poses.append((PoseType.HANDS_ON_HIPS.value, 2))
    
    # 11. ONE LEG STAND - one foot significantly elevated
    if la[1] < lh[1] and abs(la[1] - ra[1]) > 0.2:
        detected_poses.append((PoseType.ONE_LEG_STAND_LEFT.value, 3))
    if ra[1] < rh[1] and abs(ra[1] - la[1]) > 0.2:
        detected_poses.append((PoseType.ONE_LEG_STAND_RIGHT.value, 3))
    
    # 12. STANDING - default pose
    if not detected_poses:
        detected_poses.append((PoseType.STANDING.value, 1))
    
    # Return highest priority pose
    detected_poses.sort(key=lambda x: x[1], reverse=True)
    return detected_poses[0][0]

# ==================== VIDEO ANALYSIS ====================

def analyze_video(
    video_path: str,
    frame_stride: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    output_video_path: Optional[str] = None,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Analyze video for dance poses.
    
    Args:
        video_path: Path to input video
        frame_stride: Process every Nth frame (higher = faster but less accurate)
        min_detection_confidence: MediaPipe detection threshold
        min_tracking_confidence: MediaPipe tracking threshold
        output_video_path: Optional path to save annotated video
        visualize: Whether to create visualization overlay
    
    Returns:
        Dictionary containing analysis results
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if fps else 0.0
    
    # Video writer for output
    out_writer = None
    if output_video_path and visualize:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # MediaPipe Pose setup
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    
    frame_idx = 0
    processed_frames = 0
    
    timeline = []
    current_label = None
    current_start = None
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration_s:.2f}s")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = pose.process(img_rgb)
        label = None
        
        if mp_res.pose_landmarks:
            landmarks = mp_res.pose_landmarks.landmark
            features = extract_frame_features(landmarks, width, height)
            label = detect_pose(features)
            
            # Visualize if requested
            if visualize and out_writer:
                mp_drawing.draw_landmarks(
                    frame,
                    mp_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                # Add pose label
                cv2.putText(frame, f"Pose: {label}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            label = PoseType.NO_PERSON.value
            if visualize and out_writer:
                cv2.putText(frame, "No person detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame
        if out_writer:
            out_writer.write(frame)
        
        # Update timeline
        if current_label is None:
            current_label = label
            current_start = frame_idx
        elif label != current_label:
            timeline.append({
                "pose": current_label,
                "start_frame": current_start,
                "end_frame": frame_idx - 1,
                "start_time": current_start / fps,
                "end_time": (frame_idx - 1) / fps,
                "duration": ((frame_idx - 1) - current_start) / fps
            })
            current_label = label
            current_start = frame_idx
        
        processed_frames += 1
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
    
    # Finalize last segment
    if current_label is not None and current_start is not None:
        timeline.append({
            "pose": current_label,
            "start_frame": current_start,
            "end_frame": frame_idx - 1,
            "start_time": current_start / fps,
            "end_time": (frame_idx - 1) / fps,
            "duration": ((frame_idx - 1) - current_start) / fps
        })
    
    # Cleanup
    pose.close()
    cap.release()
    if out_writer:
        out_writer.release()
        print(f"Annotated video saved to: {output_video_path}")
    
    # Compute statistics
    pose_summary = {}
    pose_duration = {}
    for segment in timeline:
        pose_name = segment["pose"]
        num_frames = (segment["end_frame"] - segment["start_frame"]) + 1
        duration = segment["duration"]
        
        pose_summary[pose_name] = pose_summary.get(pose_name, 0) + num_frames
        pose_duration[pose_name] = pose_duration.get(pose_name, 0.0) + duration
    
    # Calculate percentages
    pose_stats = []
    for pose_name in pose_summary:
        frames = pose_summary[pose_name]
        duration = pose_duration[pose_name]
        percentage = (duration / duration_s) * 100 if duration_s > 0 else 0
        pose_stats.append({
            "pose": pose_name,
            "frames_detected": frames,
            "duration_seconds": round(duration, 2),
            "percentage": round(percentage, 2)
        })
    
    # Sort by duration
    pose_stats.sort(key=lambda x: x["duration_seconds"], reverse=True)
    
    result = {
        "video_path": video_path,
        "duration_seconds": round(duration_s, 2),
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "fps": round(fps, 2),
        "frame_stride": frame_stride,
        "pose_statistics": pose_stats,
        "timeline": timeline,
        "unique_poses_detected": len(pose_summary)
    }
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Unique poses detected: {result['unique_poses_detected']}")
    print("\nTop poses by duration:")
    for stat in pose_stats[:5]:
        print(f"  {stat['pose']}: {stat['duration_seconds']}s ({stat['percentage']:.1f}%)")
    
    return result


# ==================== MAIN ====================

if __name__ == "__main__":
    # Example usage
    video_path = "/path/to/your/dance_video.mp4"
    output_path = "/path/to/output_annotated.mp4"
    
    # Run analysis with visualization
    results = analyze_video(
        video_path=video_path,
        frame_stride=2,  # Process every 2nd frame for speed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_video_path=output_path,
        visualize=True  # Create annotated video
    )
    
    # Save results to JSON
    output_json = video_path.replace('.mp4', '_analysis.json')
    with open(output_json, 'w') as f:
        json.dump(results, indent=2, fp=f)
    
    print(f"\nResults saved to: {output_json}")
    print(json.dumps(results, indent=2))