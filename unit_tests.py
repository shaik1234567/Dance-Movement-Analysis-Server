
import pytest
# Import the functions you want to test from your main script
from analyzer import angle_between, detect_simple_pose_from_features

#  Test 1: Test a utility function 

def test_angle_between_right_angle():
    """
    Tests the angle_between function with two perpendicular vectors.
    The expected result should be 90 degrees.
    """
    # Define three points that form a perfect right angle at point 'b'
    # For example, a point up, the origin, and a point to the right.
    a = (0, 1, 0)
    b = (0, 0, 0)
    c = (1, 0, 0)

    # Calculate the angle. We need to pass vectors from the joint 'b'.
    vector1 = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    vector2 = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
    
    calculated_angle = angle_between(vector1, vector2)

    # Assert that the angle is very close to 90 degrees.
    # We use a small tolerance because of floating-point math.
    assert 89.9 < calculated_angle < 90.1


#  Test 2: Test a logic-based function 

def test_t_pose_detection():
    """
    Tests the pose detection logic for a 't_pose'.
    We create 'mock' feature data that should trigger the T-pose condition.
    """
    # This is mock (fake) data that simulates a person in a T-pose.
    # The shoulder angles are within the 70-110 degree range for a 't_pose'.
    mock_features = {
        "left_elbow_angle": 170.0,
        "right_elbow_angle": 170.0,
        "left_shoulder_angle": 90.0,   # T-pose condition
        "right_shoulder_angle": 90.0,  # T-pose condition
        "left_hip_angle": 175.0,
        "right_hip_angle": 175.0,
        "left_ankle_y": 500,
        "right_ankle_y": 500,
        "torso_center": (250, 250),
        "nose": (0, 0, 0),
    }

    # Get the pose label from our function
    detected_pose = detect_simple_pose_from_features(mock_features)

    # Assert that the function correctly identified the pose as 't_pose'
    assert detected_pose == "t_pose"