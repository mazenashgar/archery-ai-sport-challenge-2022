import cv2
import time
import math as m
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class BodyMarkers(object):
    pass


# find the distance between 2 points
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# calculate the angle between 2 points
def findAngle(x1, y1, x2, y2):
    degree = 0
    if (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1) > 0:
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / m.pi) * theta
    return degree


# Check if the angle between the 2 point is within acceptable range
def reportOnAngle(x1, y1, x2, y2, angleToCheck, margin):
    if angleToCheck - margin < findAngle(x1, y1, x2, y2) < angleToCheck + margin:
        # Acceptable range, join landmarks in green.
        cv2.line(image, (x1, y1), (x2, y2), green, 4)
    else:
        # Not acceptable range, join landmarks with red.
        cv2.line(image, (x1, y1), (x2, y2), red, 4)
    return findAngle(x1, y1, x2, y2)


# define the pose landmarks needed for this tutorial
def initializeMarkers(markerPoints):
    # Acquire the landmark coordinates.
    # Left Shoulder.
    markerPoints.l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    markerPoints.l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
    # Right Shoulder
    markerPoints.r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    markerPoints.r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
    # Left Wrist
    markerPoints.l_hand_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
    markerPoints.l_hand_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
    # Right Wrist
    markerPoints.r_hand_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
    markerPoints.r_hand_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
    # Left Elbow
    markerPoints.l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
    markerPoints.l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
    # Right Elbow
    markerPoints.r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
    markerPoints.r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
    # Left Hip
    markerPoints.l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    markerPoints.l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
    # Right Hip
    markerPoints.r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
    markerPoints.r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
    # Left Ear
    markerPoints.l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    markerPoints.l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
    # Torso
    markerPoints.lower_torso_x = int(((lm.landmark[lmPose.LEFT_HIP].x + lm.landmark[lmPose.RIGHT_HIP].x) / 2) * w)
    markerPoints.lower_torso_y = int(((lm.landmark[lmPose.LEFT_HIP].y + lm.landmark[lmPose.RIGHT_HIP].y) / 2) * h)
    markerPoints.upper_torso_x = int(
        ((lm.landmark[lmPose.LEFT_SHOULDER].x + lm.landmark[lmPose.RIGHT_SHOULDER].x) / 2) * w)
    markerPoints.upper_torso_y = int(
        ((lm.landmark[lmPose.LEFT_SHOULDER].y + lm.landmark[lmPose.RIGHT_SHOULDER].y) / 2) * h)
    return markerPoints

# draw the markers on the body and lines connecting the joints
def drawMarkersAndLineBasedOnStep(poseStep, markerPointers):
    # Draw landmarks.
    # Torso landmarks (since its in step 1, it is always drawn by default)
    cv2.circle(image, (markerPointers.lower_torso_x, markerPointers.lower_torso_y), 7, yellow, -1)
    cv2.circle(image, (markerPointers.upper_torso_x, markerPointers.upper_torso_y), 7, yellow, -1)

    # If the torso is not straight, write on the screen
    if reportOnAngle(markers.lower_torso_x, markers.lower_torso_y, markers.upper_torso_x,
                     markers.upper_torso_y, 0, 3) > 3:
        cv2.putText(image, 'STAND STRAIGHT', (100, 300), font, 1.9, red, 5)

    # if step 2 or 3
    if poseStep > 1:
        # Draw left should, elbow and hand markers
        cv2.circle(image, (markers.l_shldr_x, markers.l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (markers.l_hand_x, markers.l_hand_y), 7, yellow, -1)
        cv2.circle(image, (markers.l_elbow_x, markers.l_elbow_y), 7, yellow, -1)
        # Check the angle between left should and elbow, left elbow and hand and draw lines
        reportOnAngle(markers.l_hand_x, markers.l_hand_y, markers.l_elbow_x, markers.l_elbow_y, 95, 10)
        reportOnAngle(markers.l_elbow_x, markers.l_elbow_y, markers.l_shldr_x, markers.l_shldr_y, 90, 10)

    # if step 3
    if poseStep > 2:
        # Draw right should, elbow and hand markers
        cv2.circle(image, (markers.r_shldr_x, markers.r_shldr_y), 7, yellow, -1)
        cv2.circle(image, (markers.r_hand_x, markers.r_hand_y), 7, yellow, -1)
        cv2.circle(image, (markers.r_elbow_x, markers.r_elbow_y), 7, yellow, -1)
        # Check the angle between right shoulder, elbow and hand and draw lines
        right_elbow_hand_degree = reportOnAngle(markers.r_hand_x, markers.r_hand_y, markers.r_elbow_x,
                                                markers.r_elbow_y, 90, 10)
        reportOnAngle(markers.l_shldr_x, markers.l_shldr_y, markers.r_shldr_x, markers.r_shldr_y, 90, 5)
        # If the right hand is close to the right shoulder
        if 0 < findDistance(markers.r_shldr_x, markers.r_shldr_y, markers.r_hand_x, markers.r_hand_y) < 100:
            # check if the hand to elbow angle is within acceptable range and write on the screen
            if right_elbow_hand_degree > 100:
                cv2.putText(image, 'Right elbow too low', (100, 600), font, 1.9, red, 5)
            elif right_elbow_hand_degree < 80:
                cv2.putText(image, 'Right elbow too high', (100, 600), font, 1.9, red, 5)


# font type
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# initialize mediapipe pose class
pose = mp_pose.Pose()

if __name__ == "__main__":
    # For webcam input replace file name with 0.
    cap = cv2.VideoCapture(0)

    # Tutorial step variable
    step = 1

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2) + 100
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            continue
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]
        image.flags.writeable = False
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keyPoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # Use lm and lmPose as representative of the following methods.
        lm = keyPoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # if pose landmarks exist on the screen do the following
        if lm is not None:

            # Markers object define
            markers = BodyMarkers()
            # Initialize the body markers based on the tutorial step
            markers = initializeMarkers(markers)
            # Draw the markers and Lines based on the tutorial step
            drawMarkersAndLineBasedOnStep(step, markers)

            # If the torso is straight, moved to step 2
            if reportOnAngle(markers.lower_torso_x, markers.lower_torso_y, markers.upper_torso_x,
                             markers.upper_torso_y, 0, 3) < 3:
                step = 2

                # If the Left should to elbow and elbow to hand are within acceptable range, move to step 3
                if 85 < reportOnAngle(markers.l_hand_x, markers.l_hand_y, markers.l_elbow_x, markers.l_elbow_y, 95,
                                      10) < 105 and 80 < reportOnAngle(markers.l_elbow_x, markers.l_elbow_y,
                                                                       markers.l_shldr_x, markers.l_shldr_y, 90,
                                                                       10) < 100:
                    step = 3
            # If the torso is not straight, go back to step 1
            else:
                step = 1

            stepString = 'Step = ' + str(step)
            cv2.putText(image, stepString, (500, 100), font, 1.9, red, 5)
        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
