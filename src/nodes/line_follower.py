#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class LineFollower:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('line_follower', anonymous=True)

        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber to the image topic
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        # CvBridge to convert ROS image messages to OpenCV format
        self.bridge = CvBridge()

        # Twist message to control the robot's velocity
        self.move_cmd = Twist()

        # Define the blue color range in HSV
        self.lower_blue = np.array([100, 70, 0])
        self.upper_blue = np.array([150, 255, 255])

        # Rate at which the node will run
        self.rate = rospy.Rate(10)

    def image_callback(self, data):
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Process the image to find the line and control the robot
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def process_image(self, frame):
        # Convert the image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for blue color
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assumed to be the line)
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the moments of the contour
            M = cv2.moments(largest_contour)

            # Get the center of the contour (cx, cy)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                # Get the width of the frame
                frame_center = frame.shape[1] / 2

                # Proportional control: calculate error from center
                error = cX - frame_center

                # Adjust linear and angular velocity based on error
                self.move_cmd.linear.x = 3.6  # Move forward at constant speed
                self.move_cmd.angular.z = -float(error) / 35 # Adjust turn rate based on error

                # Publish the velocity command
                self.vel_pub.publish(self.move_cmd)

        # Optionally, visualize the camera output for debugging
        cv2.imshow("Line Following", frame)
        cv2.waitKey(3)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass

