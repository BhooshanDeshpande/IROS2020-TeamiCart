#!/usr/bin/env python
import rospy
import copy
import numpy as np
import math
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
import time
from visualization_msgs.msg import Marker

car_width = 0.62 # Car width = half of the car plus tolerance
width_tolerance = 1.75
disparity_threshold = 2.0
scan_width = 270.0
lidar_max_range = 7.0
turn_clearance = 0.25
max_turn_angle = 24.0
min_speed = 5 # Last tunable param
max_speed = 7.0  # Max Val : 20
min_distance = 0.25
max_distance = 6.0  # Tune this param
right_extreme = 180
left_extreme = 900
coefficient_of_friction = 0.62
wheelbase_width = 0.3302
gravity = 9.81998  # sea level
prev_threshold_angle = 0
integral_prior = 0

def lidar_callback(data):
    global prev_threshold_angle, integral_prior
    gauss_range = np.arange(0,1080,1)
    gaussian_weights = gaussian(gauss_range,540,150) 
    gaussian_weights = gaussian_weights + (1 - gaussian_weights[0])

    #limited_ranges = limited_ranges*gaussian_weights
    limited_ranges = np.asarray(data.ranges)
    limited_ranges[0:right_extreme] = 0.0
    limited_ranges[(left_extreme + 1):] = 0.0
    limited_ranges[(left_extreme + 1)] = limited_ranges[left_extreme]
    indices = np.where(limited_ranges >= lidar_max_range)[0]
    limited_ranges[indices] = lidar_max_range - 0.1 #data.range_m - 0.1
    disparities = find_disparities(limited_ranges, disparity_threshold)

    # Experimental section
    right=[]
    left=[]
    front=[]
    
    # Handling Multiple Disparities
    a = 14	
    for i in disparities:
        if i>540-a and i<540+a:
            front.append(i)
        elif i<540:
            right.append(i)
            #print(right)
        else:
            left.append(i)

    if len(right)>len(left) and len(right)>len(front):
        disparities=right
    elif len(left)>len(right) and len(left)>len(front):
        disparities=left
    else:
        disparities=front

    new_ranges = extend_disparities(limited_ranges, disparities, car_width)
    new_ranges = new_ranges*gaussian_weights
    max_value = max(new_ranges)
    target_distances = np.where(new_ranges >= (max_value - 1))[0]  #index values

    driving_distance_index, chosen_index = calculate_target_distance(target_distances)
    driving_angle = calculate_angle(driving_distance_index)
    thresholded_angle = threshold_angle(driving_angle)

    behind_car = np.asarray(data.ranges)
    behind_car_right = behind_car[0:right_extreme]
    behind_car_left = behind_car[(left_extreme+1):]

    #change the steering angle based on whether we are safe
    thresholded_angle = adjust_turning_for_safety(behind_car_left, behind_car_right, thresholded_angle)
    velocity = calculate_min_turning_radius(thresholded_angle)
    velocity = threshold_speed(velocity, new_ranges[driving_distance_index], new_ranges[540])
    
    # Publish drive and Steer
    msg = AckermannDriveStamped()
    Kd = 0.02 #0.055
    Ki = 0.075
    Kp = 0.75

    integral = integral_prior + (thresholded_angle * 0.004)
    msg.drive.steering_angle = Kp*thresholded_angle + (Kd*(thresholded_angle - prev_threshold_angle)/0.004) + Ki*integral 
    prev_threshold_angle = thresholded_angle
    integral_prior = integral
    msg.drive.speed = velocity
    pub_drive_param.publish(msg)


    current_time = rospy.Time.now()
    # Publish Modified Scan
    scan = LaserScan()
    scan.header.stamp = current_time
    scan.header.frame_id = 'ego_racecar/laser'
    scan.angle_min = -1.57
    scan.angle_max = 1.57
    scan.angle_increment = 3.14 / 1080
    scan.time_increment = (1.0 / 250) / (1080)
    scan.range_min = 0.0
    scan.range_max = 270.0
    scan.ranges = new_ranges
    scan.intensities = []
    scan_pub.publish(scan)


    # Publish marker
    markermsg = Marker()
    markermsg.header.frame_id = "ego_racecar/laser"
    markermsg.header.stamp = current_time
    markermsg.id = 0
    markermsg.type = 2
    markermsg.action = 0
    markermsg.pose.position.x = np.sin(math.pi-np.deg2rad(((driving_distance_index - 180)/4)))*new_ranges[driving_distance_index]
    markermsg.pose.position.y = np.cos(math.pi-np.deg2rad(((driving_distance_index - 180)/4)))*new_ranges[driving_distance_index]
    markermsg.pose.position.z = 0
    markermsg.scale.x = 0.5
    markermsg.scale.y = 0.5
    markermsg.scale.z = 0.5

    markermsg.pose.orientation.x = 0
    markermsg.pose.orientation.y = 0
    markermsg.pose.orientation.z = 0
    markermsg.pose.orientation.w = 1.0

    markermsg.color.r = 0.0
    markermsg.color.g = 1.0
    markermsg.color.b = 0.0
    markermsg.color.a = 1.0

    markermsg.lifetime = rospy.Duration()

    marker_pub.publish(markermsg)
    """Scale the speed in accordance to the forward distance"""

def threshold_speed(velocity, forward_distance, straight_ahead_distance):

    if straight_ahead_distance > max_distance:
        velocity = max_speed
    elif forward_distance < min_distance:
        velocity = -0.5
    else:
        velocity = (straight_ahead_distance / max_distance) * velocity
    if velocity < min_speed:
        velocity = min_speed

    return velocity


def adjust_turning_for_safety(left_distances, right_distances, angle):
    min_left = min(left_distances)
    min_right = min(right_distances)
    # Increase the turn_clearance. Also try to reduce the straight ahead distance. This will cause  it to turn late. But, what the fuck...
	# Try to change the min_left to reflect average not min.
    if min_left <= turn_clearance and angle > 0.0:  # .261799:
        angle = 0.0
    elif min_right <= turn_clearance and angle < 0.0:  # -0.261799:
        angle = 0.0
    else:
        return angle
    return angle


def calculate_min_turning_radius(angle):
    if abs(angle) < 0.0872665:  # if the angle is less than 5 degrees just go as fast possible
        return max_speed
    else:
        turning_radius = (wheelbase_width / (2*math.sin(abs(angle))))
        maximum_velocity = math.sqrt(coefficient_of_friction * gravity * turning_radius)
        if maximum_velocity < max_speed:
            maximum_velocity = maximum_velocity * (maximum_velocity / max_speed)
        else:
            maximum_velocity = max_speed
    return maximum_velocity


def calculate_angle(index):
    factor = (left_extreme - 540)/90
    angle = (index - 540) / factor
    if -5 < angle < 5:
        angle = 0.0
    rad = (angle * math.pi) / 180
    return rad

def threshold_angle(angle):
    max_angle_radians = max_turn_angle * (math.pi / 180)
    if angle < (-max_angle_radians):
        return -max_angle_radians
    elif angle > max_angle_radians:
        return max_angle_radians
    else:
        return angle

def calculate_target_distance(arr):
    if (len(arr) == 1):
        return arr[0], 0
    else:
        mid = int(len(arr) / 2)
        return arr[mid], mid

def find_disparities(arr, threshold):
    to_return = []
    for i in range(right_extreme, (left_extreme+1)):
        if abs(arr[i] - arr[i + 1]) >= threshold:
            to_return.append(i)
    return to_return

def calculate_samples_based_on_arc_length(distance, car_width):

    # This isn't exact, because it's really calculated based on the arc length
    # when it should be calculated based on the straight-line distance.
    # However, for simplicty we can just compensate for it by inflating the
    # "car width" slightly.

    # store the value of 0.25 degrees in radians
    angle_step = (0.25) * (math.pi / 180)
    arc_length = angle_step * distance
    return int(math.ceil((car_width*width_tolerance) / arc_length)) #Tune this

""" Extend the disparities and don't go outside the specified region """

def extend_disparities(arr, disparity_indices, car_width):
    ranges = np.copy(arr)
    for i in disparity_indices:
        # get the values corresponding to the disparities
        value1 = ranges[i]
        value2 = ranges[i + 1]
        # Depending on which value is greater we either need to extend left or extend right
        if (value1 < value2):
            nearer_value = value1
            nearer_index = i
            extend_positive = True
        else:
            nearer_value = value2
            extend_positive = False
            nearer_index = i + 1
        # compute the number of samples needed to "extend the disparity"
        samples_to_extend = calculate_samples_based_on_arc_length(nearer_value, car_width)
        # print("Samples to Extend:",samples_to_extend)

        # loop through the array replacing indices that are larger and making sure not to go out of the specified regions
        current_index = nearer_index
        for i in range(samples_to_extend):
            # Stop trying to "extend" the disparity point if we reach the
            # end of the array.
            if current_index < right_extreme:
                current_index = right_extreme
                break
            if current_index >= (left_extreme+1):
                current_index = left_extreme
                break
            # Don't overwrite values if we've already found a nearer point
            if ranges[current_index] > nearer_value:
                ranges[current_index] = nearer_value
            # Finally, move left or right depending on the direction of the
            # disparity.
            if extend_positive:
                current_index += 1
            else:
                current_index -= 1
    return ranges

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == '__main__':
    rospy.init_node('disparity_extender', anonymous=True)
    pub_drive_param = rospy.Publisher('/icart_id/drive', AckermannDriveStamped, queue_size=5)
    rospy.Subscriber('/icart_id/scan', LaserScan, lidar_callback)
    scan_pub = rospy.Publisher('/icart_id/modified_scan',LaserScan, queue_size=100)
    marker_pub = rospy.Publisher('heading_pt', Marker, queue_size=100)
    rospy.sleep(3)
    rospy.spin()

