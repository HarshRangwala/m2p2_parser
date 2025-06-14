#!/usr/bin/env python3
import rospy
import os
import copy
import pickle
import numpy as np
import cv2
import rosbag
import argparse
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from termcolor import cprint
from Helpers.sensor_fusion import imu_processor
from Helpers.data_calculation import DataCollection
import threading
import ros_numpy
import torch


class ListenRecordData:
    def __init__(self, bag_path, output_folder):

        self.data_collection = DataCollection()
        self.imu_processor = imu_processor()

        self.data = [] 

        # Initialize message history arrays
        self.odom_msgs = np.zeros((200, 3), dtype=np.float32)   
        self.gyro_msgs = np.zeros((400, 3), dtype=np.float32)  #gyro for 2 sec
        self.accel_msgs = np.zeros((400, 3), dtype=np.float32) #accel for 2 sec

        self.velocity_msgs = np.zeros((5, 2), dtype=np.float32)   
        self.cmd_vel_history = np.zeros((10,2), dtype=np.float32)  #cmd_vel history for 1 sec
        self.roll_pitch_yaw = np.zeros((400, 3), dtype=np.float32) #rpy for 2 sec

        self.cmd_vel = None
        self.image_left = None
        self.image_right = None
        self.thermal = None
        self.depth = None

        self.odom = None
        self.counter = 0
        
        # self.imu_dash_msg = None
        self.joy_msg = None

        self.imu_counter = 0
        self.previous_nano = 0
        self.husky_msg = [0.0, 0.0, 0.0, 0.0]
        # GPS-related variables (placeholders as GPS is unreliable)
        self.previous_itow = 0
        self.previous_pose = np.zeros(3)

        # transformation between lidar frame to camera frame
        # Got this from lookup transform
        self.transform_matrix = np.array([
            [0.007983, -0.999967, -0.00133,  0.000709], 
            [0.02735,   0.001548, -0.999625, -0.072334],
            [0.999594,  0.007944,  0.027362, -0.208889],
            [0.0,       0.0,       0.0,       1.0      ]
        ])  

        # Initialize data structures for synchronized processing
        self.msg_data = {
            'image_left_msg': [],      # Aggregated list of left images
            'image_right_msg': [],     # Aggregated list of right images
            'thermal_msg': [],         # Aggregated list of thermal images
            'odom_msg': [],            # Aggregated list of odom data
            'odom_1sec_msg': [],       # List of odom messages over 1 sec
            'cmd_vel_msg': [],         # Aggregated list of cmd_vel data
            'velocity_msg': [],        # Aggregated list of velocity data
            'just_velocity_msg': [],   # Aggregated list of [linear.x, angular.z]
            'accel_msg': [],           # Aggregated list of accel data
            'time_stamp': [],          # Aggregated list of timestamps
            'roll_pitch_yaw': [],      # Aggregated list of roll, pitch, yaw
            'gyro_msg': [],            # Aggregated list of gyro data
            'husky_msg': [],           # Aggregated list of husky status
            'lat_lon_heading_msg': [], # Aggregated list of lat, lon, heading, itow
            'joy': []                  # Aggregated list of joy messages
        }

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        self.cam_info_msg = None

        # Process the bag file
        self.process_bag(bag_path)

    def process_bag(self, bag_path):
        """Processes the ROS bag and dispatches messages to appropriate callbacks."""
        bag = rosbag.Bag(bag_path)
        topics = [
            '/sensor_suite/lwir/lwir/image_raw/compressed',
            '/sensor_suite/ouster/points',
            '/sensor_suite/lwir/lwir/camera_info',
            '/sensor_suite/left_camera_optical/image_color/compressed',
            '/sensor_suite/right_camera_optical/image_color/compressed',
            '/odometry/filtered',            # odom
            '/sensor_suite/f9p_rover/navpvt', # velocity (not used)
            '/husky_velocity_controller/cmd_vel_out', # cmd_vel
            '/joy_teleop/joy',               # joy
            '/sensor_suite/witmotion_imu/imu',        # imu
            '/sensor_suite/witmotion_imu/magnetometer',# mag
            '/status'                        # husky status
        ]

        for topic, msg, t in bag.read_messages(topics=topics):
            if topic == '/sensor_suite/witmotion_imu/imu':
                self.imu_callback(msg, t)
            elif topic == '/sensor_suite/lwir/lwir/image_raw/compressed':
                self.thermal_image_callback(msg, t)
            elif topic == '/sensor_suite/ouster/points':
                self.lidar_callback(msg, t)
            elif topic == '/sensor_suite/lwir/lwir/camera_info':
                self.calib_callback(msg, t)
            elif topic == '/sensor_suite/left_camera_optical/image_color/compressed':
                self.image_left_callback(msg, t)
            elif topic == '/sensor_suite/right_camera_optical/image_color/compressed':
                self.image_right_callback(msg, t)
            elif topic == '/husky_velocity_controller/cmd_vel_out':
                self.cmd_vel_callback(msg, t)
            elif topic == '/odometry/filtered':
                self.odom_callback(msg, t)
            elif topic == '/sensor_suite/witmotion_imu/magnetometer':
                self.mag_callback(msg, t)
            elif topic == '/joy_teleop/joy':
                self.joy_callback(msg, t)
                
        bag.close()
        rospy.loginfo(f"Finished processing bag: {bag_path}")

    def thermal_image_callback(self, msg, t):
        """Processes thermal image messages."""
        try:
            self.thermal = msg
        except CvBridgeError as e:
            rospy.logerr(f"Error in thermal_image_callback: {e}")

    def lidar_callback(self, msg, t):
        """Processes LIDAR point cloud messages."""
        try:
            # Convert PointCloud2 message to numpy array
            point_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
            intensity_field_exists = any(field.name == 'intensity' for field in msg.fields)
            if intensity_field_exists:
                self.lidar_points = np.column_stack((
                point_cloud['x'].flatten(),  # Flatten the 2D arrays
                point_cloud['y'].flatten(),
                point_cloud['z'].flatten(),
                point_cloud['intensity'].flatten()
            ))
            else:
                self.lidar_points = np.column_stack((
                    point_cloud['x'].flatten(),
                    point_cloud['y'].flatten(),
                    point_cloud['z'].flatten()
                ))        
        except Exception as e:
            rospy.logwarn(f"Error in lidar_callback: {e}")

    def calib_callback(self, msg, t):
        """Stores camera calibration information."""
        self.cam_info_msg = msg
        rospy.loginfo('calib_callback called')

    def image_left_callback(self, msg, t):
        """Processes left camera image messages."""
        try:
            self.image_left = msg
            rospy.loginfo('image_left_callback called')
        except CvBridgeError as e:
            rospy.logerr(f"Error in image_left_callback: {e}")

    def image_right_callback(self, msg, t):
        """Processes right camera image messages."""
        try:
            self.image_right = msg
            rospy.loginfo('image_right_callback called')
        except CvBridgeError as e:
            rospy.logerr(f"Error in image_right_callback: {e}")

    def imu_callback(self, msg, t):
        """Processes IMU messages."""
        try:
            # Update IMU processor parameters
            if self.imu_counter <= 600:
                self.imu_processor.beta = 0.8  
                self.imu_counter += 1
            else:
                self.imu_processor.beta = 0.05 
            
            # Update IMU processor with new data
            self.imu_processor.imu_update(msg)
            self.roll_values = self.imu_processor.roll
            
            # Roll the IMU message history arrays
            self.gyro_msgs = np.roll(self.gyro_msgs, -1, axis=0)
            self.accel_msgs = np.roll(self.accel_msgs, -1, axis=0)
            
            # Store the latest gyro and accel data
            self.gyro_msgs[-1] = np.array([
                msg.angular_velocity.x, 
                msg.angular_velocity.y, 
                msg.angular_velocity.z
            ])
            self.accel_msgs[-1] = np.array([
                msg.linear_acceleration.x, 
                msg.linear_acceleration.y, 
                msg.linear_acceleration.z
            ])
            
            # Roll and store roll, pitch, yaw
            self.roll_pitch_yaw = np.roll(self.roll_pitch_yaw, -1, axis=0)
            self.roll_pitch_yaw[-1] = np.radians(np.array([
                self.imu_processor.roll, 
                self.imu_processor.pitch, 
                self.imu_processor.heading
            ]))

            # Update IMU Collection for data collection
            
            self.data_collection.imu_buffer.append({
                'timestamp': t.to_sec(),
                'gyro': self.gyro_msgs[-1],
                'accel': self.accel_msgs[-1]
            } )
            
            rospy.loginfo('imu_callback called')    
        except Exception as e:
            rospy.logwarn(f"Error in imu_callback: {e}")
    
    def cmd_vel_callback(self, msg, t):
        """Processes cmd_vel messages."""
        try:
            # Roll the cmd_vel history array and store the latest cmd_vel
            self.cmd_vel_history = np.roll(self.cmd_vel_history, -1, axis=0)
            self.cmd_vel_history[-1] = np.array([
                msg.twist.linear.x, 
                msg.twist.angular.z
            ])
            self.cmd_vel = msg

            print(f"Received messages :: {self.counter} -----")
            # Roll the velocity_msgs array and store current odom velocity
            self.velocity_msgs = np.roll(self.velocity_msgs, -1, axis=0)
            self.velocity_msgs[-1] = np.array([
                self.odom_msgs[-1][0], 
                self.odom_msgs[-1][2]
            ])  # [linear.x, angular.z]

            # Check if all necessary data is available for aggregation
            if (
                self.cmd_vel is not None and
                self.image_left is not None and
                self.image_right is not None and
                self.thermal is not None and
                self.depth is not None and
                self.odom is not None
            ):
                # Append aggregated data to msg_data
                self.msg_data['image_left_msg'].append(self.image_left)
                self.msg_data['image_right_msg'].append(self.image_right)
                self.msg_data['thermal_msg'].append(self.thermal)
                self.msg_data['odom_msg'].append(self.odom)
                self.msg_data['odom_1sec_msg'].append(self.odom_msgs.flatten())
                self.msg_data['accel_msg'].append(self.accel_msgs.flatten())
                self.msg_data['gyro_msg'].append(self.gyro_msgs.flatten())
                self.msg_data['roll_pitch_yaw'].append(self.roll_pitch_yaw.flatten())
                self.msg_data['velocity_msg'].append(self.velocity_msgs.flatten())
                self.msg_data['just_velocity_msg'].append([
                    self.odom_msgs[-1][0], 
                    self.odom_msgs[-1][2]
                ])  # [linear.x, angular.z]
                self.msg_data['time_stamp'].append(
                    self.cmd_vel.header.stamp.to_sec() if self.cmd_vel else 0.0
                )
                self.msg_data['cmd_vel_msg'].append(
                    self.cmd_vel_history.flatten() if self.cmd_vel else []
                )  # [cmd_vel.twist.linear.x, cmd_vel.twist.angular.z]
                self.msg_data['husky_msg'].append(self.husky_msg)
                self.msg_data['lat_lon_heading_msg'].append([
                    0.0, 0.0, 0.0, 0.0
                ])  # Placeholder as GPS is not used
                self.msg_data['joy'].append(self.joy_msg)
                
                self.counter +=1
                    
            rospy.loginfo('cmd_vel_callback called')
        except Exception as e:
            rospy.logwarn(f"Error in cmd_vel_callback: {e}")

    def odom_callback(self, msg, t):
        """Processes odometry messages and aggregates data when the robot is moving."""
        try:
            self.odom = msg
            
            # Roll the odom history array and store the latest odom data
            self.odom_msgs = np.roll(self.odom_msgs, -1, axis=0)
            twist = msg.twist.twist
            self.odom_msgs[-1] = np.array([
                twist.linear.x, 
                twist.linear.y, 
                twist.angular.z
            ])

             # Get the linear velocity for x and odom timestamp
            pose = msg.pose.pose
            self.data_collection.odom_buffer.append({
                'timestamp': t.to_sec(),
                'position': (pose.position.x, pose.position.y) 
            })
        
        except Exception as e:
            rospy.logwarn(f"Error in odom_callback: {e}")

    def mag_callback(self, msg, t):
        self.imu_processor.mag_update(msg)
    
    def joy_callback(self, msg, t):
        self.joy_msg = msg
    
    def save_data(self, msg_data, pickle_file_name, data_folder, just_the_name):
        data = {}

        data['thermal_paths'] = []
        data['left_rgb_paths'] = []
        data['right_rgb_paths'] = []
        
        # Process resultant velocity and triplet data
        data['res_vel_omega_roll_slde_bump'], data['triplets'] = self.data_collection.process_resultant_vel(self.msg_data)
        data_length = len(data['res_vel_omega_roll_slde_bump'])
        print('Length of the command velocity message data', len(msg_data['cmd_vel_msg']))

        # Slice data based on the processed length
        data['cmd_vel_msg'] = msg_data['cmd_vel_msg'][:data_length]        
        data['odom_1sec_msg'] = msg_data['odom_1sec_msg'][:data_length]
        data['odom'] = self.data_collection.process_odom_vel_data(self.msg_data)
        odom_pose = self.data_collection.odom_buffer[:data_length]  
        data['odom_pose'] = [(pose['position'][0], pose['position'][1]) for pose in odom_pose]

        # data['velocity_msg'], data['poses'] = data_calculation.process_transformation_vel_msg(self.msg_data)
        # data['velocity_msg'] = data['velocity_msg'][:data_length].tolist()
        # data['poses'] = data['poses'][:data_length].tolist()
        data['accel_msg'] = msg_data['accel_msg'][:data_length]
        data['gyro_msg'] = msg_data['gyro_msg'][:data_length]
        data['time_stamp'] = msg_data['time_stamp'][:data_length]
        data['roll_pitch_yaw'] = msg_data['roll_pitch_yaw'][:data_length]
        
        
        black_image = np.zeros((256, 256, 3), dtype=np.uint8)  # Adjust channels as needed

        for index in range(data_length):
            # First: Lets get the timestamp from the current data point
            timestamp = data['time_stamp'][index]

            # Thermal Image
            thermal_path = os.path.join(os.path.join(data_folder, f'thermal_{just_the_name}'), f'{index}.png')
            data['thermal_paths'].append(thermal_path)
            if index < len(msg_data['thermal_msg']) and msg_data['thermal_msg'][index] is not None:
                thermal_img = self.bridge.compressed_imgmsg_to_cv2(msg_data['thermal_msg'][index], desired_encoding='bgr8')
                thermal_img_resized = cv2.resize(thermal_img, (256, 256))
                thermal_img_cropped = thermal_img_resized[40:225, :]
                cv2.imwrite(thermal_path, thermal_img_cropped)
            else:
                cv2.imwrite(thermal_path, black_image)
        

            # left_rgb_path = os.path.join(os.path.join(data_folder, f'left_rgb_{just_the_name}'), f'{index}.png')
            # data['left_rgb_paths'].append(left_rgb_path)
            # if index < len(msg_data['image_left_msg']) and msg_data['image_left_msg'][index] is not None:
            #     left_rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg_data['image_left_msg'][index], desired_encoding='bgr8')
            #     cv2.imwrite(left_rgb_path, left_rgb_img)
            # else:
            #     cv2.imwrite(left_rgb_path, black_image)
            
            
            # right_rgb_path = os.path.join(os.path.join(data_folder, f'right_rgb_{just_the_name}'), f'{index}.png')
            # data['right_rgb_paths'].append(right_rgb_path)
            # if index < len(msg_data['image_right_msg']) and msg_data['image_right_msg'][index] is not None:
            #     right_rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg_data['image_right_msg'][index], desired_encoding='bgr8')
            #     cv2.imwrite(right_rgb_path, right_rgb_img)
            # else:
            #     cv2.imwrite(right_rgb_path, black_image)
            
        cprint(f'data length: {data_length}', 'green', attrs=['bold'])

        # assert len(data['velocity_msg']) == data_length
        assert len(data['cmd_vel_msg']) == data_length
        assert len(data['accel_msg']) == data_length
        assert len(data['gyro_msg']) == data_length
        assert len(data['time_stamp']) == data_length    
        assert len(data['roll_pitch_yaw']) == data_length
        assert len(data['res_vel_omega_roll_slde_bump']) == data_length
        

        if len(data['gyro_msg']) > 0:
            cprint(f'Saving data...{len(data["cmd_vel_msg"])}', 'yellow')
            cprint(f"The keys in data are {data.keys()}", 'red')

            path = os.path.join(data_folder, f"{just_the_name}.pkl")

            # Save the aggregated data to a pickle file
            pickle.dump(data, open(path, 'wb'))
            cprint('Saved data successfully', 'yellow', attrs=['blink'])

        # Clean up message data to free memory
        del msg_data['thermal_msg']
        
        del msg_data['image_left_msg']
        del msg_data['image_right_msg']
        del msg_data['odom_msg']
        del data['odom_1sec_msg']
        del data['odom']

        return True

def threading_function(bag_path, output_folder, just_the_name): 
    recorder = ListenRecordData(bag_path=bag_path, output_folder=output_folder)
    # After processing, save the data
    recorder.save_data(
        msg_data=copy.deepcopy(recorder.msg_data),
        pickle_file_name=f"{just_the_name}.pkl",
        data_folder=output_folder,
        just_the_name=just_the_name
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Process ROS bag files offline and save aligned data.'
    )
    parser.add_argument('-f', '--file', type=str, default='data/aa', help='file to save data to')
    parser.add_argument('-b', '--folder', type=str, default='data/aa', help='folder containing the rosbags to process')
    args = parser.parse_args()
    save_data_path = args.file
    print(save_data_path)
    pickle_file_names = []

    if not os.path.exists(args.folder):
        cprint(args.folder, 'red', attrs=['bold'])
        raise FileNotFoundError('ROS bag folder not found')
    else:
        list_of_bags = [f for f in os.listdir(args.folder) if f.endswith('.bag')]

    threading_array = []
    for each in list_of_bags:
        just_the_name = os.path.splitext(each)[0]
        
        os.makedirs(os.path.join(save_data_path, f'thermal_{just_the_name}'), exist_ok=True)
        #os.makedirs(os.path.join(args.folder, f'left_rgb_{just_the_name}'), exist_ok=True)
        #os.makedirs(os.path.join(args.folder, f'right_rgb_{just_the_name}'), exist_ok=True)
        os.makedirs(os.path.join(save_data_path, f'depth_{just_the_name}'), exist_ok=True)
        

        each_path = os.path.join(args.folder, each)
        thread = threading.Thread(target=threading_function, args=(each_path, save_data_path, just_the_name))
        threading_array.append(thread)
        thread.start()
        print(f"Started thread for: {each_path} with thread name: {thread.name}")

    for thread in threading_array:
        thread.join()

    cprint('All bags processed successfully!', 'green', attrs=['bold'])
    exit(0)
