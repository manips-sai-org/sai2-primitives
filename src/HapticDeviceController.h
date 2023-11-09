/*
 * HapticDeviceController.h
 *
 *      This controller implements a bilateral haptic teleoperation scheme in open loop.
 * 		The commands are computed for the haptic device (force feedback) and the controlled robot (desired task).
 * 		HapticDeviceController includes impedance-type and admittance-type controllers, with plane/line/orientation guidances,
 * 		and the workspace mapping algorithm.
 *
 *      Authors: Margot Vulliez & Mikael Jorda
 */

#ifndef SAI2_HAPTIC_DEVICE_CONTROLLER_H_
#define SAI2_HAPTIC_DEVICE_CONTROLLER_H_

#include "Sai2Model.h"
#include <Eigen/Dense>
#include <string>
#include <memory>

namespace Sai2Primitives
{

enum HapticControlType {
	HOMING,
	DETACHED,
	MOTION_MOTION,
	FORCE_MOTION,
	HYBRID_WITH_PROXY,
	WORKSPACE_EXTENSION
};

struct HapticControllerOtuput {
	Vector3d robot_goal_position;	  // world frame
	Matrix3d robot_goal_orientation;  // world frame
	Vector3d device_command_force;	  // device base frame
	Vector3d device_command_moment;   // device base frame
	double device_gripper_force;

	HapticControllerOtuput()
		: robot_goal_position(Vector3d::Zero()),
		  robot_goal_orientation(Matrix3d::Identity()),
		  device_command_force(Vector3d::Zero()),
		  device_command_moment(Vector3d::Zero()),
		  device_gripper_force(0.0) {}
};

struct HapticControllerInput {
	Vector3d device_position;		   // device base frame
	Matrix3d device_orientation;	   // device base frame
	Vector3d device_linear_velocity;   // device base frame
	Vector3d device_angular_velocity;  // device base frame
	double device_gripper_position;
	double device_gripper_velocity;
	Vector3d robot_position;	 	  // world frame
	Matrix3d robot_orientation;	 	  // world frame
	Vector3d robot_linear_velocity;   // world frame
	Vector3d robot_angular_velocity;  // world frame
	Vector3d robot_sensed_force;  	  // world frame
	Vector3d robot_sensed_moment;	  // world frame

	HapticControllerInput()
		: device_position(Vector3d::Zero()),
		  device_orientation(Matrix3d::Identity()),
		  device_linear_velocity(Vector3d::Zero()),
		  device_angular_velocity(Vector3d::Zero()),
		  device_gripper_position(0.0),
		  device_gripper_velocity(0.0),
		  robot_position(Vector3d::Zero()),
		  robot_orientation(Matrix3d::Identity()),
		  robot_sensed_force(Vector3d::Zero()),
		  robot_sensed_moment(Vector3d::Zero()) {}
};

class HapticDeviceController
{

public:

struct DeviceLimits {
	double max_linear_stiffness;
	double max_angular_stiffness;
	double max_linear_damping;
	double max_angular_damping;
	double max_force;
	double max_torque;

	DeviceLimits(const Vector2d& max_stiffness, const Vector2d& max_damping, const Vector2d& max_force_torque)
		: max_linear_stiffness(max_stiffness(0)),
		  max_angular_stiffness(max_stiffness(1)),
		  max_linear_damping(max_damping(0)),
		  max_angular_damping(max_damping(1)),
		  max_force(max_force_torque(0)),
		  max_torque(max_force_torque(1)) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//// Constructor, Destructor and Initialization of the haptic controllers
////////////////////////////////////////////////////////////////////////////////////////////////////


	HapticDeviceController(const DeviceLimits& device_limits,
					const Eigen::Affine3d& robot_initial_pose,
					const Eigen::Affine3d& device_home_pose = Eigen::Affine3d::Identity(),
					const Eigen::Matrix3d& device_base_rotation_in_world = Eigen::Matrix3d::Identity());

	/**
	 * @brief Detructor  This destructor deletes the pointers, stop the haptic controller, and close the haptic device.
	 *
	 */
	~HapticDeviceController() = default;

    // disallow copy, assign and default constructors
    HapticDeviceController() = delete;
    HapticDeviceController(const HapticDeviceController&) = delete;
    void operator=(const HapticDeviceController&) = delete;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Impedance-type and admittance-type controllers in bilateral teleoperation scheme
////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Computes the haptic commands for the haptic device and the controlled robot.
	 * @details:
	 *			computeHapticCommands3-6d implements the impedance bilateral teleoperation scheme. The haptic commands
	 *				are computed from the sensed task force (haptic_feedback_from_proxy=false) or through a stiffness/
	 * 				damping field between the desired position and the proxy position (haptic_feedback_from_proxy=true).
	 *				'haptic_feedback_from_proxy' flag is defined by the user and set to false by default.
	 *				When computing the force feedback from the force sensor data, the task force must be set thanks to
	 *				updateSensedForce() before calling this function. If the proxy evaluation is used, the current position,
	 * 				rotation, and velocity of the proxy are updated with updateVirtualProxyPositionVelocity().
	 *
	 *			computeHapticCommandsWorkspaceExtension3-6d augments the classic impedance teleoperation scheme with a
	 *				dynamic workspace extension algorithm.
	 *
	 *			computeHapticCommandsAdmittance3-6d implements the admittance bilateral teleoperation scheme. The haptic commands
	 *				are computed from the velocity error in the robot controller. The current robot position and velocity must be
	 *				set thanks to updateSensedRobotPositionVelocity().
	 *
	 *	 		computeHapticCommandsUnifiedControl3-6d implements a unified Motion and Force controller for haptic teleoperation.
	 *				The slave robot is controlled in force (desired_force/torque_robot) in the physical interaction direction
	 *				(defined by the force selection matrices _sigma_force and _sigma_moment) and in motion (desired_position/rotation_robot)
	 *				in the orthogonal direction (defined by the motion selection matrices _sigma_position and _sigma_orientation).
	 *				The selection matrices must be defined before calling the controller thanks to updateSelectionMatrices().
	 *				The haptic feedback is computed as the sum of the sensed task interaction projected into the motion-controlled space
	 *				and a virtual spring-damper (_force_guidance_xx stiffness and damping parameters) into the force-controlled space.
	 *				The task force must be sent thanks to updateSensedForce() and the current robot position and velocity must be updated
	 *				with updateSensedRobotPositionVelocity() before calling this function.
	 *
	 * 			Guidance plane or line can be added to the haptic controllers (_enable_plane_guidance=true, _enable_line_guidance=true).
	 *			The plane is defined thanks to the method setPlane and the line thanks to setLine.
	 *
	 *			computeHapticCommands...6d(): The 6 DOFs are controlled and feedback.
	 *			computeHapticCommands...3d(): The haptic commands are evaluated in position only, the 3 translational DOFs
	 *										 are controlled and rendered to the user.
	 *
	 *		Make sure to update the haptic device data (position, velocity, sensed force) from the redis keys before calling this function!
	 *
	 *
	 */

	HapticControllerOtuput computeHapticControl(const HapticControllerInput& input, const bool verbose = false);

	void enableOrientationTeleoperation(const bool enable_orientation_teleoperation);

	const HapticControlType& getHapticControlType() const {
		return _haptic_control_type;
	}
	void setHapticControlType(const HapticControlType& haptic_control_type);

	bool homed() const {
		return _device_homed;
	}

private:

	HapticControllerOtuput computeDetachedControl(const HapticControllerInput& input);

	HapticControllerOtuput computeHomingControl(const HapticControllerInput& input);

	HapticControllerOtuput computeMotionMotionControl(const HapticControllerInput& input);

	// void computeMotionMotionPositionControl(const HapticControllerInput& input, HapticControllerOtuput& output);

	// HapticControllerOtuput computeHapticCommandsAdmittance6d(const HapticControllerInput& input);

	// HapticControllerOtuput computeHapticCommandsAdmittance3d(const HapticControllerInput& input);

	// HapticControllerOtuput computeHapticCommandsWorkspaceExtension6d(const HapticControllerInput& input);

	// HapticControllerOtuput computeHapticCommandsWorkspaceExtension3d(const HapticControllerInput& input);

	// HapticControllerOtuput computeHapticCommandsUnifiedControl6d(const HapticControllerInput& input);

	// HapticControllerOtuput computeHapticCommandsUnifiedControl3d(const HapticControllerInput& input);

	void validateOutput(HapticControllerOtuput& output, const bool verbose);

public:
///////////////////////////////////////////////////////////////////////////////////
// Haptic guidance related methods
///////////////////////////////////////////////////////////////////////////////////
	/**
	*  @brief computes the guidance force for the 3D plane set by the user
	*/
	Vector3d ComputePlaneGuidanceForce(const Vector3d& device_position, const Vector3d& device_velocity);

		/**
	*  @brief computes the guidance force for the 3D line set by the user
	*/
	Vector3d ComputeLineGuidanceForce(const Vector3d& device_position, const Vector3d& device_velocity);


// ///////////////////////////////////////////////////////////////////////////////////
// // Updating methods for haptic feedback computation
// ///////////////////////////////////////////////////////////////////////////////////
// 	/**
// 	 * @brief Update the sensed force from the task interaction
// 	 * @details When rendering the sensed force as haptic feedback in the impedance-type bilateral scheme, this function updates the force sensor data. The global variable
// 	 * 			'filter_on' enables the filtering of force sensor data. The filter parameters are set thanscaling_factor_trans to setFilterCutOffFreq().
// 	 *
// 	 * @param sensed_task_force    	Sensed task force from the controlled robot's sensor
// 	 */
// 	void updateSensedForce(const Eigen::VectorXd sensed_task_force = Eigen::VectorXd::Zero(6));

// 	/**
// 	 * @brief Update the current position, orientation, and velocity of the controlled robot
// 	 * @details When evaluating the haptic feedback in the admittance-type bilateral scheme, this function updates the current robot position/rotation.
// 	 *
// 	 * @param current_position_robot    	The current position of the controlled robot
// 	 * @param current_rotation_robot    	The current orientation of the controlled robot
// 	 * @param current_trans_velocity_robot  The current translational velocity of the controlled robot
// 	 * @param current_rot_velocity_robot 	The current rotational velocity of the controlled robot
// 	 */
// 	void updateSensedRobotPositionVelocity(const Eigen::Vector3d current_position_robot,
// 											const Eigen::Vector3d current_trans_velocity_robot,
// 											const Eigen::Matrix3d current_rotation_robot = Eigen::Matrix3d::Identity(),
// 											const Eigen::Vector3d current_rot_velocity_robot = Eigen::Vector3d::Zero());

// 	/**
// 	 * @brief Update the current position, orientation, and velocity of the proxy
// 	 * @details When evaluating the haptic feedback via an impedance/damping proxy, this function updates the current proxy position, rotation, and velocity.
// 	 *
// 	 * @param current_position_proxy    	The current position of the controlled virtual proxy
// 	 * @param current_rotation_proxy    	The current orientation of the controlled virtual proxy
// 	 * @param current_trans_velocity_proxy  The current translational velocity of the virtual proxy
// 	 * @param current_rot_velocity_proxy	The current rotational velocity of the virtual proxy
// 	 */
// 	void updateVirtualProxyPositionVelocity(const Eigen::Vector3d current_position_proxy,
// 											const Eigen::Vector3d current_trans_velocity_proxy,
// 											const Eigen::Matrix3d current_rotation_proxy = Eigen::Matrix3d::Identity(),
// 											const Eigen::Vector3d current_rot_velocity_proxy = Eigen::Vector3d::Zero());

// 	/**
// 	 * @brief Update the desired selection matrices to project the haptic feedback in the unified controller
// 	 *
// 	 * @param _sigma_position			The position selection matrix in robot frame
// 	 * @param _sigma_orientation		The orientation selection matrix in robot frame
// 	 * @param _sigma_force				The force selection matrix in robot frame
// 	 * @param _sigma_moment				The torque selection matrix in robot frame
// 	 */
// 	void updateSelectionMatrices(const Eigen::Matrix3d sigma_position, const Eigen::Matrix3d sigma_orientation,
// 								const Eigen::Matrix3d sigma_force, const Eigen::Matrix3d sigma_moment);



///////////////////////////////////////////////////////////////////////////////////
// Haptic device specific methods
///////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////
// Parameter setting methods
///////////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Set the scaling factors between the device Workspace and the task environment
	 *
	 * @param scaling_factor_trans      Translational scaling factor
	 * @param scaling_factor_rot       	Rotational scaling factor
	 */
	void setScalingFactors(const double scaling_factor_pos, const double scaling_factor_ori = 1.0);

	// /**
	//  * @brief Set the position controller gains of the haptic device for the homing task.
	//  * 			The gains are given as a ratio of the device maximum damping and stiffness (between 0 and 1).
	//  *
	//  * @param kp_position_ctrl_device    	Proportional gain in position
	//  * @param kv_position_ctrl_device    	Derivative term in position
	//  * @param kp_orientation_ctrl_device    Proportional gain in orientation
	//  * @param kv_orientation_ctrl_device    Derivative term in orientation
	//  */
	// void setPosCtrlGains (const double kp_position_ctrl_device, const double kv_position_ctrl_device,
	// 					  const double kp_orientation_ctrl_device, const double kv_orientation_ctrl_device);

	// /**
	//  * @brief Set the impedance/damping terms for the force feedback evaluation in admittance-type bilateral scheme
	//  * 		  Define the reduction factors between the actual task force and the rendered force
	//  *
	//  * @param kp_robot_trans_velocity       		Robot impedance term in position for feedback computation
	//  * @param ki_robot_trans_velocity 	  			Robot damping term in position for feedback computation
	//  * @param kp_robot_rot_velocity       			Robot impedance term in orientation for feedback computation
	//  * @param ki_robot_rot_velocity 	  			Robot damping term in orientation for feedback computation
	//  * @param robot_trans_admittance 				Robot desired admittance in translation
	//  * @param robot_rot_admittance 					Robot desired admittance in rotation
	//  * @param reduction_factor_torque_feedback 		Matrix of force reduction factors
	//  * @param reduction_factor_force_feedback 		Matrix of torque reduction factors
	//  */
	// void setForceFeedbackCtrlGains (const double kp_robot_trans_velocity, const double kv_robot_trans_velocity,
	// 								const double kp_robot_rot_velocity, const double kv_robot_rot_velocity,
	// 								const double robot_trans_admittance,
	// 								const double robot_rot_admittance,
	// 								const Matrix3d reduction_factor_force_feedback = Matrix3d::Identity(),
	// 								const Matrix3d reduction_factor_torque_feedback = Matrix3d::Identity());

	// void setReductionFactorForceFeedback (const Matrix3d reduction_factor_force_feedback,
	// 								const Matrix3d reduction_factor_torque_feedback);


	// /**
	//  * @brief Set the impedance/damping terms for the force feedback evaluation via virtual proxy
	//  *
	//  * @param proxy_position_impedance       		Proxy impedance term in position
	//  * @param proxy_position_damping 	  			Proxy damping term in position
	//  * @param proxy_orientation_impedance       	Proxy impedance term in orientation
	//  * @param proxy_orientation_damping 	  		Proxy damping term in orientation
	//  */
	// void setVirtualProxyGains (const double proxy_position_impedance, const double proxy_position_damping,
	// 								const double proxy_orientation_impedance, const double proxy_orientation_damping);

	// /**
	//  * @brief Set the impedance terms for the virtual force guidance computation in unified controller
	//  * and the damping terms to Zero
	//  *
	//  * @param force_guidance_position_impedance       		Guidance impedance term in position
	//  * @param force_guidance_orientation_impedance       	Guidance impedance term in orientation
	//  */
	// void setVirtualGuidanceGains (const double force_guidance_position_impedance,
	// 								const double force_guidance_orientation_impedance);	

	// /**
	//  * @brief Set the impedance/damping terms for the virtual force guidance computation in unified controller
	//  *
	//  * @param force_guidance_position_impedance       		Guidance impedance term in position
	//  * @param force_guidance_position_damping       		Guidance damping term in position
	//  * @param force_guidance_orientation_impedance       	Guidance impedance term in orientation
	//  * @param force_guidance_orientation_damping         	Guidance damping term in orientation
	//  */
	// void setVirtualGuidanceGains (const double force_guidance_position_impedance, const double force_guidance_position_damping,
	// 								const double force_guidance_orientation_impedance, const double force_guidance_orientation_damping);

	/**
	 * @brief Set the center of the device Workspace
	 * @details The haptic device home position and orientation are set through a Vector3d and a Matrix3d
	 *
	 * @param home_position_device     The home position of the haptic device in its operational space
	 * @param home_rotation_device     The home orientation of the haptic device in its operational space
	 */
	void setDeviceCenter(const Eigen::Vector3d home_position_device,
		            	 const Eigen::Matrix3d home_rotation_device = Eigen::Matrix3d::Identity());


	// /**
	//  * @brief Set the center of the task Workspace
	//  * @details The robot home position and orientation, with respect to the task, are set through a Vector3d and a Matrix3d
	//  *
	//  * @param center_position_robot     The task home position of the robot in its operational space
	//  * @param center_rotation_robot     The task home orientation of the robot in its operational space
	//  */
	// void setRobotCenter(const Eigen::Vector3d center_position_robot,
	// 	            	const Eigen::Matrix3d center_rotation_robot = Eigen::Matrix3d::Identity());

	// /**
	//  * @brief Set the rotation matrix between the haptic device global frame to the robot global frame
	//  *
	//  * @param Rotation_Matrix_DeviceToRobot    Rotation matrix between from the device to robot frame
	//  */
	// void setDeviceRobotRotation(const Eigen::Matrix3d Rotation_Matrix_DeviceToRobot = Eigen::Matrix3d::Identity());

	/**
	 * @brief Sets the size of the device Workspace to add virtual limits in the force feedback
	 * @details The size of the device Workspace is set through the radius of its equivalent sphere and the maximum tilt angles.
	 *
	 * @param device_workspace_radius_limit     Radius of the smallest sphere including the haptic device Workspace
	 * @param device_workspace_angle_limit   	Maximum tilt angle of the haptic device
	 */
	void setWorkspaceLimits(double device_workspace_radius_limit, double device_workspace_angle_limit);


///////////////////////////////////////////////////////////////////////////////////
// Workspace extension related methods
///////////////////////////////////////////////////////////////////////////////////
	// /**
	//  * @brief Sets the size of the device Workspace and the task environment
	//  * @details The size of the device Workspace and the task environment are set through the radius of the smallest sphere including each Workspace and the maximum tilt angles.
	//  *
	//  * @param device_workspace_radius_max       Radius of the smallest sphere including the haptic device Workspace
	//  * @param task_workspace_radius_max        	Radius of the smallest sphere including the task environment
	//  * @param device_workspace_tilt_angle_max   Maximum tilt angle of the haptic device
	//  * @param task_workspace_tilt_angle_max     Maximum tilt angle of the controlled robot for the task
	//  */
	// void setWorkspaceSize(double device_workspace_radius_max, double task_workspace_radius_max,
	// 									    double device_workspace_tilt_angle_max, double task_workspace_tilt_angle_max);

	// /**
	//  * @brief Sets the percentage of drift force and drift velocity accepted by the user as just noticeable difference
	//  * @details The level of drift force is set with respect to the task force feedback and the the level of drift velocity
	//  *          with respect to the device current velocity.
	//  *
	//  * @param drift_force_admissible_ratio     Percentage of drift force with repect to the task force feedback
	//  * @param drift_velocity_admissible_ratio  Percentage of drift velocity with respect to the device velocity
	//  */
	// void setNoticeableDiff(double drift_force_admissible_ratio, double drift_velocity_admissible_ratio);


///////////////////////////////////////////////////////////////////////////////////
// Haptic guidance related settings methods
///////////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Sets the stiffness and damping parameters for the haptic guidance (plane and line)
	 *
	 * @param guidance_stiffness 	Stiffness of the virtual guidance
	 * @param guidance_damping 		Damping of the virtual guidance
	 */
	void setHapticGuidanceGains(const double guidance_stiffness, const double guidance_damping);

	/**
	 * @brief Defines an artibtrary 3D plane using a point and a normal vector
	 *
	 * @param plane_point_origin coordinate vector of the origin point
	 * @param plane_normal_vec normal vector for the plane
	 */
	void setPlane(const Eigen::Vector3d plane_origin_point, const Eigen::Vector3d plane_normal_vec);

	/**
	* @brief stores user defined values for line guidance
	* @param _first_point vector to first point from world origin
	* @param _second_point vector to second point from world origin
	*/
	void setLine(const Eigen::Vector3d line_first_point, const Eigen::Vector3d line_second_point);





private:

///////////////////////////////////////////////////////////////////////////////////
// Attributes
///////////////////////////////////////////////////////////////////////////////////

//// Inputs to be define by the users ////

	bool _haptic_feedback_from_proxy; // If set to true, the force feedback is computed from a stiffness/damping proxy.
									 // Otherwise the sensed force are rendered to the user.
	bool _send_haptic_feedback;       // If set to false, send 0 forces and torques to the haptic device

	bool _enable_orientation_teleoperation; // If set to true, the orientation of the robot is controlled and rendered to the user
	bool _enable_gripper_teleoperation; // If set to true, the gripper of the robot is controlled and rendered to the user

	// bool _filter_on; //Enable filtering force sensor data. To be use only if updateSensedForce() is called cyclically.

	bool _enable_plane_guidance; // add guidance along a user-defined plane

	bool _enable_line_guidance; // add guidance along a user-defined plane

	bool _add_workspace_virtual_limit; // add a virtual sphere delimiting the haptic device workspace

	HapticControlType _haptic_control_type;

//// Status and robot/device's infos ////

	//Device specifications
	DeviceLimits _device_limits;

	double _device_workspace_radius_limit;
	double _device_workspace_angle_limit;
	// Device status
	bool _device_homed;

	// Workspace extension parameters
	bool _first_iteration;
	// Maximal haptic device velocity during the task
	double _max_rot_velocity_device, _max_trans_velocity_device;
	// Device drift force
	Eigen::Vector3d _drift_force;
	Eigen::Vector3d _drift_torque;
	// Device drift velocities
	Eigen::Vector3d _drift_rot_velocity;
	Eigen::Vector3d _drift_trans_velocity;
	// Drifting Workspace center of the controlled robot, position and orientation
	Eigen::Vector3d _center_position_robot_drift;
	Eigen::Matrix3d _center_rotation_robot_drift;

	// Haptic guidance gains
	double _guidance_stiffness;
	double _guidance_damping;
	// Guidance plane parameters
	Eigen::Vector3d _plane_origin_point;
	Eigen::Vector3d _plane_normal_vec;
	// Eigen::Vector3d _guidance_force_plane;
	// Guidance line parameters
	Eigen::Vector3d _guidance_line_vec;
	Eigen::Vector3d _line_first_point;
	Eigen::Vector3d _line_second_point;
	// Eigen::Vector3d _guidance_force_line;

	// Haptic device home position and orientation
	Eigen::Vector3d _home_position_device;
	Eigen::Matrix3d _home_rotation_device;

	// Workspace center of the controlled robot in the robot frame
	Eigen::Vector3d _center_position_robot;
	Eigen::Matrix3d _center_rotation_robot;

	//Transformation matrix from the device frame to the robot frame
	Eigen::Matrix3d _R_world_device;

//// Controllers parameters, set through setting methods ////

	// Workspace scaling factors in translation and rotation
	double _scaling_factor_trans, _scaling_factor_rot;

	//Position controller parameters for homing task
	double _kp_homing_pos;
	double _kv_homing_pos;
	double _kp_homing_ori;
	double _kv_homing_ori;
	double _homing_max_linvel;
	double _homing_max_angvel;

	// Virtual proxy parameters
	double _proxy_position_impedance;
	double _proxy_orientation_impedance;
	double _proxy_orientation_damping;
	double _proxy_position_damping;

	// Force feedback controller parameters
	double _kp_robot_trans_velocity;
	double _kp_robot_rot_velocity;
	double _kv_robot_rot_velocity;
	double _kv_robot_trans_velocity;

	double _robot_trans_admittance;
	double _robot_rot_admittance;

	Eigen::Matrix3d _reduction_factor_force_feedback;
	Eigen::Matrix3d _reduction_factor_torque_feedback;

	// Virtual force guidance parameters
	double _force_guidance_position_impedance;
	double _force_guidance_orientation_impedance;
	double _force_guidance_position_damping;
	double _force_guidance_orientation_damping;

	//Task and device workspaces' size
	double _device_workspace_radius_max, _task_workspace_radius_max; // Radius (in meter)
	double _device_workspace_tilt_angle_max, _task_workspace_tilt_angle_max; // max tilt angles (in degree)

	// Admissible drift force and velocity ratio
	double _drift_force_admissible_ratio; // As a percentage of task force
	double _drift_velocity_admissible_ratio; // As a percentage of device velocity
};


} /* namespace Sai2Primitives */

/* SAI2_HAPTIC_DEVICE_CONTROLLER_H_ */
#endif
