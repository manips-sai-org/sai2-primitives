/*
 * HapticDeviceController.h
 *
 *      This controller implements a bilateral haptic teleoperation scheme in
 * open loop. The commands are computed for the haptic device (force feedback)
 * and the controlled robot (desired task). HapticDeviceController includes
 * impedance-type and admittance-type controllers, with plane/line/orientation
 * guidances, and the workspace mapping algorithm.
 *
 *      Authors: Margot Vulliez & Mikael Jorda
 */

#ifndef SAI2_HAPTIC_DEVICE_CONTROLLER_H_
#define SAI2_HAPTIC_DEVICE_CONTROLLER_H_

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "Sai2Model.h"

namespace Sai2Primitives {

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
	Vector3d device_command_moment;	  // device base frame
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
	Vector3d robot_position;		  // world frame
	Matrix3d robot_orientation;		  // world frame
	Vector3d robot_linear_velocity;	  // world frame
	Vector3d robot_angular_velocity;  // world frame
	Vector3d robot_sensed_force;	  // world frame
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

class HapticDeviceController {
public:
	struct DeviceLimits {
		double max_linear_stiffness;
		double max_angular_stiffness;
		double max_linear_damping;
		double max_angular_damping;
		double max_force;
		double max_torque;

		DeviceLimits(const Vector2d& max_stiffness, const Vector2d& max_damping,
					 const Vector2d& max_force_torque)
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

	HapticDeviceController(
		const DeviceLimits& device_limits, const Affine3d& robot_initial_pose,
		const Affine3d& device_home_pose = Affine3d::Identity(),
		const Matrix3d& device_base_rotation_in_world = Matrix3d::Identity());

	/**
	 * @brief Detructor  This destructor deletes the pointers, stop the haptic
	 * controller, and close the haptic device.
	 *
	 */
	~HapticDeviceController() = default;

	// disallow copy, assign and default constructors
	HapticDeviceController() = delete;
	HapticDeviceController(const HapticDeviceController&) = delete;
	void operator=(const HapticDeviceController&) = delete;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// Haptic controllers for all the sypported controler types
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Computes the haptic commands for the haptic device and the
	 *controlled robot.
	 * @details:
	 *			computeHapticCommands3-6d implements the impedance bilateral
	 *teleoperation scheme. The haptic commands are computed from the sensed
	 *task force (_compute_haptic_feedback_from_proxy=false) or through a
	 *stiffness/ damping field between the desired position and the proxy
	 *position (haptic_feedback_from_proxy=true).
	 *				'_compute_haptic_feedback_from_proxy' flag is defined by the user
	 *and set to false by default. When computing the force feedback from the
	 *force sensor data, the task force must be send in the algorithm input. If
	 *the proxy evaluation is used, the current position, rotation, and velocity
	 *of the proxy are considered to be the robot current position and
	 *orientation in the input.
	 *
	 *			computeHapticCommandsWorkspaceExtension3-6d augments the classic
	 *impedance teleoperation scheme with a dynamic workspace extension
	 *algorithm.
	 *
	 *			computeHapticCommandsAdmittance3-6d implements the admittance
	 *bilateral teleoperation scheme. The haptic commands are computed from the
	 *velocity error in the robot controller. The current robot position and
	 *velocity must be set thanks to updateSensedRobotPositionVelocity().
	 *
	 *	 		computeHapticCommandsUnifiedControl3-6d implements a unified Motion
	 *and Force controller for haptic teleoperation. The slave robot is
	 *controlled in force (desired_force/torque_robot) in the physical
	 *interaction direction (defined by the force selection matrices
	 *_sigma_force and _sigma_moment) and in motion
	 *(desired_position/rotation_robot) in the orthogonal direction (defined by
	 *the motion selection matrices _sigma_position and _sigma_orientation). The
	 *selection matrices must be defined before calling the controller thanks to
	 *updateSelectionMatrices(). The haptic feedback is computed as the sum of
	 *the sensed task interaction projected into the motion-controlled space and
	 *a virtual spring-damper (_force_guidance_xx stiffness and damping
	 *parameters) into the force-controlled space. The task force must be sent
	 *thanks to updateSensedForce() and the current robot position and velocity
	 *must be updated with updateSensedRobotPositionVelocity() before calling
	 *this function.
	 *
	 * 			Guidance plane or line can be added to the haptic controllers
	 *(_enable_plane_guidance=true, _enable_line_guidance=true). The plane is
	 *defined thanks to the method setPlane and the line thanks to setLine.
	 *
	 *			computeHapticCommands...6d(): The 6 DOFs are controlled and
	 *feedback. computeHapticCommands...3d(): The haptic commands are evaluated
	 *in position only, the 3 translational DOFs are controlled and rendered to
	 *the user.
	 *
	 *		Make sure to update the haptic device data (position, velocity, sensed
	 *force) from the redis keys before calling this function!
	 *
	 *
	 */

	HapticControllerOtuput computeHapticControl(
		const HapticControllerInput& input, const bool verbose = false);

	void resetRobotOffset() { _reset_robot_offset = true; }

private:
	void validateOutput(HapticControllerOtuput& output, const bool verbose);

	HapticControllerOtuput computeDetachedControl(
		const HapticControllerInput& input);

	HapticControllerOtuput computeHomingControl(
		const HapticControllerInput& input);

	HapticControllerOtuput computeMotionMotionControl(
		const HapticControllerInput& input);

	// HapticControllerOtuput computeForceMotionControl(const
	// HapticControllerInput& input);

	// HapticControllerOtuput computeHybridControlWithProxy(const
	// HapticControllerInput& input);

	// HapticControllerOtuput computeWorkspaceExtensionControl(const
	// HapticControllerInput& input);

	void ApplyPlaneGuidanceForce(Vector3d& haptic_force_to_update,
								 const Vector3d& device_position,
								 const Vector3d& device_velocity);

	void ApplyLineGuidanceForce(Vector3d& haptic_force_to_update,
								const Vector3d& device_position,
								const Vector3d& device_velocity);

public:
	///////////////////////////////////////////////////////////////////////////////////
	// Parameter setting methods
	///////////////////////////////////////////////////////////////////////////////////

	void setHapticControlType(const HapticControlType& haptic_control_type) {
		_haptic_control_type = haptic_control_type;
	}
	const HapticControlType& getHapticControlType() const {
		return _haptic_control_type;
	}

	void setEnableOrientationTeleoperation(const bool enable);
	bool getEnableOrientationTeleoperation() const {
		return _enable_orientation_teleoperation;
	}

	void setHapticFeedbackFromProxy(const bool haptic_feedback_from_proxy) {
		_compute_haptic_feedback_from_proxy = haptic_feedback_from_proxy;
	}
	bool getHapticFeedbackFromProxy() const {
		return _compute_haptic_feedback_from_proxy;
	}

	void setSendHapticFeedback(const bool send_haptic_feedback) {
		_send_haptic_feedback = send_haptic_feedback;
	}
	bool getSendHapticFeedback() const { return _send_haptic_feedback; }

	bool getHomed() const { return _device_homed; }

	void setScalingFactors(const double scaling_factor_pos,
						   const double scaling_factor_ori = 1.0);

	/**
	 * @brief Set the Reduction Factor for force and moment. The command force
	 * and moments to the haptic device are reduced by these factors
	 *
	 * @param reduction_factor_force
	 * @param reduction_factor_moment
	 */
	void setReductionFactorForceMoment(const double reduction_factor_force,
									   const double reduction_factor_moment);

	/**
	 * @brief Set the Gains used for all the device control types:
	 * - The homing controller
	 * - The feedback from proxy in the motion-motion controller and worlspace
	 * extension
	 * - The stiffness and damping of the force-motion cotnroller
	 * - The proxy stiffness and damping in the hybrid controller
	 *
	 * @param kp_pos
	 * @param kv_pos
	 * @param kp_ori
	 * @param kv_ori
	 */
	void setDeviceControlGains(const double kp_pos, const double kv_pos);
	void setDeviceControlGains(const double kp_pos, const double kv_pos,
							   const double kp_ori, const double kv_ori);

	/**
	 * @brief Set the Haptic Guidance Gains for all the haptic guidance tasks
	 * (plane, line and worlspace virtual limits)
	 *
	 * @param kp_guidance_pos
	 * @param kv_guidance_pos
	 * @param kp_guidance_ori
	 * @param kv_guidance_ori
	 */
	void setHapticGuidanceGains(const double kp_guidance_pos,
								const double kv_guidance_pos);
	void setHapticGuidanceGains(const double kp_guidance_pos,
								const double kv_guidance_pos,
								const double kp_guidance_ori,
								const double kv_guidance_ori);

	void enablePlaneGuidance(const Vector3d plane_origin_point,
							 const Vector3d plane_normal_vec);
	void disablePlaneGuidance() { _enable_plane_guidance = false; }

	void enableLineGuidance(const Vector3d line_first_point,
							const Vector3d line_second_point);
	void disableLineGuidance() { _enable_line_guidance = false; }

	/**
	 * @brief Sets the size of the device Workspace to add virtual limits in the
	 * force feedback
	 * @details The size of the device Workspace is set through the radius of
	 * its equivalent sphere and the maximum tilt angles.
	 *
	 * @param device_workspace_radius_limit     Radius of the smallest sphere
	 * including the haptic device Workspace
	 * @param device_workspace_angle_limit   	Maximum tilt angle of the haptic
	 * device
	 */
	void setHapticWorkspaceVirtualLimits(double device_workspace_radius_limit,
										 double device_workspace_angle_limit);

private:
	// controller states
	bool _compute_haptic_feedback_from_proxy;  // If set to true, the force
											   // feedback is computed from a
											   // stiffness/damping proxy.
											   // Otherwise the sensed force are
											   // rendered to the user.
	bool _send_haptic_feedback;	 // If set to false, send 0 forces and torques
								 // to the haptic device

	bool _enable_orientation_teleoperation;	 // If set to true, the orientation
											 // of the robot is controlled and
											 // rendered to the user

	bool _enable_plane_guidance;  // add guidance along a user-defined plane
	bool _enable_line_guidance;	  // add guidance along a user-defined plane
	bool _enable_workspace_virtual_limit;  // add a virtual sphere delimiting
										   // the haptic device workspace

	HapticControlType _haptic_control_type;
	bool _device_homed;

	// Device specifications
	DeviceLimits _device_limits;

	// Rotation matrix from the device frame to the robot frame
	Matrix3d _R_world_device;

	// Haptic device home position and orientation
	Vector3d _home_position_device;
	Matrix3d _home_rotation_device;

	// Workspace center of the controlled robot in the robot frame
	Vector3d _center_position_robot;
	Matrix3d _center_rotation_robot;
	bool _reset_robot_offset;

	// Haptic guidance gains
	double _kp_guidance_pos;
	double _kv_guidance_pos;
	double _kp_guidance_ori;
	double _kv_guidance_ori;

	// Guidance plane parameters
	Vector3d _plane_origin_point;
	Vector3d _plane_normal_vec;

	// Guidance line parameters
	Vector3d _guidance_line_vec;
	Vector3d _line_first_point;
	Vector3d _line_second_point;

	// Device workspace virtual limits
	double _device_workspace_radius_limit;
	double _device_workspace_angle_limit;

	// Haptic Controller Gains
	double _kp_haptic_pos;
	double _kv_haptic_pos;
	double _kp_haptic_ori;
	double _kv_haptic_ori;

	// homing controller velocity limits
	double _homing_max_linvel;
	double _homing_max_angvel;

	// scaling and force reduction factors
	double _scaling_factor_pos;
	double _scaling_factor_ori;
	double _reduction_factor_force;
	double _reduction_factor_moment;
};

} /* namespace Sai2Primitives */

#endif /* SAI2_HAPTIC_DEVICE_CONTROLLER_H_ */
