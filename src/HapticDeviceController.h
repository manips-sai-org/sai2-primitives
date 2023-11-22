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
};

struct HapticControllerOtuput {
	Vector3d robot_goal_position;	  // world frame
	Matrix3d robot_goal_orientation;  // world frame
	Vector3d device_command_force;	  // device base frame
	Vector3d device_command_moment;	  // device base frame

	HapticControllerOtuput()
		: robot_goal_position(Vector3d::Zero()),
		  robot_goal_orientation(Matrix3d::Identity()),
		  device_command_force(Vector3d::Zero()),
		  device_command_moment(Vector3d::Zero()) {}
};

struct HapticControllerInput {
	Vector3d device_position;		   // device base frame
	Matrix3d device_orientation;	   // device base frame
	Vector3d device_linear_velocity;   // device base frame
	Vector3d device_angular_velocity;  // device base frame
	Vector3d robot_position;		   // world frame
	Matrix3d robot_orientation;		   // world frame
	Vector3d robot_linear_velocity;	   // world frame
	Vector3d robot_angular_velocity;   // world frame
	Vector3d robot_sensed_force;	   // world frame
	Vector3d robot_sensed_moment;	   // world frame

	HapticControllerInput()
		: device_position(Vector3d::Zero()),
		  device_orientation(Matrix3d::Identity()),
		  device_linear_velocity(Vector3d::Zero()),
		  device_angular_velocity(Vector3d::Zero()),
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
	 * controlled robot. The input data scructure contains the device and robot
	 * position, orientation and velocity, as well as the robot sensed force and
	 * moments if direct haptic feedback is desired. The output data structure
	 * contains the robot goal position and orientation as well as the device
	 * command force and moment.
	 *
	 * @details
	 * - Motion Motion control implements the impedance bilateral
	 * teleoperation scheme. The haptic commands can be computed from direct
	 * feedback using the sensed force and moments from the robot, or through a
	 * spring damper system attaching the device pose and the proxy pose (the
	 * proxy being the robot pose from the input). The behavior can be different
	 * in different direction and is set by the function
	 * parametrizeProxyForceFeedbackSpace or
	 * parametrizeProxyForceFeedbackSpaceFromRobotForceSpace, and their moment
	 * equivalent. The default behavior is to use direct feedback in all
	 * directions.
	 *
	 * - Force Motion Control implements an admittance type bilateral
	 * teleoperation where the user pushes the haptic device against a force
	 * field in a desired direction and this generated a robot velocity command.
	 *
	 * @param input device and robot position, orientation and velocity, and
	 * robot sensed force and moments
	 * @param verbose whether to print a message is the output was saturated by
	 * the validateOutput function
	 * @return HapticControllerOtuput: robot goal pose and device command force
	 */
	HapticControllerOtuput computeHapticControl(
		const HapticControllerInput& input, const bool verbose = false);

private:
	void validateOutput(HapticControllerOtuput& output, const bool verbose);

	HapticControllerOtuput computeDetachedControl(
		const HapticControllerInput& input);

	HapticControllerOtuput computeHomingControl(
		const HapticControllerInput& input);

	HapticControllerOtuput computeMotionMotionControl(
		const HapticControllerInput& input);

	HapticControllerOtuput computeForceMotionControl(
		const HapticControllerInput& input);

	void motionMotionControlPosition(const HapticControllerInput& input,
									 HapticControllerOtuput& output);
	void motionMotionControlOrientation(const HapticControllerInput& input,
										HapticControllerOtuput& output);

	void applyPlaneGuidanceForce(Vector3d& force_to_update,
								 const HapticControllerInput& input,
								 const bool use_device_home_as_origin);

	void applyLineGuidanceForce(Vector3d& force_to_update,
								const HapticControllerInput& input,
								const bool use_device_home_as_origin);

	void applyWorkspaceVirtualLimitsForceMoment(
		const HapticControllerInput& input, HapticControllerOtuput& output);

	double getVariableDampingKvPos(const double device_linvel) const;
	double getVariableDampingKvOri(const double device_velocity) const;

public:
	///////////////////////////////////////////////////////////////////////////////////
	// Parameter setting and getting methods
	///////////////////////////////////////////////////////////////////////////////////

	const DeviceLimits& getDeviceLimits() const { return _device_limits; }

	void setHapticControlType(const HapticControlType& haptic_control_type);
	const HapticControlType& getHapticControlType() const {
		return _haptic_control_type;
	}

	void enableOrientationTeleoperation();
	void disableOrientationTeleoperation() {
		_orientation_teleop_enabled = false;
	}
	bool getEnableOrientationTeleoperation() const {
		return _orientation_teleop_enabled;
	}

	bool getHomed() const { return _device_homed; }

	void parametrizeProxyForceFeedbackSpace(
		const int proxy_feedback_space_dimension,
		const Vector3d& proxy_or_direct_feedback_axis = Vector3d::Zero());
	void parametrizeProxyForceFeedbackSpaceFromRobotForceSpace(
		const Matrix3d& robot_sigma_force);

	void parametrizeProxyMomentFeedbackSpace(
		const int proxy_feedback_space_dimension,
		const Vector3d& proxy_or_direct_feedback_axis = Vector3d::Zero());
	void parametrizeProxyMomentFeedbackSpaceFromRobotForceSpace(
		const Matrix3d& robot_sigma_moment);

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
							 const Vector3d plane_normal_direction);
	void enablePlaneGuidance() {
		enablePlaneGuidance(_plane_origin_point, _plane_normal_direction);
	}
	void disablePlaneGuidance() { _plane_guidance_enabled = false; }
	bool getPlaneGuidanceEnabled() const { return _plane_guidance_enabled; }

	void enableLineGuidance(const Vector3d line_origin_point,
							const Vector3d line_direction);
	void enableLineGuidance() {
		enableLineGuidance(_line_origin_point, _line_direction);
	}
	void disableLineGuidance() { _line_guidance_enabled = false; }
	bool getLineGuidanceEnabled() const { return _line_guidance_enabled; }

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
	void enableHapticWorkspaceVirtualLimits(
		double device_workspace_radius_limit,
		double device_workspace_angle_limit);
	void enableHapticWorkspaceVirtualLimits() {
		_device_workspace_virtual_limits_enabled = true;
	}
	void disableHapticWorkspaceVirtualLimits() {
		_device_workspace_virtual_limits_enabled = false;
	}
	bool getHapticWorkspaceVirtualLimitsEnabled() const {
		return _device_workspace_virtual_limits_enabled;
	}

	void setVariableDampingGainsPos(
		const vector<double>& velocity_thresholds,
		const vector<double>& variable_damping_gains);

	void setVariableDampingGainsOri(
		const vector<double>& velocity_thresholds,
		const vector<double>& variable_damping_gains);

	void setAdmittanceFactors(const double device_force_to_robot_delta_position,
							  const double device_moment_to_robot_delta_orientation);

	void setHomingMaxVelocity(const double homing_max_linvel,
							  const double homing_max_angvel);
	
	void setForceDeadbandForceMotionController(const double force_deadband);
	void setMomentDeadbandForceMotionController(const double force_deadband);

private:
	// controller states
	bool _orientation_teleop_enabled;
	bool _plane_guidance_enabled;
	bool _line_guidance_enabled;
	bool _device_workspace_virtual_limits_enabled;

	HapticControlType _haptic_control_type;
	bool _device_homed;

	// Device specifications
	DeviceLimits _device_limits;

	// Rotation operator from robot world frame to device base frame
	Matrix3d _R_world_device;

	// Haptic device home pose in device base frame
	Affine3d _device_home_pose;

	// Workspace center of the controlled robot in the robot world frame
	Affine3d _robot_center_pose;
	bool _reset_robot_offset;

	// proxy feedback space selection matrices
	Matrix3d _sigma_proxy_force_feedback;
	Matrix3d _sigma_proxy_moment_feedback;

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

	// variable damping for motion-motion controller direct force feedback
	vector<double> _variable_damping_linvel_thresholds;
	vector<double> _variable_damping_angvel_thresholds;
	vector<double> _variable_damping_gains_pos;
	vector<double> _variable_damping_gains_ori;

	// admittance factors for froce-motion controller
	double _device_force_to_robot_delta_position;
	double _device_moment_to_robot_delta_orientation;

	// force and moment deadbands for force-motion controller
	double _force_deadband;
	double _moment_deadband;

	// previous output
	HapticControllerOtuput _previous_output;

	// Haptic guidance gains
	double _kp_guidance_pos;
	double _kv_guidance_pos;
	double _kp_guidance_ori;
	double _kv_guidance_ori;

	// Guidance plane parameters
	Vector3d _plane_origin_point;
	Vector3d _plane_normal_direction;

	// Guidance line parameters
	Vector3d _line_origin_point;
	Vector3d _line_direction;

	// Device workspace virtual limits
	double _device_workspace_radius_limit;
	double _device_workspace_angle_limit;
};

} /* namespace Sai2Primitives */

#endif /* SAI2_HAPTIC_DEVICE_CONTROLLER_H_ */
