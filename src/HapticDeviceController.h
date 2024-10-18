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
	CLUTCH,
	MOTION_MOTION,
	FORCE_MOTION,
};

struct HapticControllerOutput {
	Vector3d robot_goal_position;	  // world frame
	Matrix3d robot_goal_orientation;  // world frame
	Vector3d device_command_force;	  // device base frame
	Vector3d device_command_moment;	  // device base frame

	HapticControllerOutput()
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
		  robot_linear_velocity(Vector3d::Zero()),
		  robot_angular_velocity(Vector3d::Zero()),
		  robot_sensed_force(Vector3d::Zero()),
		  robot_sensed_moment(Vector3d::Zero()) {}
};

class HapticDeviceController {
public:
	struct DeviceLimits {
		double max_linear_stiffness;
		double max_angular_stiffness;
		double max_gripper_stiffness;
		double max_linear_damping;
		double max_angular_damping;
		double max_gripper_damping;
		double max_force;
		double max_torque;
		double max_gripper_force;

		DeviceLimits(const Vector3d& max_stiffness, const Vector3d& max_damping,
					 const Vector3d& max_force_torque)
			: max_linear_stiffness(max_stiffness(0)),
			  max_angular_stiffness(max_stiffness(1)),
			  max_gripper_stiffness(max_stiffness(2)),
			  max_linear_damping(max_damping(0)),
			  max_angular_damping(max_damping(1)),
			  max_gripper_damping(max_damping(2)),
			  max_force(max_force_torque(0)),
			  max_torque(max_force_torque(1)),
			  max_gripper_force(max_force_torque(2)) {}
	};

	struct DefaultParameters {
		static constexpr HapticControlType haptic_control_type = CLUTCH;
		static constexpr double scaling_factor_pos = 1.0;
		static constexpr double scaling_factor_ori = 1.0;
		static constexpr double homing_max_linvel = 0.15;
		static constexpr double homing_max_angvel = M_PI;
		static constexpr double reduction_factor_force = 1.0;
		static constexpr double reduction_factor_moment = 1.0;
		static constexpr double device_force_to_robot_delta_position = 3e-5;
		static constexpr double device_moment_to_robot_delta_orientation =
			M_PI / 2000.0;
		static constexpr double force_deadband = 2.0;
		static constexpr double moment_deadband = 0.02;
		static constexpr double device_workspace_radius_limit = 0.1;
		static constexpr double device_workspace_angle_limit = M_PI / 3.0;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Constructor, Destructor and Initialization of the haptic controllers
	////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * @brief Construct a new Haptic Device Controller object
	 *
	 * @param device_limits limits for stiffness, damping and force/torque for
	 * translation, rotation and gripper
	 * @param robot_initial_pose initial pose of the robot in the world frame
	 * @param device_home_pose home pose of the device in device base frame
	 * @param device_base_rotation_in_world rotation between world frame and
	 * device base frame
	 */
	HapticDeviceController(
		const DeviceLimits& device_limits, const Affine3d& robot_initial_pose,
		const Affine3d& device_home_pose = Affine3d::Identity(),
		const Matrix3d& device_base_rotation_in_world = Matrix3d::Identity());

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
	 * @return HapticControllerOutput: robot goal pose and device command force
	 */
	HapticControllerOutput computeHapticControl(
		const HapticControllerInput& input, const bool verbose = false);

private:
	/**
	 * @brief Validates that the output command force and torque are within the
	 * device limits, and saturates those if needed
	 *
	 * @param output the output to validate and modify if needed
	 * @param verbose if true, a warning message will be printed when saturation
	 * occurs
	 */
	void validateOutput(HapticControllerOutput& output, const bool verbose);

	/**
	 * @brief Computes the control for the clutch control mode
	 *
	 * @param input
	 * @return HapticControllerOutput
	 */
	HapticControllerOutput computeClutchControl(
		const HapticControllerInput& input);

	/**
	 * @brief Computes the control for the homing control mode
	 *
	 * @param input
	 * @return HapticControllerOutput
	 */
	HapticControllerOutput computeHomingControl(
		const HapticControllerInput& input);

	/**
	 * @brief Computes the control for the motion-motion control mode
	 *
	 * @param input
	 * @return HapticControllerOutput
	 */
	HapticControllerOutput computeMotionMotionControl(
		const HapticControllerInput& input);

	/**
	 * @brief Computes the control for the force-motion control mode
	 *
	 * @param input
	 * @return HapticControllerOutput
	 */
	HapticControllerOutput computeForceMotionControl(
		const HapticControllerInput& input);

	/**
	 * @brief Computes the position part of the motion-motion control and
	 * populates the output
	 *
	 * @param input
	 * @param output
	 */
	void motionMotionControlPosition(const HapticControllerInput& input,
									 HapticControllerOutput& output);

	/**
	 * @brief Computes the orientation part of the motion-motion control and
	 * populates the output
	 *
	 * @param input
	 * @param output
	 */
	void motionMotionControlOrientation(const HapticControllerInput& input,
										HapticControllerOutput& output);

	/**
	 * @brief Apply the guidance force in case plane guidance is enabled
	 *
	 * @param force_to_update the force to update with plane guidance
	 * @param input the hapticControlInput for the current control loop
	 * @param use_device_home_as_origin whether to use the origin of the plane
	 * defined by the user or to use the device home pose as plane origin
	 */
	void applyPlaneGuidanceForce(Vector3d& force_to_update,
								 const HapticControllerInput& input,
								 const bool use_device_home_as_origin);

	/**
	 * @brief Apply the guidance force in case line guidance is enabled
	 *
	 * @param force_to_update the force to update with line guidance
	 * @param input the hapticControlInput for the current control loop
	 * @param use_device_home_as_origin whether to use the origin of the line
	 * defined by the user or or use the device home pose as line origin
	 */
	void applyLineGuidanceForce(Vector3d& force_to_update,
								const HapticControllerInput& input,
								const bool use_device_home_as_origin);

	/**
	 * @brief Apply the haptic device virtual workspace limits in case they are
	 * enabled by the user
	 *
	 * @param input the hapticControlInput for the current control loop
	 * @param output the hapticControlOutput to modify
	 */
	void applyWorkspaceVirtualLimitsForceMoment(
		const HapticControllerInput& input, HapticControllerOutput& output);

	/**
	 * @brief Compute the kv for variable damping in position
	 *
	 * @param device_linvel linear velocity of the device
	 * @return double damping
	 */
	double computeKvPosVariableDamping(const double device_linvel) const;

	/**
	 * @brief Compute the kv for variable damping in orientation
	 *
	 * @param device_velocity angular velocity of the device
	 * @return double damping
	 */
	double computeKvOriVariableDamping(const double device_velocity) const;

public:
	///////////////////////////////////////////////////////////////////////////////////
	// Parameter setting and getting methods
	///////////////////////////////////////////////////////////////////////////////////

	const DeviceLimits& getDeviceLimits() const { return _device_limits; }

	/**
	 * @brief Return the latest computed output by the computeHapticControl
	 * function
	 *
	 * @return const HapticControllerOutput&
	 */
	const HapticControllerOutput& getLatestOutput() const {
		return _latest_output;
	}

	/**
	 * @brief Returns the latest provided input to the computeHapticControl
	 * function
	 *
	 * @return const HapticControllerInput&
	 */
	const HapticControllerInput& getLatestInput() const {
		return _latest_input;
	}

	const Matrix3d& getRotationWorldToDeviceBase() const {
		return _R_world_device;
	}

	void setHapticControlType(const HapticControlType& haptic_control_type);
	const HapticControlType& getHapticControlType() const {
		return _haptic_control_type;
	}

	void enableOrientationTeleop();
	void disableOrientationTeleop() { _orientation_teleop_enabled = false; }
	bool getOrientationTeleopEnabled() const {
		return _orientation_teleop_enabled;
	}

	bool getHomed() const { return _device_homed; }

	/**
	 * @brief sets the space in which the force feedback is computed by the
	 * proxy method instead of using direct force feedback (used in
	 * motion-motion control only). If the proxy feedback space dimension is:
	 * - 0: the direct force feedback is used in all directions
	 * - 1: the force feedback is computed by the proxy method in the direction
	 * of the provided axis, and with direct force feedback in the orthogonal
	 * plane
	 * - 2: the force feedback is computed by using direct feedback in the
	 * direction of the provided axis, and with the proxy method in the
	 * orthogonal plane
	 * - 3: the force feedback is computed by the proxy method in all directions
	 *
	 * @param proxy_feedback_space_dimension dimenson of the proxy feedback
	 * space (between 0 and 3)
	 * @param proxy_or_direct_feedback_axis provided direction (in device base
	 * frame)
	 */
	void parametrizeProxyForceFeedbackSpace(
		const int proxy_feedback_space_dimension,
		const Vector3d& proxy_or_direct_feedback_axis = Vector3d::Zero());

	/**
	 * @brief sets the space in which the moment feedback is computed by the
	 * proxy method to match the space defined by the provided robot sigma
	 * matrix.
	 *
	 * @param robot_sigma_force robot sigma matrix (in robot world frame)
	 */
	void parametrizeProxyForceFeedbackSpaceFromRobotForceSpace(
		const Matrix3d& robot_sigma_force);

	/**
	 * @brief sets the space in which the moment feedback is computed by the
	 * proxy method instead of using direct moment feedback (used in
	 * motion-motion control only). If the proxy feedback space dimension is:
	 * - 0: the direct moment feedback is used in all directions
	 * - 1: the moment feedback is computed by the proxy method in the direction
	 * of the provided axis, and with direct moment feedback in the orthogonal
	 * plane
	 * - 2: the moment feedback is computed by using direct feedback in the
	 * direction of the provided axis, and with the proxy method feedback in the
	 * orthogonal plane
	 * - 3: the moment feedback is computed by the proxy method in all
	 * directions
	 *
	 * @param proxy_feedback_space_dimension dimenson of the proxy feedback
	 * space (between 0 and 3)
	 * @param proxy_or_direct_feedback_axis provided direction (in device base
	 * frame)
	 */
	void parametrizeProxyMomentFeedbackSpace(
		const int proxy_feedback_space_dimension,
		const Vector3d& proxy_or_direct_feedback_axis = Vector3d::Zero());

	/**
	 * @brief sets the space in which the moment feedback is computed by the
	 * proxy method to match the space defined by the provided robot sigma
	 * matrix.
	 *
	 * @param robot_sigma_moment robot sigma matrix (in robot world frame)
	 */
	void parametrizeProxyMomentFeedbackSpaceFromRobotForceSpace(
		const Matrix3d& robot_sigma_moment);

	/**
	 * @brief Returns the sigma matrix that describes the space of proxy force
	 * feedback (in device base frame)
	 *
	 * @return const Matrix3d&
	 */
	const Matrix3d& getSigmaProxyForce() const {
		return _sigma_proxy_force_feedback;
	}

	/**
	 * @brief Returns the sigma matrix that describes the space of direct force
	 * feedback (in device base frame)
	 *
	 * @return Matrix3d
	 */
	Matrix3d getSigmaDirectForceFeedback() const {
		return Matrix3d::Identity() - _sigma_proxy_force_feedback;
	}

	/**
	 * @brief Returns the sigma matrix that describes the space of proxy moment
	 * feedback (in device base frame)
	 *
	 * @return const Matrix3d&
	 */
	const Matrix3d& getSigmaProxyMoment() const {
		return _sigma_proxy_moment_feedback;
	}

	/**
	 * @brief Returns the sigma matrix that describes the space of direct moment
	 * feedback (in device base frame)
	 *
	 * @return Matrix3d
	 */
	Matrix3d getSigmaDirectMomentFeedback() const {
		return Matrix3d::Identity() - _sigma_proxy_moment_feedback;
	}

	/**
	 * @brief Sets the scaling factors for position and orientation (used in
	 * motion-motion control only). The motion of the robot is multiplied by the
	 * scaling factor compared to the motion of the haptic device, so a scaling
	 * factor higher than 1 corresponds to a robot moving more than the haptic
	 * device.)
	 *
	 * @param scaling_factor_pos
	 * @param scaling_factor_ori
	 */
	void setScalingFactors(const double scaling_factor_pos,
						   const double scaling_factor_ori = 1.0);

	double getScalingFactorPos() const { return _scaling_factor_pos; }
	double getScalingFactorOri() const { return _scaling_factor_ori; }

	/**
	 * @brief Set the Reduction Factor for force. The command force
	 * to the haptic device (from direct force feedback) is reduced by this
	 * factors
	 *
	 * @param reduction_factor_force
	 */
	void setReductionFactorForce(const double reduction_factor_force);

	/**
	 * @brief Set the Reduction Factor for force. The command force
	 * to the haptic device (from direct moment feedback) is reduced by this
	 * factors
	 *
	 * @param reduction_factor_moment
	 */
	void setReductionFactorMoment(const double reduction_factor_moment);

	/**
	 * @brief Set the Gains used for all the device control types:
	 * - The homing controller
	 * - The feedback from proxy in the motion-motion controller and worlspace
	 * extension
	 * - The stiffness and damping of the force-motion controller
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

	/**
	 * @brief Enables the plane guidance with the provided plane
	 *
	 * @param plane_origin_point in device base frame
	 * @param plane_normal_direction in device base frame
	 */
	void enablePlaneGuidance(const Vector3d plane_origin_point,
							 const Vector3d plane_normal_direction);

	/**
	 * @brief enables the plane guidance with the plane in memory (either the
	 * last one defined by the user, or the default one being the X-Y plane)
	 *
	 */
	void enablePlaneGuidance() {
		enablePlaneGuidance(_plane_origin_point, _plane_normal_direction);
	}

	void disablePlaneGuidance() { _plane_guidance_enabled = false; }
	bool getPlaneGuidanceEnabled() const { return _plane_guidance_enabled; }

	/**
	 * @brief enables the line guidance with the provided line
	 *
	 * @param line_origin_point in device base frame
	 * @param line_direction in device base frame
	 */
	void enableLineGuidance(const Vector3d line_origin_point,
							const Vector3d line_direction);

	/**
	 * @brief enables the line guidance with the line in memory (either the last
	 * one defined by the user, or the default one being the Z axis)
	 *
	 */
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

	/**
	 * @brief enables the workspace virtual limits with the parameters in memory
	 * (either the last one defined by the user, or the default one being  a
	 * radius of 0.1 and an angle of PI/3)
	 *
	 */
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

	/**
	 * @brief Set the conversion factors from force to position/orientation
	 * difference for the force-motion control type. The defaults are 3e-5 for
	 * position and PI/2000 for orientation.
	 *
	 * @param device_force_to_robot_delta_position
	 * @param device_moment_to_robot_delta_orientation
	 */
	void setAdmittanceFactors(
		const double device_force_to_robot_delta_position,
		const double device_moment_to_robot_delta_orientation);

	/**
	 * @brief Set the Homing Max Velocity for the homing task. The default is
	 * 0.15 m/s for the position and PI rad/s for the orientation
	 *
	 * @param homing_max_linvel
	 * @param homing_max_angvel
	 */
	void setHomingMaxVelocity(const double homing_max_linvel,
							  const double homing_max_angvel);

	/**
	 * @brief Set the Force Deadband for force-motion controller. This is the
	 * minimum force required to start making the robot position change. the
	 * default is 2.0.
	 *
	 * @param force_deadband
	 */
	void setForceDeadbandForceMotionController(const double force_deadband);

	/**
	 * @brief Set the Moment Deadband for force-motion controller. This is the
	 * minimum moment required to start making the robot orientation change. the
	 * default is 0.02.
	 *
	 * @param force_deadband
	 */
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
	bool _reset_robot_linear_offset;
	bool _reset_robot_angular_offset;

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
	HapticControllerOutput _latest_output;

	// latest input
	HapticControllerInput _latest_input;

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
