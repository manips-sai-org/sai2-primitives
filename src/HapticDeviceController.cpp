/*
 * HapticDeviceController.cpp
 *
 *      This controller implements a bilateral haptic teleoperation scheme in
 * open loop. The commands are computed for the haptic device (force feedback)
 * and the controlled robot (desired task). HapticDeviceController includes
 * impedance-type and admittance-type controllers, with plane/line/orientation
 * guidances, and the workspace mapping algorithm.
 *
 *      Authors: Margot Vulliez & Mikael Jorda
 *
 */

#include "HapticDeviceController.h"

#include <stdexcept>

using namespace Eigen;

namespace {
Affine3d AffineFromPosAndRot(const Vector3d& pos, const Matrix3d& rot) {
	Affine3d aff;
	aff.translation() = pos;
	aff.linear() = rot;
	return aff;
}

AngleAxisd orientationErrorAngleAxis(const Matrix3d& desired_orientation,
									 const Matrix3d& current_orientation) {
	// expressed in base frame common to desired and current orientation
	return AngleAxisd(current_orientation * desired_orientation.transpose());
}

}  // namespace

namespace Sai2Primitives {

////////////////////////////////////////////////////////////////////////////////////////////////////
//// Constructor, Destructor and Initialization of the haptic controllers
////////////////////////////////////////////////////////////////////////////////////////////////////

HapticDeviceController::HapticDeviceController(
	const DeviceLimits& device_limits,
	const Eigen::Affine3d& robot_initial_pose,
	const Eigen::Affine3d& device_home_pose,
	const Eigen::Matrix3d& device_base_rotation_in_world)
	: _device_limits(device_limits),
	  _center_position_robot(robot_initial_pose.translation()),
	  _center_rotation_robot(robot_initial_pose.linear()),
	  _R_world_device(device_base_rotation_in_world),
	  _home_position_device(device_home_pose.translation()),
	  _home_rotation_device(device_home_pose.linear()) {
	_reset_robot_offset = false;

	_previous_output.robot_goal_position = _center_position_robot;
	_previous_output.robot_goal_orientation = _center_rotation_robot;

	// Initialize homing task
	_device_homed = false;

	_haptic_control_type = HapticControlType::DETACHED;

	// Initialize scaling factors (can be set through setScalingFactors())
	_scaling_factor_pos = 1.0;
	_scaling_factor_ori = 1.0;

	// Initialize position controller parameters
	_kp_haptic_pos = 0.8 * _device_limits.max_linear_stiffness;
	_kp_haptic_ori = 0.8 * _device_limits.max_angular_stiffness;
	_kv_haptic_pos = 2.0 * sqrt(_kp_haptic_pos);
	_kv_haptic_ori = 2.0 * sqrt(_kp_haptic_ori);
	if (_kv_haptic_pos > _device_limits.max_linear_damping) {
		_kv_haptic_pos = _device_limits.max_linear_damping;
	}
	if (_kv_haptic_ori > _device_limits.max_angular_damping) {
		_kv_haptic_ori = _device_limits.max_angular_damping;
	}

	_homing_max_linvel = 0.15;
	_homing_max_angvel = M_PI;

	_reduction_factor_force = 1.0;
	_reduction_factor_moment = 1.0;

	// Initialiaze force feedback computation mode
	_compute_haptic_feedback_from_proxy = true;
	_send_haptic_feedback = true;

	// Device workspace virtual limits
	_enable_workspace_virtual_limit = false;

	// Initialize haptic guidance parameters
	_enable_plane_guidance = false;
	_enable_line_guidance = false;
	_kp_guidance_pos = _kp_haptic_pos;
	_kp_guidance_ori = _kp_haptic_ori;
	_kv_guidance_pos = _kv_haptic_pos;
	_kv_guidance_ori = _kv_haptic_ori;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Haptic controllers for all the sypported controler types
////////////////////////////////////////////////////////////////////////////////////////////////////

HapticControllerOtuput HapticDeviceController::computeHapticControl(
	const HapticControllerInput& input, const bool verbose) {
	HapticControllerOtuput output;
	switch (_haptic_control_type) {
		case HapticControlType::DETACHED:
			output = computeDetachedControl(input);
			break;
		case HapticControlType::HOMING:
			output = computeHomingControl(input);
			break;
		case HapticControlType::MOTION_MOTION:
			output = computeMotionMotionControl(input);
			break;
		// case HapticControlType::FORCE_MOTION:
		// 	return computeForceMotionControl(input);
		// break;
		// case HapticControlType::HYBRID_WITH_PROXY:
		// 	return computeHybridWithProxyControl(input);
		// break;
		// case HapticControlType::WORKSPACE_EXTENSION:
		// 	return computeWorkspaceExtensionControl(input);
		// break;
		default:
			throw std::runtime_error("Unimplemented haptic control type");
			break;
	}
	validateOutput(output, verbose);
	_previous_output = output;
	return output;
}

void HapticDeviceController::validateOutput(HapticControllerOtuput& output,
											const bool verbose) {
	if (output.device_command_force.norm() > _device_limits.max_force) {
		if (verbose) {
			std::cout << "Warning: device feedback force norm is too high. "
						 "Saturating to "
					  << _device_limits.max_force << std::endl;
		}
		output.device_command_force *=
			_device_limits.max_force / output.device_command_force.norm();
	}
	if (output.device_command_moment.norm() > _device_limits.max_torque) {
		if (verbose) {
			std::cout << "Warning: device feedback moment norm is too high. "
						 "Saturating to "
					  << _device_limits.max_torque << std::endl;
		}
		output.device_command_moment *=
			_device_limits.max_torque / output.device_command_moment.norm();
	}
}

HapticControllerOtuput HapticDeviceController::computeDetachedControl(
	const HapticControllerInput& input) {
	HapticControllerOtuput output = _previous_output;

	applyLineGuidanceForce(output.device_command_force, input);
	applyPlaneGuidanceForce(output.device_command_force, input);
	applyWorkspaceVirtualLimits(output.device_command_force,
								output.device_command_moment, input);

	return output;
}

HapticControllerOtuput HapticDeviceController::computeHomingControl(
	const HapticControllerInput& input) {
	_device_homed = false;
	HapticControllerOtuput output;
	output.robot_goal_position = _previous_output.robot_goal_position;
	output.robot_goal_orientation = _previous_output.robot_goal_orientation;

	if (_kv_haptic_pos > 0) {
		Vector3d desired_velocity =
			-_kp_haptic_pos / _kv_haptic_pos *
			(input.device_position - _home_position_device);
		if (desired_velocity.norm() > _homing_max_linvel) {
			desired_velocity *= _homing_max_linvel / desired_velocity.norm();
		}
		output.device_command_force =
			-_kv_haptic_pos * (input.device_linear_velocity - desired_velocity);
	}

	Vector3d orientation_error = Vector3d::Zero();
	if (_kv_haptic_ori > 0) {
		AngleAxisd orientation_error_aa = orientationErrorAngleAxis(
			_home_rotation_device, input.device_orientation);
		orientation_error =
			orientation_error_aa.angle() * orientation_error_aa.axis();
		Vector3d desired_velocity =
			-_kp_haptic_ori / _kv_haptic_ori * orientation_error;
		if (desired_velocity.norm() > _homing_max_angvel) {
			desired_velocity *= _homing_max_angvel / desired_velocity.norm();
		}
		output.device_command_moment =
			-_kv_haptic_ori *
			(input.device_angular_velocity - desired_velocity);
	}

	if ((input.device_position - _home_position_device).norm() < 0.001 &&
		orientation_error.norm() < 0.01 &&
		input.device_linear_velocity.norm() < 0.001 &&
		input.device_angular_velocity.norm() < 0.01) {
		_device_homed = true;
	}

	return output;
}

HapticControllerOtuput HapticDeviceController::computeMotionMotionControl(
	const HapticControllerInput& input) {
	HapticControllerOtuput output;
	output.robot_goal_position = _previous_output.robot_goal_position;
	output.robot_goal_orientation = _previous_output.robot_goal_orientation;

	// Compute robot goal position
	Vector3d device_home_to_current_position =
		input.device_position - _home_position_device;	// in device base frame

	if (_reset_robot_offset) {
		_center_position_robot =
			input.robot_position - _scaling_factor_pos * _R_world_device *
									   device_home_to_current_position;
	}

	output.robot_goal_position =
		_center_position_robot +
		_scaling_factor_pos * _R_world_device * device_home_to_current_position;

	// compute robot goal orientation
	// Matrix3d device_home_to_current_orientation =
	// 	input.device_orientation *
	// 	_home_rotation_device.transpose();	// in device base frame
	// AngleAxisd device_home_to_current_orientation_aa(
	// 	device_home_to_current_orientation);
	if (_enable_orientation_teleoperation) {
		AngleAxisd device_home_to_current_orientation_aa =
			orientationErrorAngleAxis(_home_rotation_device,
									  input.device_orientation);
		AngleAxisd scaled_device_home_to_current_orientation_aa(
			_scaling_factor_ori * device_home_to_current_orientation_aa.angle(),
			device_home_to_current_orientation_aa.axis());

		if (_reset_robot_offset) {
			_center_rotation_robot =
				_R_world_device *
				scaled_device_home_to_current_orientation_aa.toRotationMatrix()
					.transpose() *
				_R_world_device.transpose() * input.robot_orientation;
		}

		output.robot_goal_orientation =
			_R_world_device *
			scaled_device_home_to_current_orientation_aa.toRotationMatrix() *
			_R_world_device.transpose() * _center_rotation_robot;
	}

	_reset_robot_offset = false;

	if (!_send_haptic_feedback) {
		return output;
	}

	// Compute the force feedback in robot frame
	Vector3d haptic_forces_robot_space = input.robot_sensed_force;
	Vector3d haptic_moments_robot_space = input.robot_sensed_moment;

	if (_compute_haptic_feedback_from_proxy) {
		// Transfer device velocity to world frame
		Vector3d device_linear_velocity_in_robot_workspace =
			_scaling_factor_pos * _R_world_device *
			input.device_linear_velocity;
		Vector3d device_angular_velocity_in_robot_workspace =
			_scaling_factor_ori * _R_world_device *
			input.device_angular_velocity;

		// Evaluate the task force through stiffness proxy
		haptic_forces_robot_space =
			_kp_haptic_pos *
				(input.robot_position - output.robot_goal_position) -
			_kv_haptic_pos * (device_linear_velocity_in_robot_workspace -
							  input.robot_linear_velocity);

		// Compute the orientation error
		Vector3d orientation_error = Sai2Model::orientationError(
			output.robot_goal_orientation, input.robot_orientation);
		// Evaluate task torque
		haptic_moments_robot_space =
			_kp_haptic_ori * orientation_error -
			_kv_haptic_ori * (device_angular_velocity_in_robot_workspace -
							  input.robot_angular_velocity);
	}

	// scale and rotate to device frame
	Vector3d haptic_force_feedback =
		_R_world_device.transpose() * _reduction_factor_force /
		_scaling_factor_pos * haptic_forces_robot_space;
	Vector3d haptic_moment_feedback =
		_R_world_device.transpose() * _reduction_factor_moment /
		_scaling_factor_ori * haptic_moments_robot_space;
	// Apply haptic guidances
	applyPlaneGuidanceForce(haptic_force_feedback, input);
	applyLineGuidanceForce(haptic_force_feedback, input);
	// if not controlling the orientation, set the moment to zero
	if (!_enable_orientation_teleoperation) {
		haptic_moment_feedback.setZero();
	}

	applyWorkspaceVirtualLimits(haptic_force_feedback, haptic_moment_feedback,
								input);

	output.device_command_force = haptic_force_feedback;
	output.device_command_moment = haptic_moment_feedback;

	return output;
}

void HapticDeviceController::applyPlaneGuidanceForce(
	Vector3d& haptic_force_to_update, const HapticControllerInput& input) {
	if (!_enable_plane_guidance) {
		return;
	}

	// vector from the plane's origin to the current position
	Eigen::Vector3d plane_to_point_vec;
	plane_to_point_vec = input.device_position - _plane_origin_point;

	// calculate the normal distance from the plane to the current position
	double distance_to_plane;
	distance_to_plane = plane_to_point_vec.dot(_plane_normal_vec);

	// current position projected onto the plane
	Eigen::Vector3d projected_current_position;
	projected_current_position =
		input.device_position - distance_to_plane * _plane_normal_vec;

	// projected device velocity on the normal vector
	Eigen::Vector3d projected_current_velocity;
	projected_current_velocity =
		input.device_linear_velocity.dot(_plane_normal_vec) * _plane_normal_vec;

	// calculate the guidance force
	Vector3d guidance_force_plane =
		-_kp_guidance_pos *
			(input.device_position - projected_current_position) -
		_kv_guidance_pos * projected_current_velocity;

	// update the haptic force
	haptic_force_to_update +=
		guidance_force_plane -
		haptic_force_to_update.dot(_plane_normal_vec) * _plane_normal_vec;
}

void HapticDeviceController::applyLineGuidanceForce(
	Vector3d& haptic_force_to_update, const HapticControllerInput& input) {
	if (!_enable_line_guidance) {
		return;
	}

	Eigen::Vector3d line_to_point_vec;
	double distance_along_line;
	Eigen::Vector3d closest_point_vec;
	Eigen::Vector3d projected_current_position;
	Eigen::Vector3d line_normal_vec;

	// vector from first point on line to current position
	line_to_point_vec = input.device_position - _line_first_point;

	// distance from the first point to the current position projected onto the
	// line
	distance_along_line = line_to_point_vec.dot(_guidance_line_vec);

	// vector to the projected point onto the line from the first point
	closest_point_vec = distance_along_line * _guidance_line_vec;

	// projected position on line in world frame
	projected_current_position = _line_first_point + closest_point_vec;

	// Normal vector to the line from current position
	line_normal_vec = input.device_position - projected_current_position;
	line_normal_vec = line_normal_vec / line_normal_vec.norm();

	// projected device velocity on the normal vector
	Eigen::Vector3d projected_current_velocity;
	projected_current_velocity =
		input.device_linear_velocity.dot(line_normal_vec) * line_normal_vec;

	// compute the force
	Vector3d guidance_force_line =
		-_kp_guidance_pos *
			(input.device_position - projected_current_position) -
		_kv_guidance_pos * projected_current_velocity;

	// update the haptic force
	Vector3d line_vector = (_line_first_point - _line_second_point) /
						   (_line_first_point - _line_second_point).norm();
	haptic_force_to_update =
		guidance_force_line +
		haptic_force_to_update.dot(line_vector) * line_vector;
}

void HapticDeviceController::applyWorkspaceVirtualLimits(
	Vector3d& haptic_force_to_update, Vector3d& haptic_moment_to_update,
	const HapticControllerInput& input) {
	if (!_enable_workspace_virtual_limit) {
		return;
	}

	// Add virtual forces according to the device operational space limits
	Vector3d force_virtual = Vector3d::Zero();
	Vector3d torque_virtual = Vector3d::Zero();

	Vector3d device_home_to_current_position =
		input.device_position - _home_position_device;
	if (device_home_to_current_position.norm() >=
		_device_workspace_radius_limit) {
		force_virtual = -_kp_guidance_pos *
							(device_home_to_current_position.norm() -
							 _device_workspace_radius_limit) *
							device_home_to_current_position /
							device_home_to_current_position.norm() -
						_kv_guidance_pos * input.device_linear_velocity;
	}
	haptic_force_to_update += force_virtual;

	AngleAxisd device_home_to_current_orientation_aa =
		orientationErrorAngleAxis(_home_rotation_device,
								  input.device_orientation);
	if (device_home_to_current_orientation_aa.angle() >=
		_device_workspace_angle_limit) {
		torque_virtual = -_kp_guidance_ori *
							 (device_home_to_current_orientation_aa.angle() -
							  _device_workspace_angle_limit) *
							 device_home_to_current_orientation_aa.axis() -
						 _kv_guidance_ori * input.device_angular_velocity;
	}
	haptic_moment_to_update += torque_virtual;
}

///////////////////////////////////////////////////////////////////////////////////
// Parameter setting methods
///////////////////////////////////////////////////////////////////////////////////

void HapticDeviceController::setEnableOrientationTeleoperation(
	const bool enable) {
	_enable_orientation_teleoperation = enable;
	if (!_enable_orientation_teleoperation) {
		resetRobotOffset();
	}
}

void HapticDeviceController::setScalingFactors(
	const double scaling_factor_pos, const double scaling_factor_ori) {
	if (scaling_factor_pos <= 0 || scaling_factor_ori <= 0) {
		throw std::runtime_error(
			"Scaling factors must be positive in "
			"HapticDeviceController::setScalingFactors");
	}
	_scaling_factor_pos = scaling_factor_pos;
	_scaling_factor_ori = scaling_factor_ori;
}

void HapticDeviceController::setReductionFactorForceMoment(
	const double reduction_factor_force, const double reduction_factor_moment) {
	if (reduction_factor_force < 0 || reduction_factor_moment < 0 ||
		reduction_factor_force > 1 || reduction_factor_moment > 1) {
		throw std::runtime_error(
			"Reduction factors must be between 0 and 1 in "
			"HapticDeviceController::setReductionFactorForceMoment");
	}
}

void HapticDeviceController::setDeviceControlGains(const double kp_pos,
												   const double kv_pos) {
	if (kp_pos < 0 || kv_pos < 0) {
		throw std::runtime_error(
			"Device control gains must be positive in "
			"HapticDeviceController::setDeviceControlGains");
	}
	_kp_haptic_pos = kp_pos;
	_kv_haptic_pos = kv_pos;
}

void HapticDeviceController::setDeviceControlGains(const double kp_pos,
												   const double kv_pos,
												   const double kp_ori,
												   const double kv_ori) {
	if (kp_pos < 0 || kv_pos < 0 || kp_ori < 0 || kv_ori < 0) {
		throw std::runtime_error(
			"Device control gains must be positive in "
			"HapticDeviceController::setDeviceControlGains");
	}
	_kp_haptic_pos = kp_pos;
	_kv_haptic_pos = kv_pos;
	_kp_haptic_ori = kp_ori;
	_kv_haptic_ori = kv_ori;
}

void HapticDeviceController::setHapticGuidanceGains(
	const double kp_guidance_pos, const double kv_guidance_pos) {
	if (kp_guidance_pos < 0 || kv_guidance_pos < 0) {
		throw std::runtime_error(
			"Guidance gains must be positive in "
			"HapticDeviceController::setHapticGuidanceGains");
	}
	_kp_guidance_pos = kp_guidance_pos;
	_kv_guidance_pos = kv_guidance_pos;
}

void HapticDeviceController::setHapticGuidanceGains(
	const double kp_guidance_pos, const double kv_guidance_pos,
	const double kp_guidance_ori, const double kv_guidance_ori) {
	if (kp_guidance_pos < 0 || kv_guidance_pos < 0 || kp_guidance_ori < 0 ||
		kv_guidance_ori < 0) {
		throw std::runtime_error(
			"Guidance gains must be positive in "
			"HapticDeviceController::setHapticGuidanceGains");
	}
}

void HapticDeviceController::enablePlaneGuidance(
	const Eigen::Vector3d plane_origin_point,
	const Eigen::Vector3d plane_normal_vec) {
	if (plane_normal_vec.norm() < 0.001) {
		throw std::runtime_error(
			"Plane normal vector must be non-zero in "
			"HapticDeviceController::enablePlaneGuidance");
	}
	if (_enable_line_guidance) {
		cout << "Warning: plane guidance is enabled while line guidance is "
				"already enabled. Disabling line guidance."
			 << endl;
		disableLineGuidance();
	}
	_enable_plane_guidance = true;
	_plane_origin_point = plane_origin_point;
	_plane_normal_vec = plane_normal_vec / plane_normal_vec.norm();
}

void HapticDeviceController::enableLineGuidance(
	const Eigen::Vector3d line_first_point,
	const Eigen::Vector3d line_second_point) {
	if (_enable_plane_guidance) {
		cout << "Warning: line guidance is enabled while plane guidance is "
				"already enabled. Disabling plane guidance."
			 << endl;
		disablePlaneGuidance();
	}
	_enable_line_guidance = true;

	// set vector from one point to the other
	_line_first_point = line_first_point;
	_line_second_point = line_second_point;

	_guidance_line_vec = line_second_point - line_first_point;
	_guidance_line_vec = _guidance_line_vec / _guidance_line_vec.norm();
}

void HapticDeviceController::enableHapticWorkspaceVirtualLimits(
	double device_workspace_radius_limit, double device_workspace_angle_limit) {
	if (device_workspace_radius_limit < 0 || device_workspace_angle_limit < 0) {
		throw std::runtime_error(
			"Workspace virtual limits must be positive in "
			"HapticDeviceController::setHapticWorkspaceVirtualLimits");
	}
	_enable_workspace_virtual_limit = true;
	_device_workspace_radius_limit = device_workspace_radius_limit;
	_device_workspace_angle_limit = device_workspace_angle_limit;
}

} /* namespace Sai2Primitives */
