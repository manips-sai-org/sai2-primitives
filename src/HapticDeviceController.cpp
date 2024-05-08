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
AngleAxisd orientationDiffAngleAxis(const Matrix3d& goal_orientation,
									const Matrix3d& current_orientation,
									const double scaling_factor = 1.0) {
	if (scaling_factor < 0 || scaling_factor > 1) {
		throw std::runtime_error(
			"Scaling factor must be between 0 and 1 in "
			"scaledOrientationErrorFromAngleAxis");
	}

	// expressed in base frame common to goal and current orientation
	AngleAxisd current_orientation_from_goal_orientation_aa(
		current_orientation * goal_orientation.transpose());

	return AngleAxisd(
		scaling_factor * current_orientation_from_goal_orientation_aa.angle(),
		current_orientation_from_goal_orientation_aa.axis());
}

Vector3d angleAxisToVector(const AngleAxisd& aa) {
	return aa.angle() * aa.axis();
}

Vector3d projectAlongDirection(const Vector3d& vector_to_project,
							   const Vector3d& direction) {
	if (direction.norm() <= 0.001) {
		throw std::runtime_error(
			"direction should be a non zero vector in projectAlongDirection");
	}
	return direction.dot(vector_to_project) * direction /
		   direction.squaredNorm();
}

double computeInterpolationCoeff(const double x, const double x0,
								 const double x1) {
	if (x0 > x1) {
		throw std::runtime_error(
			"x0 should be smaller than x1 in compute_interpolation_coeff");
	}
	if (x <= x0) {
		return 0;
	}
	if (x >= x1) {
		return 1;
	}
	return (x - x0) / (x1 - x0);
}

}  // namespace

namespace Sai2Primitives {

////////////////////////////////////////////////////////////////////////////////////////////////////
//// Constructor, Destructor and Initialization of the haptic controllers
////////////////////////////////////////////////////////////////////////////////////////////////////

HapticDeviceController::HapticDeviceController(
	const DeviceLimits& device_limits, const Affine3d& robot_initial_pose,
	const Affine3d& device_home_pose,
	const Matrix3d& device_base_rotation_in_world)
	: _device_limits(device_limits),
	  _robot_center_pose(robot_initial_pose),
	  _R_world_device(device_base_rotation_in_world),
	  _device_home_pose(device_home_pose) {
	_reset_robot_linear_offset = false;
	_reset_robot_angular_offset = false;

	_latest_output.robot_goal_position = _robot_center_pose.translation();
	_latest_output.robot_goal_orientation = _robot_center_pose.rotation();

	// Initialize homing task
	_device_homed = false;
	_haptic_control_type = DefaultParameters::haptic_control_type;

	// Initialize scaling factors (can be set through setScalingFactors())
	_scaling_factor_pos = DefaultParameters::scaling_factor_pos;
	_scaling_factor_ori = DefaultParameters::scaling_factor_ori;

	// Initialize position controller parameters
	_kp_haptic_pos = 0.5 * _device_limits.max_linear_stiffness;
	_kp_haptic_ori = 0.5 * _device_limits.max_angular_stiffness;
	_kv_haptic_pos = 2.0 * sqrt(_kp_haptic_pos);
	_kv_haptic_ori = 2.0 * sqrt(_kp_haptic_ori);
	if (_kv_haptic_pos > 0.5 * _device_limits.max_linear_damping) {
		_kv_haptic_pos = 0.5 * _device_limits.max_linear_damping;
	}
	if (_kv_haptic_ori > 0.5 * _device_limits.max_angular_damping) {
		_kv_haptic_ori = 0.5 * _device_limits.max_angular_damping;
	}

	_homing_max_linvel = DefaultParameters::homing_max_linvel;
	_homing_max_angvel = DefaultParameters::homing_max_angvel;

	_reduction_factor_force = DefaultParameters::reduction_factor_force;
	_reduction_factor_moment = DefaultParameters::reduction_factor_moment;

	_device_force_to_robot_delta_position =
		DefaultParameters::device_force_to_robot_delta_position;
	_device_moment_to_robot_delta_orientation =
		DefaultParameters::device_moment_to_robot_delta_orientation;
	_force_deadband = DefaultParameters::force_deadband;
	_moment_deadband = DefaultParameters::moment_deadband;

	// Initialiaze force feedback space
	_sigma_proxy_force_feedback.setZero();
	_sigma_proxy_moment_feedback.setZero();

	// Device workspace virtual limits
	_device_workspace_virtual_limits_enabled = false;

	// Initialize haptic guidance parameters
	_plane_guidance_enabled = false;
	_line_guidance_enabled = false;
	_kp_guidance_pos = 1.2 * _kp_haptic_pos;
	_kp_guidance_ori = 1.2 * _kp_haptic_ori;
	_kv_guidance_pos = _kv_haptic_pos;
	_kv_guidance_ori = _kv_haptic_ori;

	_plane_origin_point = _device_home_pose.translation();
	_plane_normal_direction = Vector3d::UnitZ();
	_line_origin_point = _device_home_pose.translation();
	_line_direction = Vector3d::UnitZ();

	_device_workspace_radius_limit =
		DefaultParameters::device_workspace_radius_limit;
	_device_workspace_angle_limit =
		DefaultParameters::device_workspace_angle_limit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Haptic controllers for all the sypported controler types
////////////////////////////////////////////////////////////////////////////////////////////////////

HapticControllerOtuput HapticDeviceController::computeHapticControl(
	const HapticControllerInput& input, const bool verbose) {
	HapticControllerOtuput output;
	_latest_input = input;
	switch (_haptic_control_type) {
		case HapticControlType::CLUTCH:
			output = computeClutchControl(input);
			break;
		case HapticControlType::HOMING:
			output = computeHomingControl(input);
			break;
		case HapticControlType::MOTION_MOTION:
			output = computeMotionMotionControl(input);
			break;
		case HapticControlType::FORCE_MOTION:
			output = computeForceMotionControl(input);
			break;
		default:
			throw std::runtime_error("Unimplemented haptic control type");
			break;
	}
	validateOutput(output, verbose);
	_latest_output = output;
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

HapticControllerOtuput HapticDeviceController::computeClutchControl(
	const HapticControllerInput& input) {
	HapticControllerOtuput output;
	output.robot_goal_position = _latest_output.robot_goal_position;
	output.robot_goal_orientation = _latest_output.robot_goal_orientation;

	applyWorkspaceVirtualLimitsForceMoment(input, output);
	applyLineGuidanceForce(output.device_command_force, input, false);
	applyPlaneGuidanceForce(output.device_command_force, input, false);

	return output;
}

HapticControllerOtuput HapticDeviceController::computeHomingControl(
	const HapticControllerInput& input) {
	_device_homed = false;
	HapticControllerOtuput output;
	output.robot_goal_position = _latest_output.robot_goal_position;
	output.robot_goal_orientation = _latest_output.robot_goal_orientation;

	if (_kv_haptic_pos > 0) {
		Vector3d desired_velocity =
			-_kp_haptic_pos / _kv_haptic_pos *
			(input.device_position - _device_home_pose.translation());
		if (desired_velocity.norm() > _homing_max_linvel) {
			desired_velocity *= _homing_max_linvel / desired_velocity.norm();
		}
		output.device_command_force =
			-_kv_haptic_pos * (input.device_linear_velocity - desired_velocity);
	}

	Vector3d orientation_error = Vector3d::Zero();
	if (_kv_haptic_ori > 0) {
		orientation_error = angleAxisToVector(orientationDiffAngleAxis(
			_device_home_pose.rotation(), input.device_orientation));
		Vector3d desired_velocity =
			-_kp_haptic_ori / _kv_haptic_ori * orientation_error;
		if (desired_velocity.norm() > _homing_max_angvel) {
			desired_velocity *= _homing_max_angvel / desired_velocity.norm();
		}
		output.device_command_moment =
			-_kv_haptic_ori *
			(input.device_angular_velocity - desired_velocity);
	}

	if ((input.device_position - _device_home_pose.translation()).norm() <
			0.001 &&
		input.device_linear_velocity.norm() < 0.01 &&
		(!_orientation_teleop_enabled ||
		 orientation_error.norm() < 0.01 &&
			 input.device_angular_velocity.norm() < 0.1)) {
		_device_homed = true;
	}

	return output;
}

HapticControllerOtuput HapticDeviceController::computeMotionMotionControl(
	const HapticControllerInput& input) {
	HapticControllerOtuput output;
	output.robot_goal_position = _latest_output.robot_goal_position;
	output.robot_goal_orientation = _latest_output.robot_goal_orientation;

	// position and orientation control
	motionMotionControlPosition(input, output);
	motionMotionControlOrientation(input, output);

	// consume reset robot offsets
	_reset_robot_linear_offset = false;
	_reset_robot_angular_offset = false;

	// Apply haptic guidances
	applyWorkspaceVirtualLimitsForceMoment(input, output);
	applyLineGuidanceForce(output.device_command_force, input, false);
	applyPlaneGuidanceForce(output.device_command_force, input, false);

	return output;
}

void HapticDeviceController::motionMotionControlPosition(
	const HapticControllerInput& input, HapticControllerOtuput& output) {
	// Compute robot goal position
	Vector3d device_home_to_current_position =
		input.device_position -
		_device_home_pose.translation();  // in device base frame

	if (_reset_robot_linear_offset) {
		_robot_center_pose.translation() =
			input.robot_position - _scaling_factor_pos * _R_world_device *
									   device_home_to_current_position;
	}

	// compute robot goal position
	if (_device_workspace_virtual_limits_enabled &&
		device_home_to_current_position.norm() >
			_device_workspace_radius_limit) {
		device_home_to_current_position =
			_device_workspace_radius_limit * device_home_to_current_position /
			device_home_to_current_position.norm();
	}
	output.robot_goal_position =
		_robot_center_pose.translation() +
		_scaling_factor_pos * _R_world_device * device_home_to_current_position;

	if (_plane_guidance_enabled) {
		// saturate goal position to be in the plane
		Vector3d plane_origin_robot_frame =
			_robot_center_pose.translation() +
			_R_world_device * _scaling_factor_pos *
				(_plane_origin_point - _device_home_pose.translation());
		Vector3d plane_normal_robot_frame =
			_R_world_device * _plane_normal_direction;

		output.robot_goal_position -= projectAlongDirection(
			output.robot_goal_position - plane_origin_robot_frame,
			plane_normal_robot_frame);
	} else if (_line_guidance_enabled) {
		// saturate goal position to be along the line
		Vector3d line_origin_robot_frame =
			_robot_center_pose.translation() +
			_R_world_device * _scaling_factor_pos *
				(_line_origin_point - _device_home_pose.translation());
		Vector3d line_direction_robot_frame = _R_world_device * _line_direction;

		output.robot_goal_position =
			line_origin_robot_frame +
			projectAlongDirection(
				output.robot_goal_position - line_origin_robot_frame,
				line_direction_robot_frame);
	}

	// Compute the force feedback in robot frame
	Vector3d haptic_forces_robot_space_direct_feedback =
		-input.robot_sensed_force;

	// scale and rotate to device frame
	Vector3d haptic_force_direct_feedback =
		_R_world_device.transpose() * _reduction_factor_force /
		_scaling_factor_pos * haptic_forces_robot_space_direct_feedback;

	// add damping to the direct force feedback
	if (haptic_force_direct_feedback.norm() > 1e-2) {
		haptic_force_direct_feedback -=
			computeKvPosVariableDamping(input.device_linear_velocity.norm()) *
			input.device_linear_velocity;
	}

	// Find proxy position and velocity
	Vector3d proxy_position =
		_device_home_pose.translation() +
		_R_world_device.transpose() / _scaling_factor_pos *
			(input.robot_position - _robot_center_pose.translation());
	Vector3d proxy_linear_velocity = _R_world_device.transpose() *
									 input.robot_linear_velocity /
									 _scaling_factor_pos;

	// Evaluate the feedback force through proxy
	Vector3d haptic_forces_proxy =
		-_kp_haptic_pos * (input.device_position - proxy_position) -
		_kv_haptic_pos * (input.device_linear_velocity - proxy_linear_velocity);

	// final haptic command force
	Matrix3d sigma_direct_force_feedback =
		Matrix3d::Identity() - _sigma_proxy_force_feedback;
	output.device_command_force =
		sigma_direct_force_feedback * haptic_force_direct_feedback +
		_sigma_proxy_force_feedback * haptic_forces_proxy;
}

void HapticDeviceController::motionMotionControlOrientation(
	const HapticControllerInput& input, HapticControllerOtuput& output) {
	if (!_orientation_teleop_enabled) {
		return;
	}

	// compute robot goal orientation
	AngleAxisd scaled_device_home_to_current_orientation_aa =
		orientationDiffAngleAxis(_device_home_pose.rotation(),
								 input.device_orientation, _scaling_factor_ori);

	double scaled_angle_limit =
		_scaling_factor_ori * _device_workspace_angle_limit;
	if (_device_workspace_virtual_limits_enabled &&
		scaled_device_home_to_current_orientation_aa.angle() >
			scaled_angle_limit) {
		scaled_device_home_to_current_orientation_aa.angle() =
			scaled_angle_limit;
	}

	if (_reset_robot_angular_offset) {
		_robot_center_pose.linear() =
			_R_world_device *
			scaled_device_home_to_current_orientation_aa.toRotationMatrix()
				.transpose() *
			_R_world_device.transpose() * input.robot_orientation;
	}

	output.robot_goal_orientation =
		_R_world_device *
		scaled_device_home_to_current_orientation_aa.toRotationMatrix() *
		_R_world_device.transpose() * _robot_center_pose.rotation();

	// Compute the moment feedback in robot frame
	Vector3d haptic_moments_robot_space_direct_feedback =
		-input.robot_sensed_moment;

	// scale and rotate to device frame
	Vector3d haptic_moment_direct_feedback =
		_R_world_device.transpose() * _reduction_factor_moment /
		_scaling_factor_ori * haptic_moments_robot_space_direct_feedback;

	// add damping to the direct force feedback
	if (haptic_moment_direct_feedback.norm() > 1e-2) {
		haptic_moment_direct_feedback -=
			computeKvOriVariableDamping(input.device_angular_velocity.norm()) *
			input.device_angular_velocity;
	}

	// Find proxy orientation and angular velocity
	AngleAxisd scaled_robot_orientation_from_center_aa =
		orientationDiffAngleAxis(_robot_center_pose.rotation(),
								 input.robot_orientation,
								 1.0 / _scaling_factor_ori);
	Matrix3d proxy_orientation =
		_R_world_device.transpose() *
		scaled_robot_orientation_from_center_aa.toRotationMatrix() *
		_R_world_device * _device_home_pose.rotation();

	Vector3d proxy_angular_velocity = _R_world_device.transpose() *
									  input.robot_angular_velocity /
									  _scaling_factor_ori;

	// Evaluate the feedback moment through proxy
	Vector3d orientation_error_from_proxy = angleAxisToVector(
		orientationDiffAngleAxis(proxy_orientation, input.device_orientation));
	Vector3d haptic_moments_proxy =
		-_kp_haptic_ori * orientation_error_from_proxy -
		_kv_haptic_ori *
			(input.device_angular_velocity - proxy_angular_velocity);

	// final haptic command moment
	Matrix3d sigma_direct_moment_feedback =
		Matrix3d::Identity() - _sigma_proxy_moment_feedback;
	output.device_command_moment =
		sigma_direct_moment_feedback * haptic_moment_direct_feedback +
		_sigma_proxy_moment_feedback * haptic_moments_proxy;
}

HapticControllerOtuput HapticDeviceController::computeForceMotionControl(
	const HapticControllerInput& input) {
	HapticControllerOtuput output;
	output.robot_goal_position = _latest_output.robot_goal_position;
	output.robot_goal_orientation = _latest_output.robot_goal_orientation;

	// compute force from stiffness/damping field
	Vector3d device_force =
		-_kp_haptic_pos *
			(input.device_position - _device_home_pose.translation()) -
		_kv_haptic_pos * input.device_linear_velocity;

	// compute robot goal position
	Vector3d projected_device_force = device_force;
	if (_plane_guidance_enabled) {
		projected_device_force =
			device_force -
			projectAlongDirection(device_force, _plane_normal_direction);
	} else if (_line_guidance_enabled) {
		projected_device_force =
			projectAlongDirection(device_force, _line_direction);
	}

	if (projected_device_force.norm() < _force_deadband) {
		projected_device_force.setZero();
	} else {
		projected_device_force -=
			_force_deadband * projected_device_force.normalized();
	}

	Vector3d robot_goal_position_increment =
		_device_force_to_robot_delta_position * _R_world_device *
		projected_device_force;
	output.robot_goal_position -= robot_goal_position_increment;

	applyLineGuidanceForce(device_force, input, true);
	applyPlaneGuidanceForce(device_force, input, true);

	output.device_command_force = device_force;

	// orientation control
	// moments from stiffness/damping field
	AngleAxisd home_to_current_orientation = orientationDiffAngleAxis(
		_device_home_pose.rotation(), input.device_orientation);
	Vector3d device_moment =
		-_kp_haptic_ori * angleAxisToVector(home_to_current_orientation) -
		_kv_haptic_ori * input.device_angular_velocity;

	output.device_command_moment = device_moment;

	if (_orientation_teleop_enabled) {
		Vector3d device_moment_after_deadband = device_moment;
		if (device_moment_after_deadband.norm() < _moment_deadband) {
			device_moment_after_deadband.setZero();
		} else {
			device_moment_after_deadband -=
				_moment_deadband * device_moment_after_deadband.normalized();
		}
		// compute robot goal orientation
		Matrix3d robot_goal_orientation_increment = Matrix3d::Identity();
		if (device_moment_after_deadband.norm() > 1e-3) {
			robot_goal_orientation_increment =
				AngleAxisd(
					-_device_moment_to_robot_delta_orientation *
						device_moment_after_deadband.norm(),
					_R_world_device * device_moment_after_deadband.normalized())
					.toRotationMatrix();
		}
		output.robot_goal_orientation =
			robot_goal_orientation_increment * output.robot_goal_orientation;
	}

	return output;
}

void HapticDeviceController::applyPlaneGuidanceForce(
	Vector3d& force_to_update, const HapticControllerInput& input,
	const bool use_device_home_as_origin) {
	if (!_plane_guidance_enabled) {
		return;
	}

	Vector3d reference_point = use_device_home_as_origin
								   ? _device_home_pose.translation()
								   : _plane_origin_point;

	// apply a spring damper system to bring the haptic device to the origin
	// point, and only keep the component along the plane normal
	Vector3d guidance_force_3d =
		-_kp_guidance_pos * (input.device_position - reference_point) -
		_kv_guidance_pos * input.device_linear_velocity;
	Vector3d guidance_force_1d =
		projectAlongDirection(guidance_force_3d, _plane_normal_direction);

	// only keep the component of the non guidance haptic force inside the plane
	force_to_update =
		force_to_update -
		projectAlongDirection(force_to_update, _plane_normal_direction) +
		guidance_force_1d;
}

void HapticDeviceController::applyLineGuidanceForce(
	Vector3d& force_to_update, const HapticControllerInput& input,
	const bool use_device_home_as_origin) {
	if (!_line_guidance_enabled) {
		return;
	}

	Vector3d reference_point = use_device_home_as_origin
								   ? _device_home_pose.translation()
								   : _line_origin_point;

	// apply a spring damper system to bring the haptic device to the origin
	// point, and remove the component along the line
	Vector3d guidance_force_3d =
		-_kp_guidance_pos * (input.device_position - reference_point) -
		_kv_guidance_pos * input.device_linear_velocity;
	Vector3d guidance_force_2d =
		guidance_force_3d -
		projectAlongDirection(guidance_force_3d, _line_direction);

	// only keep the component of the non guidance haptic force along the line
	force_to_update = projectAlongDirection(force_to_update, _line_direction) +
					  guidance_force_2d;
}

void HapticDeviceController::applyWorkspaceVirtualLimitsForceMoment(
	const HapticControllerInput& input, HapticControllerOtuput& output) {
	if (!_device_workspace_virtual_limits_enabled) {
		return;
	}

	// Add virtual forces according to the device virtual workspace limits
	Vector3d torque_virtual = Vector3d::Zero();

	Vector3d device_home_to_current_position =
		input.device_position - _device_home_pose.translation();
	if (device_home_to_current_position.norm() >=
		_device_workspace_radius_limit) {
		// add force to bring the device back inside the workspace
		output.device_command_force +=
			-_kp_guidance_pos *
				(device_home_to_current_position.norm() -
				 _device_workspace_radius_limit) *
				device_home_to_current_position /
				device_home_to_current_position.norm() -
			_kv_guidance_pos *
				projectAlongDirection(input.device_linear_velocity,
									  device_home_to_current_position);
	}

	AngleAxisd device_home_to_current_orientation_aa = orientationDiffAngleAxis(
		_device_home_pose.rotation(), input.device_orientation);
	if (device_home_to_current_orientation_aa.angle() >=
		// add moment to bring the device back inside the workspace
		_device_workspace_angle_limit) {
		output.device_command_moment +=
			-_kp_guidance_ori *
				(device_home_to_current_orientation_aa.angle() -
				 _device_workspace_angle_limit) *
				device_home_to_current_orientation_aa.axis() -
			_kv_guidance_ori *
				projectAlongDirection(
					input.device_angular_velocity,
					device_home_to_current_orientation_aa.axis());
	}
}

double HapticDeviceController::computeKvPosVariableDamping(
	const double device_velocity) const {
	if (_variable_damping_linvel_thresholds.empty()) {
		return 0;
	}

	if (device_velocity < _variable_damping_linvel_thresholds.at(0)) {
		double interpolation_coeff = computeInterpolationCoeff(
			device_velocity, 0, _variable_damping_linvel_thresholds.at(0));
		return interpolation_coeff * _variable_damping_gains_pos.at(0);
	}

	for (int i = 1; i < _variable_damping_linvel_thresholds.size(); ++i) {
		if (device_velocity < _variable_damping_linvel_thresholds.at(i)) {
			double interpolation_coeff = computeInterpolationCoeff(
				device_velocity, _variable_damping_linvel_thresholds.at(i - 1),
				_variable_damping_linvel_thresholds.at(i));
			return interpolation_coeff * _variable_damping_gains_pos.at(i) +
				   (1 - interpolation_coeff) *
					   _variable_damping_gains_pos.at(i - 1);
		}
	}
	return _variable_damping_gains_pos.back();
}

double HapticDeviceController::computeKvOriVariableDamping(
	const double device_velocity) const {
	if (_variable_damping_angvel_thresholds.empty()) {
		return 0;
	}

	if (device_velocity < _variable_damping_angvel_thresholds.at(0)) {
		double interpolation_coeff = computeInterpolationCoeff(
			device_velocity, 0, _variable_damping_angvel_thresholds.at(0));
		return interpolation_coeff * _variable_damping_gains_ori.at(0);
	}

	for (int i = 1; i < _variable_damping_angvel_thresholds.size(); ++i) {
		if (device_velocity < _variable_damping_angvel_thresholds.at(i)) {
			double interpolation_coeff = computeInterpolationCoeff(
				device_velocity, _variable_damping_angvel_thresholds.at(i - 1),
				_variable_damping_angvel_thresholds.at(i));
			return interpolation_coeff * _variable_damping_gains_ori.at(i) +
				   (1 - interpolation_coeff) *
					   _variable_damping_gains_ori.at(i - 1);
		}
	}
	return _variable_damping_gains_ori.back();
}

///////////////////////////////////////////////////////////////////////////////////
// Parameter setting methods
///////////////////////////////////////////////////////////////////////////////////

void HapticDeviceController::setHapticControlType(
	const HapticControlType& haptic_control_type) {
	if (haptic_control_type == _haptic_control_type) {
		return;
	}
	_device_homed = false;
	_reset_robot_linear_offset = true;
	_reset_robot_angular_offset = true;
	if (haptic_control_type == HapticControlType::FORCE_MOTION &&
		_haptic_control_type != HapticControlType::HOMING) {
		cout << "warning: force motion control can only be set from homing "
				"control. Setting homing control"
			 << endl;
		_haptic_control_type = HapticControlType::HOMING;
		return;
	}
	_haptic_control_type = haptic_control_type;
}

void HapticDeviceController::enableOrientationTeleop() {
	_orientation_teleop_enabled = true;
	_reset_robot_angular_offset = true;
}

void HapticDeviceController::parametrizeProxyForceFeedbackSpace(
	const int proxy_feedback_space_dimension,
	const Vector3d& proxy_or_direct_feedback_axis) {
	if (proxy_feedback_space_dimension < 0 ||
		proxy_feedback_space_dimension > 3) {
		throw std::runtime_error(
			"Proxy feedback space dimension must be between 0 and 3 in "
			"HapticDeviceController::parametrizeProxyForceFeedbackSpace");
	}

	Vector3d normalized_axis = proxy_or_direct_feedback_axis;
	if (proxy_feedback_space_dimension == 1 ||
		proxy_feedback_space_dimension == 2) {
		if (proxy_or_direct_feedback_axis.norm() < 0.001) {
			throw std::runtime_error(
				"Proxy or direct feedback axis must be non-zero in "
				"HapticDeviceController::parametrizeProxyForceFeedbackSpace if "
				"the dimension of the space is 1 or 2");
		}
		normalized_axis = proxy_or_direct_feedback_axis /
						  proxy_or_direct_feedback_axis.norm();
	}

	switch (proxy_feedback_space_dimension) {
		case 0:
			_sigma_proxy_force_feedback.setZero();
			break;
		case 1:
			_sigma_proxy_force_feedback =
				normalized_axis * normalized_axis.transpose();
			break;
		case 2:
			_sigma_proxy_force_feedback =
				Matrix3d::Identity() -
				normalized_axis * normalized_axis.transpose();
			break;
		case 3:
			_sigma_proxy_force_feedback = Matrix3d::Identity();
			break;
	}
}

void HapticDeviceController::
	parametrizeProxyForceFeedbackSpaceFromRobotForceSpace(
		const Matrix3d& robot_sigma_force) {
	if (!robot_sigma_force.isApprox(robot_sigma_force.transpose())) {
		throw std::runtime_error(
			"Robot sigma force matrix must be symmetric in "
			"HapticDeviceController::"
			"parametrizeProxyForceFeedbackSpaceFromRobotForceSpace");
	}
	if (!robot_sigma_force.isApprox(robot_sigma_force * robot_sigma_force)) {
		throw std::runtime_error(
			"Robot sigma force matrix must be a projection matrix in "
			"HapticDeviceController::"
			"parametrizeProxyForceFeedbackSpaceFromRobotForceSpace");
	}
	_sigma_proxy_force_feedback =
		_R_world_device.transpose() * robot_sigma_force * _R_world_device;
}

void HapticDeviceController::parametrizeProxyMomentFeedbackSpace(
	const int proxy_feedback_space_dimension,
	const Vector3d& proxy_or_direct_feedback_axis) {
	if (proxy_feedback_space_dimension < 0 ||
		proxy_feedback_space_dimension > 3) {
		throw std::runtime_error(
			"Proxy feedback space dimension must be between 0 and 3 in "
			"HapticDeviceController::parametrizeProxyMomentFeedbackSpace");
	}

	Vector3d normalized_axis = proxy_or_direct_feedback_axis;
	if (proxy_feedback_space_dimension == 1 ||
		proxy_feedback_space_dimension == 2) {
		if (proxy_or_direct_feedback_axis.norm() < 0.001) {
			throw std::runtime_error(
				"Proxy or direct feedback axis must be non-zero in "
				"HapticDeviceController::parametrizeProxyMomentFeedbackSpace "
				"if the dimension of the space is 1 or 2");
		}
		normalized_axis = proxy_or_direct_feedback_axis /
						  proxy_or_direct_feedback_axis.norm();
	}

	switch (proxy_feedback_space_dimension) {
		case 0:
			_sigma_proxy_moment_feedback.setZero();
			break;
		case 1:
			_sigma_proxy_moment_feedback =
				normalized_axis * normalized_axis.transpose();
			break;
		case 2:
			_sigma_proxy_moment_feedback =
				Matrix3d::Identity() -
				normalized_axis * normalized_axis.transpose();
			break;
		case 3:
			_sigma_proxy_moment_feedback = Matrix3d::Identity();
			break;
	}
}

void HapticDeviceController::
	parametrizeProxyMomentFeedbackSpaceFromRobotForceSpace(
		const Matrix3d& robot_sigma_moment) {
	if (!robot_sigma_moment.isApprox(robot_sigma_moment.transpose())) {
		throw std::runtime_error(
			"Robot sigma moment matrix must be symmetric in "
			"HapticDeviceController::"
			"parametrizeProxyMomentFeedbackSpaceFromRobotForceSpace");
	}
	if (!robot_sigma_moment.isApprox(robot_sigma_moment * robot_sigma_moment)) {
		throw std::runtime_error(
			"Robot sigma moment matrix must be a projection matrix in "
			"HapticDeviceController::"
			"parametrizeProxyMomentFeedbackSpaceFromRobotForceSpace");
	}
	_sigma_proxy_moment_feedback =
		_R_world_device.transpose() * robot_sigma_moment * _R_world_device;
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

void HapticDeviceController::setReductionFactorForce(
	const double reduction_factor_force) {
	if (reduction_factor_force < 0 || reduction_factor_force > 1) {
		throw std::runtime_error(
			"Reduction factors must be between 0 and 1 in "
			"HapticDeviceController::setReductionFactorForceMoment");
	}
	_reduction_factor_force = reduction_factor_force;
}

void HapticDeviceController::setReductionFactorMoment(
	const double reduction_factor_moment) {
	if (reduction_factor_moment < 0 || reduction_factor_moment > 1) {
		throw std::runtime_error(
			"Reduction factors must be between 0 and 1 in "
			"HapticDeviceController::setReductionFactorForceMoment");
	}
	_reduction_factor_moment = reduction_factor_moment;
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

	if (_kp_haptic_pos > _device_limits.max_linear_stiffness) {
		cout << "Warning: kp_pos is higher than the device max linear "
				"stiffness. "
				"Saturating to the device max linear stiffness."
			 << endl;
		_kp_haptic_pos = _device_limits.max_linear_stiffness;
	}
	if (_kv_haptic_pos > _device_limits.max_linear_damping) {
		cout << "Warning: kv_pos is higher than the device max linear damping. "
				"Saturating to the device max linear damping."
			 << endl;
		_kv_haptic_pos = _device_limits.max_linear_damping;
	}
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
	if (_kp_haptic_pos > _device_limits.max_linear_stiffness) {
		cout << "Warning: kp_pos is higher than the device max linear "
				"stiffness. "
				"Saturating to the device max linear stiffness."
			 << endl;
		_kp_haptic_pos = _device_limits.max_linear_stiffness;
	}
	if (_kv_haptic_pos > _device_limits.max_linear_damping) {
		cout << "Warning: kv_pos is higher than the device max linear damping. "
				"Saturating to the device max linear damping."
			 << endl;
		_kv_haptic_pos = _device_limits.max_linear_damping;
	}
	if (_kp_haptic_ori > _device_limits.max_angular_stiffness) {
		cout << "Warning: kp_ori is higher than the device max angular "
				"stiffness. "
				"Saturating to the device max angular stiffness."
			 << endl;
		_kp_haptic_ori = _device_limits.max_angular_stiffness;
	}
	if (_kv_haptic_ori > _device_limits.max_angular_damping) {
		cout
			<< "Warning: kv_ori is higher than the device max angular damping. "
			   "Saturating to the device max angular damping."
			<< endl;
		_kv_haptic_ori = _device_limits.max_angular_damping;
	}
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
	if (_kp_guidance_pos > _device_limits.max_linear_stiffness) {
		cout << "Warning: kp_guidance_pos is higher than the device max linear "
				"stiffness. "
				"Saturating to the device max linear stiffness."
			 << endl;
		_kp_guidance_pos = _device_limits.max_linear_stiffness;
	}
	if (_kv_guidance_pos > _device_limits.max_linear_damping) {
		cout << "Warning: kv_guidance_pos is higher than the device max linear "
				"damping. "
				"Saturating to the device max linear damping."
			 << endl;
		_kv_guidance_pos = _device_limits.max_linear_damping;
	}
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
	_kp_guidance_pos = kp_guidance_pos;
	_kv_guidance_pos = kv_guidance_pos;
	_kp_guidance_ori = kp_guidance_ori;
	_kv_guidance_ori = kv_guidance_ori;
	if (_kp_guidance_pos > _device_limits.max_linear_stiffness) {
		cout << "Warning: kp_guidance_pos is higher than the device max linear "
				"stiffness. "
				"Saturating to the device max linear stiffness."
			 << endl;
		_kp_guidance_pos = _device_limits.max_linear_stiffness;
	}
	if (_kv_guidance_pos > _device_limits.max_linear_damping) {
		cout << "Warning: kv_guidance_pos is higher than the device max linear "
				"damping. "
				"Saturating to the device max linear damping."
			 << endl;
		_kv_guidance_pos = _device_limits.max_linear_damping;
	}
	if (_kp_guidance_ori > _device_limits.max_angular_stiffness) {
		cout
			<< "Warning: kp_guidance_ori is higher than the device max angular "
			   "stiffness. "
			   "Saturating to the device max angular stiffness."
			<< endl;
		_kp_guidance_ori = _device_limits.max_angular_stiffness;
	}
	if (_kv_guidance_ori > _device_limits.max_angular_damping) {
		cout
			<< "Warning: kv_guidance_ori is higher than the device max angular "
			   "damping. "
			   "Saturating to the device max angular damping."
			<< endl;
		_kv_guidance_ori = _device_limits.max_angular_damping;
	}
}

void HapticDeviceController::enablePlaneGuidance(
	const Vector3d plane_origin_point, const Vector3d plane_normal_direction) {
	if (plane_normal_direction.norm() < 0.001) {
		throw std::runtime_error(
			"Plane normal direction must be non-zero in "
			"HapticDeviceController::enablePlaneGuidance");
	}
	if (_line_guidance_enabled) {
		cout << "Warning: plane guidance is enabled while line guidance is "
				"already enabled. Disabling line guidance."
			 << endl;
		disableLineGuidance();
	}
	if (_device_workspace_virtual_limits_enabled &&
		(plane_origin_point - _device_home_pose.translation()).norm() >
			_device_workspace_radius_limit) {
		cout << "Warning: plane guidance is enabled while the plane origin "
				"point is outside the device virtual workspace limits. "
				"Disabling "
				"device virtual workspace limits."
			 << endl;
		disableHapticWorkspaceVirtualLimits();
	}
	_plane_guidance_enabled = true;
	_plane_origin_point = plane_origin_point;
	_plane_normal_direction =
		plane_normal_direction / plane_normal_direction.norm();
}

void HapticDeviceController::enableLineGuidance(
	const Vector3d line_origin_point, const Vector3d line_direction) {
	if (line_direction.norm() < 0.001) {
		throw std::runtime_error(
			"Line direction must be non-zero in "
			"HapticDeviceController::enablePlaneGuidance");
	}
	if (_plane_guidance_enabled) {
		cout << "Warning: line guidance is enabled while plane guidance is "
				"already enabled. Disabling plane guidance."
			 << endl;
		disablePlaneGuidance();
	}
	if (_device_workspace_virtual_limits_enabled &&
		(line_origin_point - _device_home_pose.translation()).norm() >
			_device_workspace_radius_limit) {
		cout << "Warning: line guidance is enabled while the line origin "
				"point is outside the device virtual workspace limits. "
				"Disabling "
				"device virtual workspace limits."
			 << endl;
		disableHapticWorkspaceVirtualLimits();
	}
	_line_guidance_enabled = true;
	_line_origin_point = line_origin_point;
	_line_direction = line_direction / line_direction.norm();
}

void HapticDeviceController::enableHapticWorkspaceVirtualLimits(
	double device_workspace_radius_limit, double device_workspace_angle_limit) {
	if (device_workspace_radius_limit < 0 || device_workspace_angle_limit < 0) {
		throw std::runtime_error(
			"Workspace virtual limits must be positive in "
			"HapticDeviceController::setHapticWorkspaceVirtualLimits");
	}
	if (_plane_guidance_enabled &&
		(_plane_origin_point - _device_home_pose.translation()).norm() >
			device_workspace_radius_limit) {
		cout << "Warning: virtual workspace limits are enabled enabled while "
				"the plane guidance origin point is outside the device virtual "
				"workspace limits. Disabling plane guidance."
			 << endl;
		disablePlaneGuidance();
	}
	if (_line_guidance_enabled &&
		(_line_origin_point - _device_home_pose.translation()).norm() >
			device_workspace_radius_limit) {
		cout << "Warning: virtual workspace limits are enabled enabled while "
				"the line guidance origin point is outside the device virtual "
				"workspace limits. Disabling line guidance."
			 << endl;
		disableLineGuidance();
	}
	_device_workspace_virtual_limits_enabled = true;
	_device_workspace_radius_limit = device_workspace_radius_limit;
	_device_workspace_angle_limit = device_workspace_angle_limit;
}

void HapticDeviceController::setVariableDampingGainsPos(
	const vector<double>& velocity_thresholds,
	const vector<double>& variable_damping_gains) {
	if (velocity_thresholds.size() != variable_damping_gains.size()) {
		cout << "Warning: velocity thresholds and variable damping gains must "
				"have the same size in "
				"HapticDeviceController::setVariableDampingGainsPos. Ignoring "
				"setting of the variable damping gains."
			 << endl;
	}
	for (int i = 0; i < velocity_thresholds.size(); ++i) {
		if (velocity_thresholds[i] < 0) {
			cout << "Warning: velocity thresholds must be positive in "
					"HapticDeviceController::setVariableDampingGainsPos. "
					"Ignoring setting of the variable damping gains."
				 << endl;
			return;
		}
		if (variable_damping_gains[i] < 0) {
			cout << "Warning: variable damping gains must be positive in "
					"HapticDeviceController::setVariableDampingGainsPos. "
					"Ignoring setting of the variable damping gains."
				 << endl;
			return;
		}
		if (variable_damping_gains[i] > _device_limits.max_linear_damping) {
			cout << "Warning: variable damping gains must be lower than the "
					"device max linear damping in "
					"HapticDeviceController::setVariableDampingGainsPos. "
					"Ignoring setting of the variable damping gains."
				 << endl;
			return;
		}
		if (i > 0 && velocity_thresholds[i] <= velocity_thresholds[i - 1]) {
			cout << "Warning: velocity thresholds must be in strictly "
					"increasing order in "
					"HapticDeviceController::setVariableDampingGainsPos. "
					"Ignoring setting of the variable damping gains."
				 << endl;
			return;
		}
	}
	_variable_damping_linvel_thresholds = velocity_thresholds;
	_variable_damping_gains_pos = variable_damping_gains;
}

void HapticDeviceController::setVariableDampingGainsOri(
	const vector<double>& velocity_thresholds,
	const vector<double>& variable_damping_gains) {
	if (velocity_thresholds.size() != variable_damping_gains.size()) {
		cout << "Warning: velocity thresholds and variable damping gains must "
				"have the same size in "
				"HapticDeviceController::setVariableDampingGainsOri. Ignoring "
				"setting of the variable damping gains."
			 << endl;
	}
	for (int i = 0; i < velocity_thresholds.size(); ++i) {
		if (velocity_thresholds[i] < 0) {
			cout << "Warning: velocity thresholds must be positive in "
					"HapticDeviceController::setVariableDampingGainsOri. "
					"Ignoring setting of the variable damping gains."
				 << endl;
			return;
		}
		if (variable_damping_gains[i] < 0) {
			cout << "Warning: variable damping gains must be positive in "
					"HapticDeviceController::setVariableDampingGainsOri. "
					"Ignoring setting of the variable damping gains."
				 << endl;
			return;
		}
		if (i > 0 && velocity_thresholds[i] <= velocity_thresholds[i - 1]) {
			cout << "Warning: velocity thresholds must be in strictly "
					"increasing order in "
					"HapticDeviceController::setVariableDampingGainsOri. "
					"Ignoring setting of the variable damping gains."
				 << endl;
			return;
		}
	}
	_variable_damping_angvel_thresholds = velocity_thresholds;
	_variable_damping_gains_ori = variable_damping_gains;
}

void HapticDeviceController::setAdmittanceFactors(
	const double device_force_to_robot_delta_position,
	const double device_moment_to_robot_delta_orientation) {
	if (device_force_to_robot_delta_position < 0 ||
		device_moment_to_robot_delta_orientation < 0) {
		throw std::runtime_error(
			"Admittance factors must be positive in "
			"HapticDeviceController::setAdmittanceFactors");
	}
	_device_force_to_robot_delta_position =
		device_force_to_robot_delta_position;
	_device_moment_to_robot_delta_orientation =
		device_moment_to_robot_delta_orientation;
}

void HapticDeviceController::setHomingMaxVelocity(
	const double homing_max_linvel, const double homing_max_angvel) {
	if (homing_max_linvel <= 0 || homing_max_angvel <= 0) {
		throw std::runtime_error(
			"Homing max velocities must be strictly positive in "
			"HapticDeviceController::setHomingMaxVelocity");
	}
	_homing_max_linvel = homing_max_linvel;
	_homing_max_angvel = homing_max_angvel;
}

void HapticDeviceController::setForceDeadbandForceMotionController(
	const double force_deadband) {
	if (force_deadband < 0) {
		throw std::runtime_error(
			"Force deadband must be positive in "
			"HapticDeviceController::setForceDeadbandForceMotionController");
	}
	_force_deadband = force_deadband;
}

void HapticDeviceController::setMomentDeadbandForceMotionController(
	const double moment_deadband) {
	if (moment_deadband < 0) {
		throw std::runtime_error(
			"Moment deadband must be positive in "
			"HapticDeviceController::setMomentDeadbandForceMotionController");
	}
	_moment_deadband = moment_deadband;
}

} /* namespace Sai2Primitives */
