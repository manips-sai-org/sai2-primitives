/*
 * HapticDeviceController.cpp
 *
 *      This controller implements a bilateral haptic teleoperation scheme in open loop.
 * 		The commands are computed for the haptic device (force feedback) and the controlled robot (desired task).
 * 		HapticDeviceController includes impedance-type and admittance-type controllers, with plane/line/orientation guidances,
 * 		and the workspace mapping algorithm.
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
}  // namespace

namespace Sai2Primitives
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//// Constructor, Destructor and Initialization of the haptic controllers
////////////////////////////////////////////////////////////////////////////////////////////////////

HapticDeviceController::HapticDeviceController(const DeviceLimits& device_limits,
					const Eigen::Matrix3d& robot_base_rotation_in_world,
					const Eigen::Affine3d& robot_initial_pose,
					const Eigen::Matrix3d& device_base_rotation_in_world,
					const Eigen::Affine3d& device_home_pose)
{

	_device_limits = device_limits;

	_device_base_to_robot_base_rotation = device_base_rotation_in_world.transpose() * robot_base_rotation_in_world;

	_center_position_robot = robot_initial_pose.translation();
	_center_rotation_robot = robot_initial_pose.linear();

	_home_position_device = device_home_pose.translation();
	_home_rotation_device = device_home_pose.linear();

	//Initialize homing task
	_device_homed = false;

	_haptic_control_type = HapticControlType::DETACHED;

	// Initialize scaling factors (can be set through setScalingFactors())
	_scaling_factor_trans=1.0;
	_scaling_factor_rot=1.0;

	//Initialize position controller parameters
	_kp_homing_pos = 0.2 * _device_limits.max_linear_stiffness;
	_kp_homing_ori = 0.2 * _device_limits.max_angular_stiffness;
	_kv_homing_pos = 2.0 * sqrt(_kp_homing_pos);
	_kv_homing_ori = 2.0 * sqrt(_kp_homing_ori);
	if(_kv_homing_pos > _device_limits.max_linear_damping) {
		_kv_homing_pos = _device_limits.max_linear_damping;
	}
	if(_kv_homing_ori > _device_limits.max_angular_damping) {
		_kv_homing_ori = _device_limits.max_angular_damping;
	}
	_homing_max_linvel = 0.1;
	_homing_max_angvel = M_PI / 2;

	//Initialize virtual proxy parameters
	_proxy_position_impedance = 1400.0;
	_proxy_orientation_impedance = 10.0;
	_proxy_position_damping = 20.0;
	_proxy_orientation_damping = 0.5;

	//Initialize force feedback controller parameters
	_kp_robot_trans_velocity = 10.0;
	_kp_robot_rot_velocity = 10.0;
	_kv_robot_trans_velocity = 0.0;
	_kv_robot_rot_velocity = 0.0;

	_robot_trans_admittance = 1/50.0;
	_robot_rot_admittance = 1/1.5;

	_reduction_factor_torque_feedback << 1/20.0, 0.0, 0.0,
						  0.0, 1/20.0, 0.0,
						  0.0, 0.0, 1/20.0;

	_reduction_factor_force_feedback << 1/2.0, 0.0, 0.0,
						  0.0, 1/2.0, 0.0,
						  0.0, 0.0, 1/2.0;

	// Initialize virtual force guidance gains for unified controller
	_force_guidance_position_impedance = 200.0;
	_force_guidance_orientation_impedance = 20.0;
	_force_guidance_position_damping = 20.0;
	_force_guidance_orientation_damping = 0.04;

	//Initialiaze force feedback computation mode
	_haptic_feedback_from_proxy = false;
	_send_haptic_feedback = false;

	// Initialize Workspace extension parameters
	_center_position_robot_drift.setZero();
	_center_rotation_robot_drift.setIdentity();

	_max_rot_velocity_device=0.001;
	_max_trans_velocity_device=0.001;

	_drift_force.setZero(3);
	_drift_torque.setZero(3);
	_drift_rot_velocity.setZero(3);
	_drift_trans_velocity.setZero(3);

	// Default drift force percentage (can be change with setForceNoticeableDiff())
	_drift_force_admissible_ratio=50.0/100.0;
	_drift_velocity_admissible_ratio = 50.0/100.0;

	// Initialization of the controller parameters
	_device_workspace_radius_max=0.025;
	_task_workspace_radius_max=0.2;
	_device_workspace_tilt_angle_max=20*M_PI/180.0;
	_task_workspace_tilt_angle_max=45*M_PI/180.0;

	// Device workspace virtual limits
	_add_workspace_virtual_limit=false;
	_device_workspace_radius_limit = 0.1;
	_device_workspace_angle_limit = 90*M_PI/180.0;

	//Initialize haptic guidance parameters
	_enable_plane_guidance=false;
	_enable_line_guidance=false;
	_guidance_stiffness=0.7;
	_guidance_damping=0.8;
}

HapticControllerOtuput HapticDeviceController::computeHapticControl(
	const HapticControllerInput& input, const bool verbose) {
	HapticControllerOtuput output;
	switch (_haptic_control_type) {
		case HapticControlType::DETACHED:
			output = computeDetachedControl(input);
		case HapticControlType::HOMING:
			output = computeHomingControl(input);
		case HapticControlType::MOTION_MOTION:
			output = computeMotionMotionControl(input);
		// case HapticControlType::FORCE_MOTION:
		// 	return computeForceMotionControl(input);
		// case HapticControlType::HYBRID_WITH_PROXY:
		// 	return computeHybridWithProxyControl(input);
		// case HapticControlType::WORKSPACE_EXTENSION:
		// 	return computeWorkspaceExtensionControl(input);
		default:
			throw std::runtime_error("Unimplemented haptic control type");
	}
	validateOutput(output, verbose);
	return output;
}

void HapticDeviceController::validateOutput(HapticControllerOtuput& output, const bool verbose) {
	if(output.device_feedback_force.norm() > _device_limits.max_force) {
		if(verbose) {
			std::cout << "Warning: device feedback force norm is too high. Saturating to "
					  << _device_limits.max_force << std::endl;
		}
		output.device_feedback_force *= _device_limits.max_force / output.device_feedback_force.norm();
	}
	if(output.device_feedback_moment.norm() > _device_limits.max_torque) {
		if(verbose) {
			std::cout << "Warning: device feedback moment norm is too high. Saturating to "
					  << _device_limits.max_torque << std::endl;
		}
		output.device_feedback_moment *= _device_limits.max_torque / output.device_feedback_moment.norm();
	}
}

HapticControllerOtuput HapticDeviceController::computeDetachedControl(const HapticControllerInput& input) {
	HapticControllerOtuput output;
	output.robot_goal_position = input.robot_position;
	output.robot_goal_orientation = input.robot_orientation;

	return output;
}

HapticControllerOtuput HapticDeviceController::computeHomingControl(const HapticControllerInput& input) {
	HapticControllerOtuput output;
	output.robot_goal_position = input.robot_position;
	output.robot_goal_orientation = input.robot_orientation;

	if(_kv_homing_pos > 0) {
		Vector3d desired_velocity = - _kp_homing_pos / _kv_homing_pos * (input.device_position - _home_position_device);
		if(desired_velocity.norm() > _homing_max_linvel) {
			desired_velocity *= _homing_max_linvel / desired_velocity.norm();
		}
		output.device_feedback_force = -_kv_homing_pos * (input.device_linear_velocity - desired_velocity);
	}

	if(_kv_homing_ori > 0) {
		Vector3d orientation_error = Sai2Model::orientationError(_home_rotation_device, input.device_orientation);
		Vector3d desired_velocity = - _kp_homing_ori / _kv_homing_ori * orientation_error;
		if(desired_velocity.norm() > _homing_max_angvel) {
			desired_velocity *= _homing_max_angvel / desired_velocity.norm();
		}
		output.device_feedback_moment = -_kv_homing_ori * (input.device_angular_velocity - desired_velocity);
	}

	if( (input.device_position - _home_position_device).norm()<0.002)
	{
		_device_homed = true;
	}

	return output;
}


HapticControllerOtuput HapticDeviceController::computeMotionMotionControl(const HapticControllerInput& input)
{
	HapticControllerOtuput output;
	output.robot_goal_position = input.robot_position;
	output.robot_goal_orientation = input.robot_orientation;

	// Compute robot goal position
	Vector3d device_home_to_current_position =
		input.device_position - _home_position_device;	// in device base frame
	output.robot_goal_position =
		_center_position_robot +
		_scaling_factor_trans *
			_device_base_to_robot_base_rotation.transpose() *
			device_home_to_current_position;

	// compute robot goal orientation
	Matrix3d device_home_to_current_orientation =
		input.device_orientation *
		_home_rotation_device.transpose();	// in device base frame
	AngleAxisd device_home_to_current_orientation_aa(
		device_home_to_current_orientation);
	AngleAxisd scaled_device_home_to_current_orientation_aa(_scaling_factor_rot *
															device_home_to_current_orientation_aa.angle(),
															device_home_to_current_orientation_aa.axis());

	output.robot_goal_orientation =
		_device_base_to_robot_base_rotation.transpose() *
		scaled_device_home_to_current_orientation_aa.toRotationMatrix() *
		_device_base_to_robot_base_rotation * _center_rotation_robot;


	if(!_send_haptic_feedback) {
		return output;
	}

	// Compute the force feedback in robot frame
	Vector3d haptic_forces_robot_space = input.robot_sensed_force;
	Vector3d haptic_moments_robot_space = input.robot_sensed_moment;

	if (_haptic_feedback_from_proxy)
	{
		// Transfer device velocity to robot global frame
		Vector3d device_linear_velocity_in_robot_workspace =
			_scaling_factor_trans *
			_device_base_to_robot_base_rotation.transpose() *
			input.device_linear_velocity;
		Vector3d device_angular_velocity_in_robot_workspace =
			_scaling_factor_rot *
			_device_base_to_robot_base_rotation.transpose() *
			input.device_angular_velocity;

		// Evaluate the task force through stiffness proxy
		haptic_forces_robot_space =
			_proxy_position_impedance *
				(input.robot_position - output.robot_goal_position) -
			_proxy_position_damping *
				(device_linear_velocity_in_robot_workspace -
				 input.robot_linear_velocity);

		// Compute the orientation error
		Vector3d orientation_error = Sai2Model::orientationError(
			output.robot_goal_orientation, input.robot_orientation);
		// Evaluate task torque
		haptic_moments_robot_space =
			_proxy_orientation_impedance * orientation_error -
			_proxy_orientation_damping *
				(device_angular_velocity_in_robot_workspace -
				 input.robot_angular_velocity);

	}

	// scale and rotate to device frame
	Vector3d haptic_force_feedback =
		_device_base_to_robot_base_rotation * _reduction_factor_force_feedback /
		_scaling_factor_trans * haptic_forces_robot_space;
	Vector3d haptic_moment_feedback = _device_base_to_robot_base_rotation *
								 _reduction_factor_torque_feedback /
								 _scaling_factor_rot *
								 haptic_moments_robot_space;

	// Apply haptic guidances if activated
	if (_enable_plane_guidance) {
		Vector3d guidance_force_plane = ComputePlaneGuidanceForce(
			input.device_position, input.device_linear_velocity);
		haptic_force_feedback +=
			guidance_force_plane -
			haptic_force_feedback.dot(_plane_normal_vec) * _plane_normal_vec;
	}

	if (_enable_line_guidance) {
		Vector3d guidance_force_line = ComputeLineGuidanceForce(
			input.device_position, input.device_linear_velocity);
		Vector3d line_vector = (_line_first_point - _line_second_point) /
							   (_line_first_point - _line_second_point).norm();
		haptic_force_feedback =
			guidance_force_line +
			haptic_force_feedback.dot(line_vector) * line_vector;
	}

	if (_add_workspace_virtual_limit) {
		// Add virtual forces according to the device operational space limits
		Vector3d force_virtual = Vector3d::Zero();
		Vector3d torque_virtual = Vector3d::Zero();

		if (device_home_to_current_position.norm() >=
			_device_workspace_radius_limit) {
			force_virtual = -(0.8 * _device_limits.max_linear_stiffness *
							  (device_home_to_current_position.norm() -
							   _device_workspace_radius_limit) /
							  (device_home_to_current_position.norm())) *
							device_home_to_current_position;
		}
		haptic_force_feedback += force_virtual;

		if (device_home_to_current_orientation_aa.angle() >=
			_device_workspace_angle_limit) {
			torque_virtual = -(0.8 * _device_limits.max_angular_stiffness *
							   (device_home_to_current_orientation_aa.angle() -
								_device_workspace_angle_limit)) *
							 device_home_to_current_orientation_aa.axis();
		}
		haptic_moment_feedback += torque_virtual;
	}

	output.device_feedback_force = haptic_force_feedback;
	output.device_feedback_moment = haptic_moment_feedback;

	return output;

}


// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // Admittance controllers in bilateral teleoperation scheme
// ////////////////////////////////////////////////////////////////////////////////////////////////////
// void HapticDeviceController::computeHapticCommandsAdmittance6d(Eigen::Vector3d& desired_trans_velocity_robot,
// 														Eigen::Vector3d& desired_rot_velocity_robot)
// {
// 	device_homed = false;

//  	// Transfer the desired force in the robot global frame
//  	Vector3d _desired_force_robot;
//  	Vector3d _desired_torque_robot;
// 	_desired_force_robot = _scaling_factor_trans * _Rotation_Matrix_DeviceToRobot.transpose() * _sensed_force_device;
// 	_desired_torque_robot = _scaling_factor_rot * _Rotation_Matrix_DeviceToRobot.transpose() * _sensed_torque_device;

//  	// Compute the desired velocity through the set admittance
// 	_desired_trans_velocity_robot = _robot_trans_admittance *_desired_force_robot;
// 	_desired_rot_velocity_robot = _robot_rot_admittance *_desired_torque_robot;

// 	// Compute the force feedback in robot frame from the robot desired and current position (admittance-type scheme)
// 	Vector3d f_task_trans;
// 	Vector3d f_task_rot;
// 	Vector3d orientation_dev;

// 	// Integrate the translational velocity error
// 	_integrated_trans_velocity_error += (_desired_trans_velocity_robot - _current_trans_velocity_robot) * _loop_timer;
// 	// Evaluate the task force
// 	f_task_trans = - _kp_robot_trans_velocity * (_desired_trans_velocity_robot - _current_trans_velocity_robot)- _kv_robot_trans_velocity * _integrated_trans_velocity_error;
// 	// Integrate the rotational velocity error
// 	_integrated_rot_velocity_error += (_desired_rot_velocity_robot - _current_rot_velocity_robot) * _loop_timer;
// 	// Evaluate task torque
// 	f_task_rot = - _kp_robot_rot_velocity * (_desired_rot_velocity_robot - _current_rot_velocity_robot)- _kv_robot_rot_velocity * _integrated_rot_velocity_error;


// 	// Apply reduction factors to force feedback
// 	f_task_trans = _reduction_factor_force_feedback * f_task_trans;
// 	f_task_rot = _reduction_factor_torque_feedback * f_task_rot;
// 	//Transfer task force from robot to haptic device global frame
// 	_commanded_force_device = _Rotation_Matrix_DeviceToRobot * f_task_trans;
// 	_commanded_torque_device = _Rotation_Matrix_DeviceToRobot * f_task_rot;

// 	// Scaling of the force feedback
// 	Eigen::Matrix3d scaling_factor_trans;
// 	Eigen::Matrix3d scaling_factor_rot;

// 		scaling_factor_trans << 1/_scaling_factor_trans, 0.0, 0.0,
// 						  0.0, 1/_scaling_factor_trans, 0.0,
// 						  0.0, 0.0, 1/_scaling_factor_trans;
// 		scaling_factor_rot << 1/_scaling_factor_rot, 0.0, 0.0,
// 						  0.0, 1/_scaling_factor_rot, 0.0,
// 						  0.0, 0.0, 1/_scaling_factor_rot;

// 	_commanded_force_device = scaling_factor_trans * _commanded_force_device;
// 	_commanded_torque_device = scaling_factor_rot * _commanded_torque_device;

// 	// Saturate to Force and Torque limits of the haptic device
// 	if (_commanded_force_device.norm() > _max_force_device)
// 	{
// 		_commanded_force_device = _max_force_device*_commanded_force_device/(_commanded_force_device.norm());
// 	}
// 	if (_commanded_torque_device.norm() > _max_torque_device)
// 	{
// 		_commanded_torque_device = _max_torque_device*_commanded_torque_device/(_commanded_torque_device.norm());
// 	}

// 	if(!_send_haptic_feedback)
// 	{
// 		_commanded_force_device.setZero();
// 		_commanded_torque_device.setZero();
// 	}

//     // Send set velocity to the robot
// 	desired_trans_velocity_robot = _desired_trans_velocity_robot;
// 	desired_rot_velocity_robot = _desired_rot_velocity_robot;

// }


// ///////////////////////////////////////////////////////////////////////////////////
// // Impedance controllers with workspace extension algorithm
// ///////////////////////////////////////////////////////////////////////////////////
// void HapticDeviceController::computeHapticCommandsWorkspaceExtension6d(Eigen::Vector3d& desired_position_robot,
// 																Eigen::Matrix3d& desired_rotation_robot)
// {
// 	 if(_first_iteration)
// 	 {
// 	 	_first_iteration = false;
// 	 	// Set the initial desired position to the robot center
// 	 	_desired_position_robot = _current_position_robot;
// 	 	_desired_rotation_robot = _current_rotation_robot;
// 	 	_center_position_robot_drift = _center_position_robot;
// 	 	_center_rotation_robot_drift = _center_rotation_robot;
// 	 	// Reinitialize maximum device velocity for the task
// 	 	_max_rot_velocity_device=0.001;
// 		_max_trans_velocity_device=0.001;

// 	 }

// 	device_homed = false;

// 	// Update the maximum velocities for the task
// 	if (_current_rot_velocity_device.norm()>=_max_rot_velocity_device)
// 	{
// 		_max_rot_velocity_device = _current_rot_velocity_device.norm();
//     }

// 	if (_current_trans_velocity_device.norm()>=_max_trans_velocity_device)
// 	{
// 		_max_trans_velocity_device = _current_trans_velocity_device.norm();
// 	}

// 	//Transfer device velocity to robot global frame
// 	_current_trans_velocity_device_RobFrame = _scaling_factor_trans * _Rotation_Matrix_DeviceToRobot.transpose() * _current_trans_velocity_device;
// 	_current_rot_velocity_device_RobFrame = _scaling_factor_rot * _Rotation_Matrix_DeviceToRobot.transpose() * _current_rot_velocity_device;


// 	// Compute the force feedback in robot frame
// 	Vector3d f_task_trans;
// 	Vector3d f_task_rot;
// 	Vector3d orientation_dev;

// 	if (_haptic_feedback_from_proxy)
// 	{
// 		// Evaluate the task force through stiffness proxy
// 		f_task_trans = _proxy_position_impedance*(_current_position_proxy - _desired_position_robot) - _proxy_position_damping * (_current_trans_velocity_device_RobFrame - _current_trans_velocity_proxy);

// 		// Compute the orientation error
// 		orientation_dev = Sai2Model::orientationError(_desired_rotation_robot, _current_rotation_proxy);
// 		// Evaluate task torque
// 		f_task_rot = _proxy_orientation_impedance*orientation_dev - _proxy_orientation_damping * (_current_rot_velocity_device_RobFrame - _current_rot_velocity_proxy);

// 	}
// 	else
// 	{
// 		// Read sensed task force
// 		f_task_trans = _sensed_task_force.head(3);
// 		f_task_rot = _sensed_task_force.tail(3);																				//
// 	}

// 	// Apply reduction factors to force feedback
// 	f_task_trans = _reduction_factor_force_feedback * f_task_trans;
// 	f_task_rot = _reduction_factor_torque_feedback * f_task_rot;
// 	//Transfer task force from robot to haptic device global frame
// 	_commanded_force_device = _Rotation_Matrix_DeviceToRobot * f_task_trans;
// 	_commanded_torque_device = _Rotation_Matrix_DeviceToRobot * f_task_rot;

// 	//// Evaluation of the drift velocities ////
// 	//Translational drift
// 	Vector3d relative_position_device;
// 	relative_position_device = _current_position_device-_home_position_device;
// 	_drift_trans_velocity = -_current_trans_velocity_device.norm()*relative_position_device/(_device_workspace_radius_max*_max_trans_velocity_device);
// 	// Rotational drift
// 	Matrix3d relative_rotation_device = _current_rotation_device * _home_rotation_device.transpose(); // Rotation with respect with home orientation
// 	AngleAxisd relative_orientation_angle_device = AngleAxisd(relative_rotation_device);
// 	_drift_rot_velocity=-(_current_rot_velocity_device.norm()*relative_orientation_angle_device.angle()*relative_orientation_angle_device.axis())/(_device_workspace_tilt_angle_max*_max_rot_velocity_device);

// 	//// Computation of the scaling factors ////
// 	_scaling_factor_trans = 1.0 + relative_position_device.norm()*(_task_workspace_radius_max/_device_workspace_radius_max-1.0)/_device_workspace_radius_max;
// 	_scaling_factor_rot = 1.0 + relative_orientation_angle_device.angle()*(_task_workspace_tilt_angle_max/_device_workspace_tilt_angle_max-1.0)/_device_workspace_tilt_angle_max;

// 	// Scaling of the force feedback
// 	Eigen::Matrix3d scaling_factor_trans;
// 	Eigen::Matrix3d scaling_factor_rot;
// 		scaling_factor_trans << 1/_scaling_factor_trans, 0.0, 0.0,
// 						  0.0, 1/_scaling_factor_trans, 0.0,
// 						  0.0, 0.0, 1/_scaling_factor_trans;
// 		scaling_factor_rot << 1/_scaling_factor_rot, 0.0, 0.0,
// 						  0.0, 1/_scaling_factor_rot, 0.0,
// 						  0.0, 0.0, 1/_scaling_factor_rot;

// 	_commanded_force_device = scaling_factor_trans * _commanded_force_device;
// 	_commanded_torque_device = scaling_factor_rot * _commanded_torque_device;


// 	_device_force = _commanded_force_device;
// 	_device_torque = _commanded_torque_device;

// 	//// Evaluation of the drift force ////
// 	// Definition of the velocity gains from task feedback
// 	Matrix3d _Kv_translation = _drift_force_admissible_ratio*(_commanded_force_device.asDiagonal());
// 	Matrix3d _Kv_rotation = _drift_force_admissible_ratio*(_commanded_torque_device.asDiagonal());


// 	// Drift force computation
// 	_drift_force = _Kv_translation * _drift_trans_velocity;
// 	_drift_torque = _Kv_rotation * _drift_rot_velocity;
// 	//_drift_force = Lambda*_drift_force_0; //Drift force weigthed through the device inertia matrix

// 	// cout << "Fdrift \n" << _drift_force << endl;
// 	// cout << "Cdrift \n" << _drift_torque << endl;

// 	//// Desired cartesian force to apply to the haptic device ////
// 	_commanded_force_device += _drift_force;
// 	_commanded_torque_device += _drift_torque;

// 	if(!_send_haptic_feedback)
// 	{
// 		_commanded_force_device.setZero();
// 		_commanded_torque_device.setZero();
// 	}

// 	//// Add virtual forces according to the device operational space limits ////
// 	Vector3d force_virtual = Vector3d::Zero();
// 	Vector3d torque_virtual = Vector3d::Zero();

// 	if (relative_position_device.norm() >= _device_workspace_radius_max)
// 	{
// 		force_virtual = -(0.8 * _max_linear_stiffness_device * (relative_position_device.norm()-_device_workspace_radius_max)/(relative_position_device.norm()))*relative_position_device;
// 	}
// 	_commanded_force_device += force_virtual;

// 	if (relative_orientation_angle_device.angle() >= _device_workspace_tilt_angle_max)
// 	{
// 		torque_virtual = -(0.8 * _max_angular_stiffness_device *(relative_orientation_angle_device.angle()-_device_workspace_tilt_angle_max))*relative_orientation_angle_device.axis();
// 	}
// 	_commanded_torque_device += torque_virtual;

// 	// Saturate to Force and Torque limits of the haptic device
// 	if (_commanded_force_device.norm() > _max_force_device)
// 	{
// 		_commanded_force_device = _max_force_device*_commanded_force_device/(_commanded_force_device.norm());
// 	}
// 	if (_commanded_torque_device.norm() > _max_torque_device)
// 	{
// 		_commanded_torque_device = _max_torque_device*_commanded_torque_device/(_commanded_torque_device.norm());
// 	}

// 	//// Computation of the desired position for the controlled robot after drift of the device ////
// 	// Estimated drift velocity considering drift force
// 	// VectorXd _vel_drift_est = _loop_timer*Lambda.inverse()*_Fdrift; // if estimated human+device mass matrix
// 	Vector3d _vel_drift_est_trans = _loop_timer*_drift_force;
// 	Vector3d _vel_drift_est_rot = _loop_timer*_drift_torque;

// 	// Drift of the center of the task workspace
// 	_center_position_robot_drift -= _Rotation_Matrix_DeviceToRobot.transpose()*(_loop_timer*_scaling_factor_trans*_vel_drift_est_trans);

// 	double _centerRot_angle = -_loop_timer*_scaling_factor_rot*_vel_drift_est_rot.norm();
// 	Vector3d _centerRot_axis;
// 	if (abs(_centerRot_angle) >= 0.00001)
// 	{
// 		_centerRot_axis = _vel_drift_est_rot/_vel_drift_est_rot.norm();
// 		AngleAxisd _centerRot_angleAxis=AngleAxisd(_centerRot_angle,_centerRot_axis);

// 		_center_rotation_robot_drift = _Rotation_Matrix_DeviceToRobot.transpose()*(_centerRot_angleAxis.toRotationMatrix())*_Rotation_Matrix_DeviceToRobot*_center_rotation_robot_drift;
// 	}

// 	//// Compute position of the controlled robot after drift ////
// 	 	// Compute the set position from the haptic device
// 	_desired_position_robot = _scaling_factor_trans*relative_position_device;
// 	// Rotation with respect with home orientation

// 	// Compute set orientation from the haptic device
// 	Eigen::AngleAxisd desired_rotation_robot_aa = Eigen::AngleAxisd(_scaling_factor_rot*relative_orientation_angle_device.angle(),relative_orientation_angle_device.axis());			//
// 	_desired_rotation_robot = desired_rotation_robot_aa.toRotationMatrix();					//


// 	//Transfer set position and orientation from device to robot global frame
// 	_desired_position_robot = _Rotation_Matrix_DeviceToRobot.transpose() * _desired_position_robot;
// 	_desired_rotation_robot = _Rotation_Matrix_DeviceToRobot.transpose() * _desired_rotation_robot * _Rotation_Matrix_DeviceToRobot * _center_rotation_robot_drift; 					//

// 	// Adjust set position to the center of the task Workspace
// 	_desired_position_robot = _desired_position_robot + _center_position_robot_drift;

// 	// Send set position orientation of the robot
// 	desired_position_robot = _desired_position_robot;
// 	desired_rotation_robot = _desired_rotation_robot;										//
// }



// ///////////////////////////////////////////////////////////////////////////////////
// // Unified force and motion haptic controller
// ///////////////////////////////////////////////////////////////////////////////////
// void HapticDeviceController::computeHapticCommandsUnifiedControl6d(Eigen::Vector3d& desired_position_robot,
// 												Eigen::Matrix3d& desired_rotation_robot,
// 												Eigen::Vector3d& desired_force_robot,
// 												Eigen::Vector3d& desired_torque_robot)
// {
// 	device_homed = false;

// 	if(_first_iteration)
// 	 {
// 	 	_first_iteration = false;
// 	 	// Set the initial desired position to the robot center
// 	 	_desired_position_robot = _current_position_robot;
// 	 	_desired_rotation_robot = _current_rotation_robot;
// 	 }

// 	//Transfer device velocity to robot global frame
// 	_current_trans_velocity_device_RobFrame = _scaling_factor_trans * _Rotation_Matrix_DeviceToRobot.transpose() * _current_trans_velocity_device;
// 	_current_rot_velocity_device_RobFrame = _scaling_factor_rot * _Rotation_Matrix_DeviceToRobot.transpose() * _current_rot_velocity_device;
// 	// Position of the device with respect with home position
// 	Vector3d relative_position_device;
// 	relative_position_device = _current_position_device-_home_position_device;
// 	// Rotation of the device with respect with home orientation
// 	Matrix3d relative_rotation_device = _current_rotation_device * _home_rotation_device.transpose(); // Rotation with respect with home orientation
// 	AngleAxisd relative_orientation_angle_device = AngleAxisd(relative_rotation_device);

// 	//// Compute the interaction forces in robot frame ////
// 	Vector3d f_task_trans;
// 	Vector3d f_task_rot;
// 	Vector3d orientation_dev;
// 	if (_haptic_feedback_from_proxy)
// 	{
// 		// Evaluate the task force through stiffness proxy
// 		f_task_trans = _proxy_position_impedance*(_current_position_proxy - _desired_position_robot) - _proxy_position_damping * (_current_trans_velocity_device_RobFrame - _current_trans_velocity_proxy);

// 		// Compute the orientation error
// 		orientation_dev = Sai2Model::orientationError(_desired_rotation_robot, _current_rotation_proxy);
// 		// Evaluate task torque
// 		f_task_rot = _proxy_orientation_impedance*orientation_dev - _proxy_orientation_damping * (_current_rot_velocity_device_RobFrame - _current_rot_velocity_proxy);

// 	}
// 	else
// 	{
// 		// Read sensed task force
// 		f_task_trans = _sensed_task_force.head(3);
// 		f_task_rot = _sensed_task_force.tail(3);
// 	}
// 	//Transfer task force from robot to haptic device global frame
// 	f_task_trans = _Rotation_Matrix_DeviceToRobot * f_task_trans;
// 	f_task_rot = _Rotation_Matrix_DeviceToRobot * f_task_rot;

// 	//// Compute the virtual force guidance in robot frame ////
// 	// Evaluate the virtual guidance force with the spring-damping model
// 	_f_virtual_trans = _force_guidance_position_impedance*(_current_position_robot - _desired_position_robot) - _force_guidance_position_damping * (_current_trans_velocity_device_RobFrame - _current_trans_velocity_robot);
// 	// Compute the orientation error
// 	orientation_dev = Sai2Model::orientationError(_desired_rotation_robot, _current_rotation_robot);
// 	// Evaluate task torque
// 	_f_virtual_rot = _force_guidance_orientation_impedance*orientation_dev - _force_guidance_orientation_damping * (_current_rot_velocity_device_RobFrame - _current_rot_velocity_robot);
// 	//Transfer guidance force from robot to haptic device global frame
// 	_f_virtual_trans = _Rotation_Matrix_DeviceToRobot * _f_virtual_trans;
// 	_f_virtual_rot = _Rotation_Matrix_DeviceToRobot * _f_virtual_rot;

// 	// Apply reduction factors to force feedback
// 	f_task_trans = _reduction_factor_force_feedback * f_task_trans;
// 	f_task_rot = _reduction_factor_torque_feedback * f_task_rot;

// 	// Scaling of the force feedback
// 	Eigen::Matrix3d scaling_factor_trans;
// 	Eigen::Matrix3d scaling_factor_rot;
// 	scaling_factor_trans << 1/_scaling_factor_trans, 0.0, 0.0,
// 					  0.0, 1/_scaling_factor_trans, 0.0,
// 					  0.0, 0.0, 1/_scaling_factor_trans;
// 	scaling_factor_rot << 1/_scaling_factor_rot, 0.0, 0.0,
// 					  0.0, 1/_scaling_factor_rot, 0.0,
// 					  0.0, 0.0, 1/_scaling_factor_rot;

// 	f_task_trans = scaling_factor_trans * f_task_trans;
// 	f_task_rot = scaling_factor_rot * f_task_rot;

// 	//// Projection of the sensed interaction and virtual guidance forces along the force/motion-controlled directions ////
// 	_commanded_force_device = _sigma_force*_f_virtual_trans + _sigma_position*f_task_trans;
// 	_commanded_torque_device = _sigma_moment*_f_virtual_rot + _sigma_orientation*f_task_rot;


// 	// Apply haptic guidances if activated
// 	if (_enable_plane_guidance)
// 	{
// 		ComputePlaneGuidanceForce();
// 		_commanded_force_device += _guidance_force_plane - _commanded_force_device.dot(_plane_normal_vec) * _plane_normal_vec;
// 	}

// 	if (_enable_line_guidance)
// 	{
// 		ComputeLineGuidanceForce();
// 		Vector3d line_vector = (_line_first_point - _line_second_point) / (_line_first_point - _line_second_point).norm();
// 		_commanded_force_device = _guidance_force_line + _commanded_force_device.dot(line_vector) * line_vector;
// 	}

// 	// Add virtual forces according to the device operational space limits
// 	if(_add_workspace_virtual_limit)
// 	{
// 		Vector3d force_virtual = Vector3d::Zero();
// 		Vector3d torque_virtual = Vector3d::Zero();

// 		if (relative_position_device.norm() >= _device_workspace_radius_limit)
// 		{
// 			force_virtual = -(0.8 * _max_linear_stiffness_device * (relative_position_device.norm()-_device_workspace_radius_limit)/(relative_position_device.norm()))*relative_position_device;
// 		}
// 		_commanded_force_device += force_virtual;

// 		if (relative_orientation_angle_device.angle() >= _device_workspace_angle_limit)
// 		{
// 			torque_virtual = -(0.8 * _max_angular_stiffness_device *(relative_orientation_angle_device.angle()-_device_workspace_angle_limit))*relative_orientation_angle_device.axis();
// 		}
// 		_commanded_torque_device += torque_virtual;
// 	}

// 	// Saturate to Force and Torque limits of the haptic device
// 	if (_commanded_force_device.norm() > _max_force_device)
// 	{
// 		_commanded_force_device = _max_force_device*_commanded_force_device/(_commanded_force_device.norm());
// 	}
// 	if (_commanded_torque_device.norm() > _max_torque_device)
// 	{
// 		_commanded_torque_device = _max_torque_device*_commanded_torque_device/(_commanded_torque_device.norm());
// 	}

// 	// Cancel haptic feedback if required
// 	if(!_send_haptic_feedback)
// 	{
// 		_commanded_force_device.setZero();
// 		_commanded_torque_device.setZero();
// 	}

// 	//// Compute the new desired robot position from the haptic device ////
// 	_desired_position_robot = _scaling_factor_trans*relative_position_device;
// 	// Compute set orientation from the haptic device
// 	Eigen::AngleAxisd desired_rotation_robot_aa = Eigen::AngleAxisd(_scaling_factor_rot*relative_orientation_angle_device.angle(),relative_orientation_angle_device.axis());
// 	_desired_rotation_robot = desired_rotation_robot_aa.toRotationMatrix();
// 	//Transfer set position and orientation from device to robot global frame
// 	_desired_position_robot = _Rotation_Matrix_DeviceToRobot.transpose() * _desired_position_robot;
// 	_desired_rotation_robot = _Rotation_Matrix_DeviceToRobot.transpose() * _desired_rotation_robot * _Rotation_Matrix_DeviceToRobot * _center_rotation_robot;
// 	// Adjust set position to the center of the task workspace
// 	_desired_position_robot = _desired_position_robot + _center_position_robot;

// 	//// Compute the new desired robot force from the haptic device ////
// 	_desired_force_robot = - _Rotation_Matrix_DeviceToRobot.transpose() * _commanded_force_device;
// 	_desired_torque_robot = - _Rotation_Matrix_DeviceToRobot.transpose() * _commanded_torque_device;

//     // Send set position and orientation to the robot
// 	desired_position_robot = _desired_position_robot;
// 	desired_rotation_robot = _desired_rotation_robot;
// 	// Send set force and torque to the robot
// 	desired_force_robot = _desired_force_robot;
// 	desired_torque_robot = _desired_torque_robot;
// }

///////////////////////////////////////////////////////////////////////////////////
// Haptic guidance related methods
///////////////////////////////////////////////////////////////////////////////////
Vector3d HapticDeviceController::ComputePlaneGuidanceForce(
	const Vector3d& device_position, const Vector3d& device_velocity) {
	// vector from the plane's origin to the current position
	Eigen::Vector3d plane_to_point_vec;
	plane_to_point_vec = device_position - _plane_origin_point;

	// calculate the normal distance from the plane to the current position
	double distance_to_plane;
	distance_to_plane = plane_to_point_vec.dot(_plane_normal_vec);

	// current position projected onto the plane
	Eigen::Vector3d projected_current_position;
	projected_current_position =
		device_position - distance_to_plane * _plane_normal_vec;

	// projected device velocity on the normal vector
	Eigen::Vector3d projected_current_velocity;
	projected_current_velocity =
		device_velocity.dot(_plane_normal_vec) *
		_plane_normal_vec;

	// calculate the force feedback
	Vector3d guidance_force_plane = -_guidance_stiffness *
								_device_limits.max_linear_stiffness *
								(device_position - projected_current_position) -
							_guidance_damping * projected_current_velocity;

	return guidance_force_plane;
}

Vector3d HapticDeviceController::ComputeLineGuidanceForce(
	const Vector3d& device_position, const Vector3d& device_velocity) {
	Eigen::Vector3d line_to_point_vec;
	double distance_along_line;
	Eigen::Vector3d closest_point_vec;
	Eigen::Vector3d projected_current_position;
	Eigen::Vector3d line_normal_vec;

	// vector from first point on line to current position
	line_to_point_vec = device_position - _line_first_point;

	// distance from the first point to the current position projected onto the
	// line
	distance_along_line = line_to_point_vec.dot(_guidance_line_vec);

	// vector to the projected point onto the line from the first point
	closest_point_vec = distance_along_line * _guidance_line_vec;

	// projected position on line in world frame
	projected_current_position = _line_first_point + closest_point_vec;

	// Normal vector to the line from current position
	line_normal_vec = device_position - projected_current_position;
	line_normal_vec = line_normal_vec / line_normal_vec.norm();

	// projected device velocity on the normal vector
	Eigen::Vector3d projected_current_velocity;
	projected_current_velocity =
		device_velocity.dot(line_normal_vec) * line_normal_vec;

	// compute the force
	Vector3d guidance_force_line =
		-(_guidance_stiffness)*_device_limits.max_linear_stiffness *
			(device_position - projected_current_position) -
		_guidance_damping * projected_current_velocity;

	return guidance_force_line;
}

///////////////////////////////////////////////////////////////////////////////////
// Parameter setting methods
///////////////////////////////////////////////////////////////////////////////////

void HapticDeviceController::setScalingFactors(const double scaling_factor_trans, const double scaling_factor_rot)
{
	_scaling_factor_trans = scaling_factor_trans;
	_scaling_factor_rot = scaling_factor_rot;
}

void HapticDeviceController::setDeviceCenter(const Eigen::Vector3d home_position_device, const Eigen::Matrix3d home_rotation_device)
{
	_home_position_device = home_position_device;
	_home_rotation_device = home_rotation_device;
}



// void HapticDeviceController::setRobotCenter(const Eigen::Vector3d center_position_robot, const Eigen::Matrix3d center_rotation_robot)
// {
// 	_center_position_robot = center_position_robot;
// 	_center_rotation_robot = center_rotation_robot;

// 	//Initialize the set position and orientation of the controlled robot
// 	_desired_position_robot = _center_position_robot;
// 	_desired_rotation_robot = _center_rotation_robot;

// }

void HapticDeviceController::setWorkspaceLimits(double device_workspace_radius_limit, double device_workspace_angle_limit)
{
	_device_workspace_radius_limit = device_workspace_radius_limit;
	_device_workspace_angle_limit = device_workspace_angle_limit;
}


///////////////////////////////////////////////////////////////////////////////////
// Workspace extension related methods
///////////////////////////////////////////////////////////////////////////////////
// void HapticDeviceController::setWorkspaceSize(double device_workspace_radius_max, double task_workspace_radius_max, double device_workspace_tilt_angle_max, double task_workspace_tilt_angle_max)
// {
// 	_device_workspace_radius_max = device_workspace_radius_max;
// 	_task_workspace_radius_max = task_workspace_radius_max;
// 	_device_workspace_tilt_angle_max = device_workspace_tilt_angle_max;
// 	_task_workspace_tilt_angle_max = task_workspace_tilt_angle_max;
// }


// void HapticDeviceController::setNoticeableDiff(double drift_force_admissible_ratio, double drift_velocity_admissible_ratio)
// {
// 	_drift_force_admissible_ratio = drift_force_admissible_ratio;
// 	_drift_velocity_admissible_ratio = drift_velocity_admissible_ratio;
// }

///////////////////////////////////////////////////////////////////////////////////
// Haptic guidance related settings methods
///////////////////////////////////////////////////////////////////////////////////

void HapticDeviceController::setHapticGuidanceGains(const double guidance_stiffness, const double guidance_damping)
{
	_guidance_stiffness = guidance_stiffness;
	_guidance_damping = guidance_damping;
}

void HapticDeviceController::setPlane(const Eigen::Vector3d plane_origin_point, const Eigen::Vector3d plane_normal_vec)
{
	_plane_origin_point = plane_origin_point;
	_plane_normal_vec = plane_normal_vec / plane_normal_vec.norm();
}

void HapticDeviceController::setLine(const Eigen::Vector3d line_first_point, const Eigen::Vector3d line_second_point)
{
	//set vector from one point to the other
	_line_first_point = line_first_point;
	_line_second_point = line_second_point;

	_guidance_line_vec = line_second_point - line_first_point;
	_guidance_line_vec = _guidance_line_vec / _guidance_line_vec.norm();
}



} /* namespace Sai2Primitives */
