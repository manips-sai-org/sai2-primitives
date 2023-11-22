/*
 * POPCBilateralTeleoperation.cpp
 *
 *      Author: Mikael Jorda
 */

#include "POPCBilateralTeleoperation.h"

using namespace std;
using namespace Eigen;

namespace Sai2Primitives {

namespace {
Matrix3d extractKpGainMatrix(vector<PIDGains> gains) {
	if (gains.size() == 1) {
		return gains.at(0).kp * Matrix3d::Identity();
	}
	Matrix3d kp = Matrix3d::Zero();
	for (int i = 0; i < 3; i++) {
		kp(i, i) = gains.at(i).kp;
	}
	return kp;
}
}  // namespace

POPCBilateralTeleoperation::POPCBilateralTeleoperation(
	const shared_ptr<MotionForceTask>& motion_force_task,
	const shared_ptr<HapticDeviceController>& haptic_controller,
	const double loop_dt)
	: _motion_force_task(motion_force_task),
	  _haptic_controller(haptic_controller),
	  _loop_dt(loop_dt) {
	_max_alpha_force = _haptic_controller->getDeviceLimits().max_linear_damping;
	_max_alpha_moment =
		_haptic_controller->getDeviceLimits().max_angular_damping;

	reInitialize();
}

void POPCBilateralTeleoperation::reInitialize() {
	_passivity_observer_force = 0;
	_stored_energy_force = 0;
	_PO_buffer_force = queue<double>();

	_passivity_observer_moment = 0;
	_stored_energy_moment = 0;
	_PO_buffer_moment = queue<double>();

	_alpha_force = 0;
	_damping_force.setZero();
	_alpha_moment = 0;
	_damping_moment.setZero();

	// _first_iteration_force = true;
	// _first_iteration_moment = true;
}

pair<Vector3d, Vector3d> POPCBilateralTeleoperation::computeAdditionalHapticDampingForce() {
	pair<Vector3d, Vector3d> damping_force_and_moment;
	damping_force_and_moment.first = computePOPCForce();
	if(_haptic_controller->getEnableOrientationTeleoperation()) {
		damping_force_and_moment.second = computePOPCTorque();
	}
	else {
		damping_force_and_moment.second.setZero();
	}
	return damping_force_and_moment;
}


Vector3d POPCBilateralTeleoperation::computePOPCForce() {
	// // compute dt for passivity observer
	// chrono::high_resolution_clock::time_point t_curr =
	// 	chrono::high_resolution_clock::now();
	// if (_first_iteration_force) {
	// 	_t_prev_force = t_curr;
	// 	_first_iteration_force = false;
	// }
	// chrono::duration<double> t_diff = t_curr - _t_prev_force;
	// double dt = t_diff.count();

	// // compute stored energy
	// Eigen::Vector3d dx =
	// 	_posori_task->_sigma_position *
	// 	(_posori_task->_desired_position - _posori_task->_current_position);
	// _stored_energy_force = 0.5 * _posori_task->_kp_pos * dx.squaredNorm();
	// _stored_energy_force = 0;

	// power output on robot side
	double power_output_robot_side =
		_motion_force_task->getCurrentVelocity().dot(
			_motion_force_task->sigmaPosition() *
			_motion_force_task->getUnitMassForce().head(3));

	// power output on haptic side
	Vector3d device_force_in_direct_feedback_space =
		_haptic_controller->getSigmaDirectForceFeedback() *
		_haptic_controller->getLatestOutput().device_command_force;

	Vector3d device_velocity =
		_haptic_controller->getLatestInput().device_linear_velocity;

	double power_output_haptic_side =
		device_velocity.dot(device_force_in_direct_feedback_space);
	// double power_output_haptic_side =
	// 	_haptic_task->_current_trans_velocity_device.dot(
	// 		device_force_in_direct_feedback_space) -
	// 	_posori_task->_kp_pos *
	// 		_haptic_task->_current_trans_velocity_device_RobFrame.dot(dx);

	// substract power input to the robot controller
	Vector3d robot_position_error = _motion_force_task->getPositionError();
	Vector3d device_velocity_in_robot_frame =
		_haptic_controller->getRotationWorldToDeviceBase() *
		_haptic_controller->getScalingFactorPos() * device_velocity;
	power_output_haptic_side -= device_velocity_in_robot_frame.dot(
		extractKpGainMatrix(_motion_force_task->getPosControlGains()) *
		robot_position_error);

	// compute total power input
	double total_power_input =
		(-power_output_haptic_side - power_output_robot_side) * _loop_dt;

	// compute passivity observer
	_PO_buffer_force.push(total_power_input);
	_passivity_observer_force += total_power_input;

	// compute the passivity controller
	if (_passivity_observer_force + _stored_energy_force < 0.0) {
		double vh_norm_square = device_velocity.squaredNorm();

		// if velocity of haptic device is too low, we cannot dissipate
		// through damping
		if (vh_norm_square < 1e-6) {
			vh_norm_square = 1e-6;
		}

		_alpha_force = -(_passivity_observer_force + _stored_energy_force) /
					   (vh_norm_square * _loop_dt);

		// limit the amount of damping otherwise the real hardware has
		// issues
		if (_alpha_force > _max_alpha_force) {
			_alpha_force = _max_alpha_force;
		}

		_damping_force = -_haptic_controller->getSigmaDirectForceFeedback() *
						 _alpha_force * device_velocity;

		double passivity_observer_correction =
			_loop_dt * device_velocity.dot(_damping_force);
		_passivity_observer_force -= passivity_observer_correction;
		_PO_buffer_force.back() -= passivity_observer_correction;
	} else {
		// no passivity controller correction
		_alpha_force = 0;
		_damping_force.setZero();

		while (_PO_buffer_force.size() > _PO_buffer_size_force) {
			// do not reset if it would make your system think it is going
			// to be active
			if (_passivity_observer_force > _PO_buffer_force.front()) {
				if (_PO_buffer_force.front() >
					0)	// only forget dissipated energy
				{
					_passivity_observer_force -= _PO_buffer_force.front();
				}
				_PO_buffer_force.pop();
			} else {
				break;
			}
		}
	}

	return _damping_force;
}

Vector3d POPCBilateralTeleoperation::computePOPCTorque() {
	// // compute dt for passivity observer
	// chrono::high_resolution_clock::time_point t_curr =
	// 	chrono::high_resolution_clock::now();
	// if (_first_iteration_moment) {
	// 	_t_prev_moment = t_curr;
	// 	_first_iteration_moment = false;
	// }
	// chrono::duration<double> t_diff = t_curr - _t_prev_moment;
	// double dt = t_diff.count();

	// // compute stored energy
	// Vector3d ori_error =
	// 	-_posori_task->_sigma_orientation * _posori_task->_orientation_error;
	// _stored_energy_moment =
	// 	0.5 * _posori_task->_kp_ori * ori_error.squaredNorm();
	// _stored_energy_moment = 0;

	// compute power input and output
	double power_output_robot_side =
		_motion_force_task->getCurrentVelocity().dot(
			_motion_force_task->sigmaOrientation() *
			_motion_force_task->getUnitMassForce().tail(3));
	// double power_output_robot_side =
	// 	_posori_task->_current_angular_velocity.dot(
	// 		_posori_task->_unit_mass_force.tail(3));

	Vector3d device_moment_in_motion_space =
		_haptic_controller->getSigmaDirectMomentFeedback() *
		_haptic_controller->getLatestOutput().device_command_moment;
	Vector3d device_angvel =
		_haptic_controller->getLatestInput().device_angular_velocity;
	double power_output_haptic_side =
		device_angvel.dot(device_moment_in_motion_space);

	// substract power input to the robot controller
	Vector3d robot_orientation_error =
		_motion_force_task->getOrientationError();
	Vector3d device_angular_velocity_in_robot_frame =
		_haptic_controller->getRotationWorldToDeviceBase() *
		_haptic_controller->getScalingFactorOri() * device_angvel;
	power_output_haptic_side -= device_angular_velocity_in_robot_frame.dot(
		extractKpGainMatrix(_motion_force_task->getOriControlGains()) *
		robot_orientation_error);

	// double power_output_haptic_side =
	// 	_haptic_task->_current_rot_velocity_device.dot(
	// 		device_moment_in_motion_space) -
	// 	_posori_task->_kp_ori *
	// 		_haptic_task->_current_rot_velocity_device_RobFrame.dot(ori_error);

	double total_power_input =
		(-power_output_haptic_side - power_output_robot_side) * _loop_dt;

	// compute passivity observer
	_PO_buffer_moment.push(total_power_input);
	_passivity_observer_moment += total_power_input;

	// compute the passivity controller
	if (_passivity_observer_moment + _stored_energy_moment < 0.0) {
		double vh_norm_square = device_angvel.squaredNorm();

		// if velocity of haptic device is too low, we cannot dissipate through
		// gamping
		if (vh_norm_square < 1e-6) {
			vh_norm_square = 1e-6;
		}

		_alpha_moment = -(_passivity_observer_moment + _stored_energy_moment) /
						(vh_norm_square * _loop_dt);

		// limit the amount of damping otherwise the real hardware has issues
		if (_alpha_moment > _max_alpha_moment) {
			_alpha_moment = _max_alpha_moment;
		}

		_damping_moment = -_alpha_moment * device_angvel;

		double passivity_observer_correction =
			_loop_dt * device_angvel.dot(_damping_moment);
		_passivity_observer_moment -= passivity_observer_correction;
		_PO_buffer_moment.back() -= passivity_observer_correction;
	} else {
		// no passivity controller correction
		_alpha_moment = 0;
		_damping_moment.setZero();

		while (_PO_buffer_moment.size() > _PO_buffer_size_moment) {
			// do not reset if it would make your system think it is going to be
			// active
			if (_passivity_observer_moment > _PO_buffer_moment.front()) {
				if (_PO_buffer_moment.front() >
					0)	// only forget dissipated energy
				{
					_passivity_observer_moment -= _PO_buffer_moment.front();
				}
				_PO_buffer_moment.pop();
			} else {
				break;
			}
		}
	}

	return _damping_moment;
}

} /* namespace Sai2Primitives */