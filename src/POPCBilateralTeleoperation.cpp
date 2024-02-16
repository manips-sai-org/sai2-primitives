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

const int window_size = 30;

const double linvel_lower_bound = 1e-4;
const double angvel_lower_bound = 1e-3;

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
	_max_damping_force =
		0.9 * _haptic_controller->getDeviceLimits().max_linear_damping;
	_max_damping_moment =
		0.9 * _haptic_controller->getDeviceLimits().max_angular_damping;

	_latest_haptic_controller_type = HapticControlType::CLUTCH;

	reInitialize();
}

void POPCBilateralTeleoperation::reInitialize() {
	_passivity_observer_force = 0;
	_PO_buffer_force = queue<double>();

	_passivity_observer_moment = 0;
	_PO_buffer_moment = queue<double>();
}

pair<Vector3d, Vector3d>
POPCBilateralTeleoperation::computeAdditionalHapticDampingForce() {
	pair<Vector3d, Vector3d> damping_force_and_moment =
		make_pair(Vector3d::Zero(), Vector3d::Zero());

	if (_haptic_controller->getHapticControlType() !=
		HapticControlType::MOTION_MOTION) {
		return damping_force_and_moment;
	}
	if (_latest_haptic_controller_type != HapticControlType::MOTION_MOTION) {
		reInitialize();
	}
	_latest_haptic_controller_type = _haptic_controller->getHapticControlType();

	damping_force_and_moment.first = computePOPCForce();
	if (_haptic_controller->getOrientationTeleopEnabled()) {
		damping_force_and_moment.second = computePOPCTorque();
	}
	return damping_force_and_moment;
}

Vector3d POPCBilateralTeleoperation::computePOPCForce() {
	// compute stored energy
	Vector3d robot_position_error = _motion_force_task->getPositionError();
	Vector3d controller_P_force =
		extractKpGainMatrix(_motion_force_task->getPosControlGains()) *
		robot_position_error;
	double stored_energy_force =
		0.5 * robot_position_error.dot(controller_P_force);

	// power output on robot side
	double power_output_robot_side =
		_motion_force_task->getCurrentLinearVelocity().dot(
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

	// power input to the robot controller from the haptic device
	Vector3d device_velocity_in_robot_frame =
		_haptic_controller->getRotationWorldToDeviceBase() *
		_haptic_controller->getScalingFactorPos() * device_velocity;
	double power_input_haptic_to_robot =
		device_velocity_in_robot_frame.dot(controller_P_force);

	// compute total power input
	double total_power_input =
		(power_input_haptic_to_robot - power_output_haptic_side -
		 power_output_robot_side) *
		_loop_dt;

	// compute passivity observer
	_PO_buffer_force.push(total_power_input);
	_passivity_observer_force += total_power_input;

	// compute the passivity controller
	Vector3d damping_force = Vector3d::Zero();
	if (_passivity_observer_force + stored_energy_force < 0.0) {
		// passivity controller triggered
		double vh_norm_square = device_velocity.squaredNorm();

		// Lower bound velocity to ensurre that we can dissipate energy
		if (vh_norm_square < linvel_lower_bound) {
			vh_norm_square = linvel_lower_bound;
		}

		// compute damping gain
		double alpha_force =
			-(_passivity_observer_force + stored_energy_force) /
			(vh_norm_square * _loop_dt);
		if (alpha_force > _max_damping_force) {
			alpha_force = _max_damping_force;
		}

		// compute damping force
		damping_force = -_haptic_controller->getSigmaDirectForceFeedback() *
						alpha_force * device_velocity;

		// correction to observer due to damping
		double passivity_observer_correction =
			_loop_dt * device_velocity.dot(damping_force);
		_passivity_observer_force -= passivity_observer_correction;
		_PO_buffer_force.back() -= passivity_observer_correction;
	} else {
		// passivity controller not triggered
		while (_PO_buffer_force.size() > window_size) {
			// do not reset if it would make your system think it is going
			// to be active
			if (_passivity_observer_force > _PO_buffer_force.front()) {
				// only forget dissipated energy
				if (_PO_buffer_force.front() > 0) {
					_passivity_observer_force -= _PO_buffer_force.front();
				}
				_PO_buffer_force.pop();
			} else {
				break;
			}
		}
	}

	return damping_force;
}

Vector3d POPCBilateralTeleoperation::computePOPCTorque() {
	// compute stored energy
	Vector3d robot_orientation_error =
		_motion_force_task->getOrientationError();
	Vector3d controller_P_moment =
		extractKpGainMatrix(_motion_force_task->getOriControlGains()) *
		robot_orientation_error;
	double stored_energy_moment =
		0.5 * robot_orientation_error.dot(controller_P_moment);

	// power output on robot side
	double power_output_robot_side =
		_motion_force_task->getCurrentLinearVelocity().dot(
			_motion_force_task->sigmaOrientation() *
			_motion_force_task->getUnitMassForce().tail(3));

	// power output haptic side
	Vector3d device_moment_in_motion_space =
		_haptic_controller->getSigmaDirectMomentFeedback() *
		_haptic_controller->getLatestOutput().device_command_moment;
	Vector3d device_angvel =
		_haptic_controller->getLatestInput().device_angular_velocity;
	double power_output_haptic_side =
		device_angvel.dot(device_moment_in_motion_space);

	// power input to the robot controller from the haptic device
	Vector3d device_angular_velocity_in_robot_frame =
		_haptic_controller->getRotationWorldToDeviceBase() *
		_haptic_controller->getScalingFactorOri() * device_angvel;
	double power_input_haptic_to_robot =
		device_angular_velocity_in_robot_frame.dot(controller_P_moment);

	// total power input
	double total_power_input =
		(power_input_haptic_to_robot - power_output_haptic_side -
		 power_output_robot_side) *
		_loop_dt;

	// compute passivity observer
	_PO_buffer_moment.push(total_power_input);
	_passivity_observer_moment += total_power_input;

	// compute the passivity controller
	Vector3d damping_moment = Vector3d::Zero();
	if (_passivity_observer_moment + stored_energy_moment < 0.0) {
		double vh_norm_square = device_angvel.squaredNorm();

		// Lower bound velocity to ensurre that we can dissipate energy
		if (vh_norm_square < angvel_lower_bound) {
			vh_norm_square = angvel_lower_bound;
		}

		// compute damping gain
		double alpha_moment =
			-(_passivity_observer_moment + stored_energy_moment) /
			(vh_norm_square * _loop_dt);
		if (alpha_moment > _max_damping_moment) {
			alpha_moment = _max_damping_moment;
		}

		// compute damping moment
		damping_moment = -alpha_moment * device_angvel;

		// correction to observer due to damping
		double passivity_observer_correction =
			_loop_dt * device_angvel.dot(damping_moment);
		_passivity_observer_moment -= passivity_observer_correction;
		_PO_buffer_moment.back() -= passivity_observer_correction;
	} else {
		while (_PO_buffer_moment.size() > window_size) {
			// do not reset if it would make your system think it is going to be
			// active
			if (_passivity_observer_moment > _PO_buffer_moment.front()) {
				// only forget dissipated energy
				if (_PO_buffer_moment.front() > 0) {
					_passivity_observer_moment -= _PO_buffer_moment.front();
				}
				_PO_buffer_moment.pop();
			} else {
				break;
			}
		}
	}

	return damping_moment;
}

} /* namespace Sai2Primitives */