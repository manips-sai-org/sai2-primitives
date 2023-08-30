/**
 * OTG_6dof_cartesian.cpp
 *
 *	A wrapper to use the Ruckig OTG library
 *	specifically to work for 6DOF position and orientation
 *
 * Author: Mikael Jorda
 * Created: August 2023
 */

#include "OTG_6dof_cartesian.h"

using namespace Eigen;
using namespace ruckig;

namespace Sai2Primitives {

namespace {
bool isValidRotation(const Matrix3d mat) {
	if ((mat.transpose() * mat - Matrix3d::Identity()).norm() > 1e-6) {
		return false;
	}
	if (abs(mat.determinant() - 1) > 1e-6) {
		return false;
	}
	return true;
}
}  // namespace

OTG_6dof_cartesian::OTG_6dof_cartesian(const Vector3d& initial_position,
									   const Matrix3d& initial_orientation,
									   const double loop_time) {
	_otg = std::make_shared<Ruckig<6, EigenVector>>(loop_time);
	_input = InputParameter<6, EigenVector>();
	_output = OutputParameter<6, EigenVector>();
	_input.synchronization = Synchronization::Phase;

	_reference_frame = initial_orientation;
	reInitialize(initial_position, initial_orientation);
}

void OTG_6dof_cartesian::reInitialize(const Vector3d& initial_position,
									  const Matrix3d& initial_orientation) {
	setGoalPosition(initial_position);
	setGoalOrientation(initial_orientation);

	_input.current_position = _input.target_position;
	_input.current_velocity.setZero();
	_input.current_acceleration.setZero();

	_output.new_position = _input.target_position;
	_output.new_velocity.setZero();
	_output.new_acceleration.setZero();
}

void OTG_6dof_cartesian::setMaxLinearVelocity(
	const Vector3d& max_linear_velocity) {
	if (max_linear_velocity.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max velocity set to 0 or negative value in some directions in "
			"OTG_6dof_cartesian::setMaxLinearVelocity\n");
	}
	_input.max_velocity.head<3>() = max_linear_velocity;
}

void OTG_6dof_cartesian::setMaxLinearAcceleration(
	const Vector3d& max_linear_acceleration) {
	if (max_linear_acceleration.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max acceleration set to 0 or negative value in some directions in "
			"OTG_6dof_cartesian::setMaxLinearAcceleration\n");
	}
	_input.max_acceleration.head<3>() = max_linear_acceleration;
}

void OTG_6dof_cartesian::setMaxAngularVelocity(const Vector3d& max_velocity) {
	if (max_velocity.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max velocity set to 0 or negative value in some directions in "
			"OTG_6dof_cartesian::setMaxAngularVelocity\n");
	}

	_input.max_velocity.tail<3>() = max_velocity;
}

void OTG_6dof_cartesian::setMaxAngularAcceleration(
	const Vector3d& max_angular_acceleration) {
	if (max_angular_acceleration.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max acceleration set to 0 or negative value in some directions in "
			"OTG_6dof_cartesian::setMaxAngularAcceleration\n");
	}

	_input.max_acceleration.tail<3>() = max_angular_acceleration;
}

void OTG_6dof_cartesian::setMaxJerk(const Vector3d& max_linear_jerk,
									const Vector3d& max_angular_jerk) {
	if (max_linear_jerk.minCoeff() <= 0 || max_angular_jerk.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max jerk set to 0 or negative value in some directions in "
			"OTG_6dof_cartesian::setMaxJerk\n");
	}

	_input.max_jerk.head<3>() = max_linear_jerk;
	_input.max_jerk.tail<3>() = max_angular_jerk;
}

void OTG_6dof_cartesian::setGoalPositionAndLinearVelocity(
	const Vector3d& goal_position, const Vector3d& goal_linear_velocity) {
	if (goal_position.isApprox(_input.target_position.head<3>(), 1e-3) &&
		goal_linear_velocity.isApprox(_input.target_velocity.head<3>(), 1e-3)) {
		return;
	}
	_goal_reached = false;
	_input.target_position.head<3>() = goal_position;
	_input.target_velocity.head<3>() = goal_linear_velocity;
}

void OTG_6dof_cartesian::setGoalOrientationAndAngularVelocity(
	const Matrix3d& goal_orientation, const Vector3d& goal_angular_velocity) {
	if (!isValidRotation(goal_orientation)) {
		throw std::invalid_argument(
			"goal orientation is not a valid rotation matrix "
			"OTG_6dof_cartesian::setGoalOrientationAndAngularVelocity\n");
	}

	if (_goal_orientation_in_base_frame.isApprox(goal_orientation, 1e-3) &&
		_goal_angular_velocity_in_base_frame.isApprox(goal_angular_velocity,
													  1e-3)) {
		return;
	}

	_goal_reached = false;
	// the new reference frame is the current orientation
	Matrix3d new_reference_frame = getNextOrientation();
	Matrix3d R_new_to_previous_reference =
		new_reference_frame.transpose() * _reference_frame;
	_reference_frame = new_reference_frame;
	_goal_orientation_in_base_frame = goal_orientation;
	_goal_angular_velocity_in_base_frame = goal_angular_velocity;

	// set the new orientation representation vector in otg to zero and
	// rotate input current velocity and acceleration to the new reference
	// frame
	_output.new_position.tail<3>().setZero();
	_output.new_velocity.tail<3>() =
		R_new_to_previous_reference * _output.new_velocity.tail<3>();
	_output.new_acceleration.tail<3>() =
		R_new_to_previous_reference * _output.new_acceleration.tail<3>();
	_output.pass_to_input(_input);

	// set the target position and velocity in the new reference frame
	Matrix3d reference_to_goal =
		_reference_frame.transpose() * _goal_orientation_in_base_frame;
	AngleAxisd reference_to_goal_angle_axis = AngleAxisd(reference_to_goal);
	_input.target_position.tail<3>() = reference_to_goal_angle_axis.angle() *
									   reference_to_goal_angle_axis.axis();
	_input.target_velocity.tail<3>() =
		_reference_frame.transpose() * _goal_angular_velocity_in_base_frame;
}

void OTG_6dof_cartesian::update() {
	// compute next state and get result value
	_result_value = _otg->update(_input, _output);

	// if the goal is reached, either return if the current velocity is
	// zero, or set a new goal to the current position with zero velocity
	if (_result_value == Result::Finished) {
		if (_output.new_velocity.norm() < 1e-3) {
			_goal_reached = true;
		} else {
			setGoalPosition(_input.target_position.head<3>());
			setGoalOrientation(_goal_orientation_in_base_frame);
		}
		return;
	}

	// if the goal is not reached, update the current state and return
	if (_result_value == Result::Working) {
		_output.pass_to_input(_input);
		return;
	}

	// is an error occured, throw an exception
	throw std::runtime_error(
		"error in computing next state in OTG_6dof_cartesian::update.\n");
}

Matrix3d OTG_6dof_cartesian::getNextOrientation() const {
	Matrix3d next_orientation;
	if (_output.new_position.tail<3>().norm() < 1e-3) {
		next_orientation.setIdentity();
	} else {
		next_orientation =
			AngleAxisd(_output.new_position.tail<3>().norm(),
					   _output.new_position.tail<3>().normalized())
				.toRotationMatrix();
	}
	return _reference_frame * next_orientation;
}

} /* namespace Sai2Primitives */