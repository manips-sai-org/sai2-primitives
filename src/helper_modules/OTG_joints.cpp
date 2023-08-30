/**
 * OTG_joints.cpp
 *
 *	A wrapper to use the Ruckig OTG library
 *
 * Author: Mikael Jorda
 * Created: August 2023
 */

#include "OTG_joints.h"

using namespace Eigen;
using namespace ruckig;

namespace Sai2Primitives {

namespace {
bool isVectorZero(std::vector<double> vec) {
	for (auto& v : vec) {
		if (fabs(v) > 1e-3) {
			return false;
		}
	}
	return true;
}
}  // namespace

OTG_joints::OTG_joints(const VectorXd& initial_position,
					   const double loop_time) {
	_dim = initial_position.size();
	_otg = std::make_shared<Ruckig<DynamicDOFs, StandardVector, true>>(_dim, loop_time);
	_input = InputParameter<DynamicDOFs>(_dim);
	_output = OutputParameter<DynamicDOFs>(_dim);
	_input.synchronization = Synchronization::Phase;

	_goal_position_eigen.resize(_dim);
	_goal_velocity_eigen.resize(_dim);

	reInitialize(initial_position);
}

void OTG_joints::reInitialize(const VectorXd& initial_position) {
	if (initial_position.size() != _dim) {
		throw std::invalid_argument(
			"initial position size does not match the dimension of the "
			"OTG_joints object in OTG_joints::reInitialize\n");
	}

	setGoalPosition(initial_position);

	for (int i = 0; i < _dim; ++i) {
		_input.current_position[i] = initial_position[i];
		_input.current_velocity[i] = 0;
		_input.current_acceleration[i] = 0;

		_output.new_position[i] = initial_position[i];
		_output.new_velocity[i] = 0;
		_output.new_acceleration[i] = 0;
	}
}

void OTG_joints::setMaxVelocity(const VectorXd& max_velocity) {
	if (max_velocity.size() != _dim) {
		throw std::invalid_argument(
			"max velocity size does not match the dimension of the OTG_joints "
			"object in OTG_joints::setMaxVelocity\n");
	}
	if (max_velocity.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max velocity cannot be 0 or negative in any directions in "
			"OTG_joints::setMaxVelocity\n");
	}

	for (int i = 0; i < _dim; ++i) {
		_input.max_velocity[i] = max_velocity[i];
	}
}

void OTG_joints::setMaxAcceleration(const VectorXd& max_acceleration) {
	if (max_acceleration.size() != _dim) {
		throw std::invalid_argument(
			"max acceleration size does not match the dimension of the "
			"OTG_joints object in OTG_joints::setMaxAcceleration\n");
	}
	if (max_acceleration.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max acceleration cannot be 0 or negative in any "
			"directions in OTG_joints::setMaxAcceleration\n");
	}

	for (int i = 0; i < _dim; ++i) {
		_input.max_acceleration[i] = max_acceleration[i];
	}
}

void OTG_joints::setMaxJerk(const VectorXd& max_jerk) {
	if (max_jerk.size() != _dim) {
		throw std::invalid_argument(
			"max jerk size does not match the dimension of the OTG_joints "
			"object in OTG_joints::setMaxJerk\n");
	}
	if (max_jerk.minCoeff() <= 0) {
		throw std::invalid_argument(
			"max jerk cannot be 0 or negative in any directions in "
			"OTG_joints::setMaxJerk\n");
	}

	for (int i = 0; i < _dim; ++i) {
		_input.max_jerk[i] = max_jerk[i];
	}
}

void OTG_joints::disableJerkLimits() {
	for (int i = 0; i < _dim; ++i) {
		_input.max_jerk[i] = std::numeric_limits<double>::infinity();
		_input.current_acceleration[i] = 0;
	}
}

void OTG_joints::setGoalPositionAndVelocity(const VectorXd& goal_position,
											const VectorXd& goal_velocity) {
	if (goal_position.size() != _dim || goal_velocity.size() != _dim) {
		throw std::invalid_argument(
			"goal position or velocity size does not match the dimension of "
			"the OTG_joints object in "
			"OTG_joints::setGoalPositionAndVelocity\n");
	}

	if((goal_position - _goal_position_eigen).norm() > 1e-3 ||
		(goal_velocity - _goal_velocity_eigen).norm() > 1e-3)
	{
		_goal_reached = false;
	}

	_goal_position_eigen = goal_position;
	_goal_velocity_eigen = goal_velocity;

	for (int i = 0; i < _dim; ++i) {
		_input.target_position[i] = _goal_position_eigen[i];
		_input.target_velocity[i] = _goal_velocity_eigen[i];
	}
}

void OTG_joints::update() {
	// compute next state and get result value
	_result_value = _otg->update(_input, _output);

	// if the goal is reached, either return if the current velocity is
	// zero, or set a new goal to the current position with zero velocity
	if (_result_value == Result::Finished) {
		if (isVectorZero(_output.new_velocity)) {
			_goal_reached = true;
		} else {
			setGoalPosition(_goal_position_eigen);
		}
		return;
	}

	// if still working, update the next input and return
	if (_result_value == Result::Working) {
		_output.pass_to_input(_input);
		return;
	}

	// if an error occurred, throw an exception
	throw std::runtime_error(
		"error in computing next state in OTG_joints::update.\n");
}

VectorXd OTG_joints::getNextPosition() {
	VectorXd next_position(_dim);
	for (int i = 0; i < _dim; ++i) {
		next_position[i] = _output.new_position[i];
	}
	return next_position;
}

VectorXd OTG_joints::getNextVelocity() {
	VectorXd next_velocity(_dim);
	for (int i = 0; i < _dim; ++i) {
		next_velocity[i] = _output.new_velocity[i];
	}
	return next_velocity;
}

VectorXd OTG_joints::getNextAcceleration() {
	VectorXd next_acceleration(_dim);
	for (int i = 0; i < _dim; ++i) {
		next_acceleration[i] = _output.new_acceleration[i];
	}
	return next_acceleration;
}

} /* namespace Sai2Primitives */
