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

OTG_joints::OTG_joints(const VectorXd& initial_position,
					   const double loop_time) {
	_dim = initial_position.size();
	_otg.reset(new Ruckig<DynamicDOFs, EigenVector>(_dim, loop_time));
	_input = InputParameter<DynamicDOFs, EigenVector>(_dim);
	_output = OutputParameter<DynamicDOFs, EigenVector>(_dim);
	_input.synchronization = Synchronization::Phase;

	reInitialize(initial_position);
}

void OTG_joints::reInitialize(const VectorXd& initial_position) {
	if (initial_position.size() != _dim) {
		throw std::invalid_argument(
			"initial position size does not match the dimension of the "
			"OTG_joints object in OTG_joints::reInitialize\n");
	}

	setGoalPosition(initial_position);

	_output.new_position = initial_position;
	_output.new_velocity.setZero();
	_output.new_acceleration.setZero();
	_output.pass_to_input(_input);
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

	_input.max_velocity = max_velocity;
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

	_input.max_acceleration = max_acceleration;
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

	_input.max_jerk = max_jerk;
}

void OTG_joints::disableJerkLimits() {
	_input.max_jerk.setConstant(std::numeric_limits<double>::infinity());
	_input.current_acceleration.setZero();
}

bool OTG_joints::getJerkLimitEnabled() const {
	return _input.max_jerk != VectorXd::Constant(_dim, std::numeric_limits<double>::infinity());
}

void OTG_joints::setGoalPositionAndVelocity(const VectorXd& goal_position,
											const VectorXd& goal_velocity) {
	if (goal_position.size() != _dim || goal_velocity.size() != _dim) {
		throw std::invalid_argument(
			"goal position or velocity size does not match the dimension of "
			"the OTG_joints object in "
			"OTG_joints::setGoalPositionAndVelocity\n");
	}

	if(goal_position.isApprox(_input.target_position) &&
		goal_velocity.isApprox(_input.target_velocity))
	{
		return;
	}

	_goal_reached = false;
	_input.target_position = goal_position;
	_input.target_velocity = goal_velocity;
}

void OTG_joints::update() {
	if(_goal_reached)
	{
		return;
	}
	// compute next state and get result value
	_result_value = _otg->update(_input, _output);

	// if the goal is reached, either return if the current velocity is
	// zero, or set a new goal to the current position with zero velocity
	if (_result_value == Result::Finished) {
		if (_output.new_velocity.norm() < 1e-3) {
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

} /* namespace Sai2Primitives */
