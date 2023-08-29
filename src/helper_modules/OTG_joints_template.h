/**
 * OTG_joints.h
 *
 *	A wrapper to use the Ruckig OTG library
 *
 * Author: Mikael Jorda
 * Created: August 2023
 */

#ifndef SAI2_PRIMITIVES_OTG_JOINTS_H
#define SAI2_PRIMITIVES_OTG_JOINTS_H

#include <Eigen/Dense>
#include <iostream>
#include <ruckig/ruckig.hpp>

using namespace Eigen;
using namespace ruckig;

namespace Sai2Primitives {

template <size_t dim>
class OTG_joints {
	using VectorDimd = Matrix<double, dim, 1>;

public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  initial_position   Initial joint position. Serves to
	 * initialize the dimension of the space
	 * @param[in]  loop_time          The duration of a control loop (typically,
	 * 0.001 if the robot is controlled at 1 kHz)
	 */
	OTG_joints(const VectorDimd& initial_position, const double loop_time) {
		_otg = Ruckig<dim, EigenVector>{loop_time};
		_input.synchronization = Synchronization::Phase;

		reInitialize(initial_position);
	}

	/**
	 * @brief      destructor
	 */
	~OTG_joints() = default;

	/**
	 * @brief 	Reinitializes the OTG_joints with a new initial position
	 *
	 * @param initial_position
	 */
	void reInitialize(const VectorDimd& initial_position) {
		setGoalPosition(initial_position);

		_input.current_position = initial_position;
		_input.current_velocity.setZero();
		_input.current_acceleration.setZero();

		_output.new_position = initial_position;
		_output.new_velocity.setZero();
		_output.new_acceleration.setZero();
	}

	/**
	 * @brief      Sets the maximum velocity for the trajectory generator
	 *
	 * @param[in]  max_velocity  Vector of the maximum velocity per direction
	 */
	void setMaxVelocity(const VectorDimd& max_velocity) {
		if (max_velocity.minCoeff() <= 0) {
			throw std::invalid_argument(
				"max velocity cannot be 0 or negative in any directions in "
				"OTG_joints::setMaxVelocity\n");
		}
		_input.max_velocity = max_velocity;
	}

	/**
	 * @brief      Sets the maximum velocity.
	 *
	 * @param[in]  max_velocity  Scalar of the maximum velocity in all
	 * directions
	 */
	void setMaxVelocity(const double max_velocity) {
		setMaxVelocity(max_velocity * VectorDimd::Ones());
	}

	/**
	 * @brief      Sets the maximum acceleration.
	 *
	 * @param[in]  max_acceleration  Vector of the maximum acceleration
	 */
	void setMaxAcceleration(const VectorXd& max_acceleration) {
		if (max_acceleration.minCoeff() <= 0) {
			throw std::invalid_argument(
				"max acceleration cannot be 0 or negative in any "
				"directions in OTG_joints::setMaxAcceleration\n");
		}
		_input.max_acceleration = max_acceleration;
	}

	/**
	 * @brief      Sets the maximum acceleration.
	 *
	 * @param[in]  max_acceleration  Scalar of the maximum acceleration
	 */
	void setMaxAcceleration(const double max_acceleration) {
		setMaxAcceleration(max_acceleration * VectorDimd::Ones());
	}

	/**
	 * @brief      Sets the maximum jerk and enables jerk limitation for the
	 * trajectory generator
	 *
	 * @param[in]  max_jerk  Vector of the maximum jerk
	 */
	void setMaxJerk(const VectorXd& max_jerk) {
		if (max_jerk.minCoeff() <= 0) {
			throw std::invalid_argument(
				"max jerk cannot be 0 or negative in any directions in "
				"OTG_joints::setMaxJerk\n");
		}
		_input.max_jerk = max_jerk;
	}

	/**
	 * @brief      Sets the maximum jerk and enables jerk limitation for the
	 * trajectory generator
	 *
	 * @param[in]  max_jerk  Scalar of the maximum jerk
	 */
	void setMaxJerk(const double max_jerk) {
		setMaxJerk(max_jerk * VectorDimd::Ones());
	}

	/**
	 * @brief      Disables jerk limitation for the trajectory generator (enable
	 * them by setting jerk limits with the setMaxJerk function)
	 */
	void disableJerkLimits() {
		_input.max_jerk.setConstant(std::numeric_limits<double>::max());
		_input.current_acceleration.setZero();
	}

	// /**
	//  * @brief      Sets the goal position and velocity
	//  *
	//  * @param[in]  goal_position  The goal position
	//  * @param[in]  goal_velocity  The goal velocity
	//  */
	void setGoalPositionAndVelocity(const VectorDimd& goal_position,
									const VectorDimd& goal_velocity) {
		_goal_reached = false;
		_input.target_position = goal_position;
		_input.target_velocity = goal_velocity;
	}

	// /**
	//  * @brief      Sets the goal position and zero target velocity
	//  *
	//  * @param[in]  goal_position  The goal position
	//  */
	void setGoalPosition(const VectorDimd& goal_position) {
		setGoalPositionAndVelocity(goal_position, VectorDimd::Zero());
	}

	/**
	 * @brief      Runs the trajectory generation to compute the next desired
	 * state. Should be called once per control loop
	 *
	 */
	void update() {
		// compute next state and get result value
		_result_value = _otg.update(_input, _output);

		// if the goal is reached, either return if the current velocity is
		// zero, or set a new goal to the current position with zero velocity
		if (_result_value == Result::Finished) {
			if (_output.new_velocity.norm() < 1e-6) {
				_goal_reached = true;
			} else {
				setGoalPosition(_input.target_position);
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

	/**
	 * @brief      Gets the next position.
	 *
	 * @return     The next position.
	 */
	const VectorDimd& getNextPosition() { return _output.new_position; }

	/**
	 * @brief      Gets the next velocity.
	 *
	 * @return     The next velocity.
	 */
	const VectorDimd& getNextVelocity() { return _output.new_velocity; }

	/**
	 * @brief      Gets the next acceleration.
	 *
	 * @return     The next acceleration.
	 */
	const VectorDimd& getNextAcceleration() { return _output.new_acceleration; }

	/**
	 * @brief      Function to know if the goal position and velocity is
	 reached
	 *
	 * @return     true if the goal state is reached, false otherwise
	 */
	bool isGoalReached() const { return _goal_reached;}

private:
	bool _goal_reached = false;

	// Reflexxes variables
	int _result_value = Result::Finished;

	Ruckig<dim, EigenVector> _otg;
	InputParameter<dim> _input;
	OutputParameter<dim> _output;
};

} /* namespace Sai2Primitives */

#endif	// SAI2_PRIMITIVES_OTG_JOINTS_H