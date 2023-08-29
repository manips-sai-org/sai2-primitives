/**
 * OTG_6dof_cartesian.h
 *
 *	A wrapper to use the Ruckig OTG library
 *	specifically to work for 6DOF position and orientation
 *
 * Author: Mikael Jorda
 * Created: August 2023
 */

#ifndef SAI2_PRIMITIVES_OTG_6DOF_CARTESIAN_H
#define SAI2_PRIMITIVES_OTG_6DOF_CARTESIAN_H

#include <Eigen/Dense>
#include <ReflexxesAPI.h>

#include <iostream>

namespace Sai2Primitives
{

class OTG_6dof_cartesian {
public:


	/**
	 * @brief      constructor
	 *
	 * @param[in]  initial_position     The initial position
	 * @param[in]  initial_orientation  The initial orientation to initialize
	 *                                  the trajectory generation
	 * @param[in]  loop_time            The duration of a control loop
	 *                                  (typically, 0.001 if the robot is
	 *                                  controlled at 1 kHz)
	 */
	OTG_6dof_cartesian(const Eigen::VectorXd& initial_position, const Eigen::Matrix3d& initial_orientation, const double loop_time);

	/**
	 * @brief      destructor
	 */
	~OTG_6dof_cartesian();

	/**
	 * @brief      Re initializes the trajectory generator so that the goal state is the state given as argument
	 *
	 * @param[in]  initial_position     The initial position
	 * @param[in]  initial_orientation  The initial orientation
	 */
	void reInitialize(const Eigen::VectorXd& initial_position, const Eigen::Matrix3d& initial_orientation);

	/**
	 * @brief      Sets the maximum linear velocity for the trajectory generator
	 *
	 * @param[in]  max_linear_velocity  Vector of the maximum velocity per direction
	 */
	void setMaxLinearVelocity(const Eigen::Vector3d max_linear_velocity);

	/**
	 * @brief      Sets the maximum linear velocity.
	 *
	 * @param[in]  max_linear_velocity  Scalar of the maximum velocity in all directions
	 */
	void setMaxLinearVelocity(const double max_linear_velocity);

	/**
	 * @brief      Sets the maximum linear acceleration.
	 *
	 * @param[in]  max_linear_acceleration  Vector of the maximum acceleration
	 */
	void setMaxLinearAcceleration(const Eigen::Vector3d max_linear_acceleration);

	/**
	 * @brief      Sets the maximum linear acceleration.
	 *
	 * @param[in]  max_linear_acceleration  Scalar of the maximum acceleration
	 */
	void setMaxLinearAcceleration(const double max_linear_acceleration);

	/**
	 * @brief      Sets the maximum linear jerk.
	 *
	 * @param[in]  max_linear_jerk  Vector of the maximum jerk
	 */
	void setMaxLinearJerk(const Eigen::Vector3d max_linear_jerk);

	/**
	 * @brief      Sets the maximum linear jerk.
	 *
	 * @param[in]  max_linear_jerk  Scalar of the maximum jerk
	 */
	void setMaxLinearJerk(const double max_linear_jerk);

	/**
	 * @brief      Sets the maximum angular velocity for the trajectory generator
	 *
	 * @param[in]  max_angular_velocity  Vector of the maximum velocity per direction
	 */
	void setMaxAngularVelocity(const Eigen::Vector3d max_angular_velocity);

	/**
	 * @brief      Sets the maximum angular velocity.
	 *
	 * @param[in]  max_angular_velocity  Scalar of the maximum velocity in all directions
	 */
	void setMaxAngularVelocity(const double max_angular_velocity);

	/**
	 * @brief      Sets the maximum angular acceleration.
	 *
	 * @param[in]  max_angular_acceleration  Vector of the maximum acceleration
	 */
	void setMaxAngularAcceleration(const Eigen::Vector3d max_angular_acceleration);

	/**
	 * @brief      Sets the maximum angular acceleration.
	 *
	 * @param[in]  max_angular_acceleration  Scalar of the maximum acceleration
	 */
	void setMaxAngularAcceleration(const double max_angular_acceleration);

	/**
	 * @brief      Sets the maximum angular jerk.
	 *
	 * @param[in]  max_angular_jerk  Vector of the maximum jerk
	 */
	void setMaxAngularJerk(const Eigen::Vector3d max_angular_jerk);

	/**
	 * @brief      Sets the maximum angular jerk.
	 *
	 * @param[in]  max_angular_jerk  Scalar of the maximum jerk
	 */
	void setMaxAngularJerk(const double max_angular_jerk);

	/**
	 * @brief      Sets the goal position and velocity
	 *
	 * @param[in]  goal_position  The goal position
	 * @param[in]  goal_velocity  The goal velocity
	 */
	void setGoalPositionAndLinearVelocity(const Eigen::Vector3d goal_position, const Eigen::Vector3d goal_linear_velocity);

	/**
	 * @brief      Sets the goal orientation and angular velocity
	 *
	 * @param[in]  goal_orientation     The goal orientation
	 * @param[in]  current_orientation  The current orientation
	 * @param[in]  goal_velocity        The goal velocity
	 */
	void setGoalOrientationAndAngularVelocity(const Eigen::Matrix3d goal_orientation, const Eigen::Matrix3d current_orientation, const Eigen::Vector3d goal_angular_velocity);

	/**
	 * @brief      Calculates the next desired position, orientation and velocities
	 *
	 * @param      next_position          The next position
	 * @param      next_velocity          The next velocity
	 * @param      next_orientation       The desired orientation in the next
	 *                                    step
	 * @param      next_angular_velocity  The desired velocity in the next step
	 */
	void computeNextState(Eigen::Vector3d& next_position, Eigen::Vector3d& next_velocity, Eigen::Vector3d& next_acceleration,
		Eigen::Matrix3d& next_orientation, Eigen::Vector3d& next_angular_velocity, Eigen::Vector3d& next_angular_acceleration);

	/**
	 * @brief      Function to know if the goal position and velocity is reached
	 *
	 * @return     true if the goal state is reached, false otherwise
	 */
	bool goalReached();

	double _loop_time;
	bool _goal_reached = false;

	Eigen::Matrix3d _initial_orientation;
	Eigen::Matrix3d _goal_orientation;
	Eigen::Vector3d _goal_angular_velocity_in_base_frame;

	// Reflexxes variables
	int _ResultValue = 0;
    ReflexxesAPI *_RML = NULL;
    RMLPositionInputParameters *_IP = NULL;
    RMLPositionOutputParameters *_OP = NULL;
    RMLPositionFlags _Flags;

};

} /* namespace Sai2Primitives */

#endif //SAI2_PRIMITIVES_OTG_6DOF_CARTESIAN_H