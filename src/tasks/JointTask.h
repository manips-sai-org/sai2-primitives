/*
 * JointTask.h
 *
 *      This class creates a joint controller for a robotic manipulator using
 * dynamic decoupling and an underlying PID compensator. It requires a robot
 * model parsed from a urdf file to a Sai2Model object.
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_JOINT_TASK_H_
#define SAI2_PRIMITIVES_JOINT_TASK_H_

#include <Eigen/Dense>
#include <chrono>
#include <string>

#include "Sai2Model.h"

#ifdef USING_OTG
#include "trajectory_generation/OTG.h"
#endif

namespace Sai2Primitives {

class JointTask {
	enum DynamicDecouplingType {
		FULL_DYNAMIC_DECOUPLING,	// use the real Mass matrix
		BOUNDED_INERTIA_ESTIMATES,	// use a Mass matrix computed from
									// saturating the minimal values of the Mass
									// Matrix
		IMPEDANCE,					// use Identity for the Mass matrix
	};

public:
	/**
	 * @brief      Constructor
	 *
	 * @param      robot      A pointer to a Sai2Model object for the robot that
	 *                        is to be controlled
	 * @param[in]  loop_time  time taken by a control loop. Used only in
	 * trajectory generation
	 */
	JointTask(Sai2Model::Sai2Model* robot, const double loop_timestep = 0.001);

	/**
	 * @brief      update the task model (only _N_prec for a joint task)
	 *
	 * @param      N_prec  The nullspace matrix of all the higher priority
	 *                     tasks. If this is the highest priority task, use
	 *                     identity of size n*n where n in the number of DoF of
	 *                     the robot.
	 */
	void updateTaskModel(const Eigen::MatrixXd N_prec);

	/**
	 * @brief      Computes the torques associated with this task.
	 * @details    Computes the torques taking into account the last model
	 *             update and updated values for the robot joint
	 *             positions/velocities assumes the desired position and
	 *             velocity has been updated
	 *
	 * @param      task_joint_torques  the vector to be filled with the new
	 *                                 joint torques to apply for the task
	 */
	Eigen::VectorXd computeTorques();

	/**
	 * @brief      reinitializes the desired state to the current robot
	 *             configuration as well as the integrator terms
	 */
	void reInitializeTask();

	const Eigen::VectorXd& getCurrentPosition() { return _current_position; }

	void setDesiredPosition(const Eigen::VectorXd desired_position) {
		_desired_position = desired_position;
	}
	const Eigen::VectorXd& getDesiredPosition() const {
		return _desired_position;
	}

	void setDesiredvelocity(const Eigen::VectorXd desired_velocity) {
		_desired_velocity = desired_velocity;
	}
	const Eigen::VectorXd& getDesiredvelocity() const {
		return _desired_velocity;
	}

	void setDesiredacceleration(const Eigen::VectorXd desired_acceleration) {
		_desired_acceleration = desired_acceleration;
	}
	const Eigen::VectorXd& getDesiredacceleration() const {
		return _desired_acceleration;
	}

	void setGains(double kp, double kv, double ki = 0) {
		_kp = kp;
		_kv = kv;
		_ki = ki;
	}

	void enableVelocitySaturation(const Eigen::VectorXd& saturation_velocity) {
		_use_velocity_saturation_flag = true;
		_saturation_velocity = saturation_velocity;
	}
	void disableVelocitySaturation(const Eigen::VectorXd& saturation_velocity) {
		_use_velocity_saturation_flag = false;
	}

	// ---------- set dynamic decoupling type for the controller
	// ----------------
	void setDynamicDecouplingFull();
	void setDynamicDecouplingBIE();
	void setDynamicDecouplingNone();

	void setNonIsotropicGains(const Eigen::VectorXd& kp,
							  const Eigen::VectorXd& kv,
							  const Eigen::VectorXd& ki);
	void setIsotropicGains(const double kp, const double kv, const double ki);

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

private:
	// inputs to be defined by the user
	Eigen::VectorXd _desired_position;	// defaults to the current configuration
										// when the task is created
	Eigen::VectorXd _desired_velocity;	// defaults to zero
	Eigen::VectorXd _desired_acceleration;	// defaults to zero

	double _kp;	 // defaults to 50.0
	double _kv;	 // defaults to 14.0
	double _ki;	 // defaults to 0.0

	bool _use_velocity_saturation_flag;	   // defaults to false
	Eigen::VectorXd _saturation_velocity;  // defaults to PI/3 for all axes

	// bool _use_isotropic_gains;              // defaults to true
	// Eigen::VectorXd _kp_vec;
	// Eigen::VectorXd _kv_vec;
	// Eigen::VectorXd _ki_vec;

// trajectory generation via interpolation using Reflexxes Library
#ifdef USING_OTG
	bool _use_interpolation_flag;  // defaults to true

	// default limits for trajectory generation (same in all directions) :
	// Velocity      - PI/3  Rad/s
	// Acceleration  - PI    Rad/s^2
	// Jerk          - 3PI   Rad/s^3
#endif

	Sai2Model::Sai2Model* _robot;
	double _loop_timestep;

	Eigen::VectorXd _task_force;
	Eigen::MatrixXd _N_prec;

	// internal variables, not to be touched by the user
	Eigen::VectorXd _current_position;
	Eigen::VectorXd _current_velocity;

	Eigen::VectorXd _integrated_position_error;

	Eigen::VectorXd _step_desired_position;
	Eigen::VectorXd _step_desired_velocity;
	Eigen::VectorXd _step_desired_acceleration;

	bool _use_isotropic_gains;
	Eigen::MatrixXd _kp_mat;
	Eigen::MatrixXd _kv_mat;
	Eigen::MatrixXd _ki_mat;

	Eigen::MatrixXd _M_modified;
	int _dynamic_decoupling_type = BOUNDED_INERTIA_ESTIMATES;

#ifdef USING_OTG
	double _loop_time;
	OTG* _otg;
#endif
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_JOINT_TASK_H_ */
#endif