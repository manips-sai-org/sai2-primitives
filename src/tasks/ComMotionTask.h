/*
 * ComMotionTask.h
 *
 *      This class creates a 6Dof position + orientation hybrid controller for a
 * robotic manipulator using operational space formulation and an underlying PID
 * compensator. If used for hybrid position force control, assumes a force
 * sensor is attached to the same link as the control frame and the force sensed
 * values are given in sensor frame. Besides, the force sensed and moment sensed
 * are assumed to be the force and moment that the robot applies to the
 * environment. It requires a robot model parsed from a urdf file to a Sai2Model
 * object, as well as the definition of a control frame as a link at which the
 * frame is attached, and an affine transform that determines the position and
 * orientation of the control frame in this link
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_COMMOTIONTASK_TASK_H_
#define SAI2_PRIMITIVES_COMMOTIONTASK_TASK_H_

#include <helper_modules/OTG_6dof_cartesian.h>
#include <helper_modules/POPCExplicitForceControl.h>
#include <helper_modules/Sai2PrimitivesCommonDefinitions.h>

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "Sai2Model.h"
#include "TemplateTask.h"

using namespace Eigen;
using namespace std;

namespace Sai2Primitives {

class ComMotionTask : public TemplateTask {
public:
	enum DynamicDecouplingType {
		FULL_DYNAMIC_DECOUPLING,	 // use the real Lambda matrix
		PARTIAL_DYNAMIC_DECOUPLING,	 // Use Lambda for position part, Identity
									 // for orientation and Zero for cross
									 // coupling
		IMPEDANCE,					 // use Identity for the mass matrix
		BOUNDED_INERTIA_ESTIMATES,	 // Use a Lambda computed from a saturated
									 // joint space mass matrix
	};

	//------------------------------------------------
	// Constructor
	//------------------------------------------------

	/**
	 * @brief Construct a new Motion Force Task
	 *
	 * @param robot A pointer to a Sai2Model object for the robot that is to be
	 * controlled
	 * @param link_name The name of the link in the robot at which to attach the
	 * compliant frame
	 * @param compliant_frame Compliant frame with respect to link frame
	 * @param task_name Name of the task
	 * @param is_force_motion_parametrization_in_compliant_frame Whether the
	 * force and motion space (and potential non isotropic gains) are defined
	 * with respect to the compliant frome or the robot base frame
	 * @param loop_timestep Time taken by a control loop. Used in trajectory
	 * generation and integral control.
	 */
	ComMotionTask(
		std::shared_ptr<Sai2Model::Sai2Model>& robot, 
		const std::string& task_name = "com_motion_task",
		const double loop_timestep = 0.001);

	//------------------------------------------------
	// Getters Setters
	//------------------------------------------------

	/**
	 * @brief Get the Current Position
	 *
	 * @return const Vector3d& current position of the control point
	 */
	const Vector3d& getCurrentPosition() const { return _current_position; }
	/**
	 * @brief Get the Current Velocity
	 *
	 * @return const Vector3d& current velocity of the control point
	 */
	const Vector3d& getCurrentVelocity() const { return _current_velocity; }

	/**
	 * @brief Get the nullspace of this task. Will be 0 if ths
	 * is a full joint task
	 *
	 * @return const MatrixXd& Nullspace matrix
	 */
	MatrixXd getTaskNullspace() const override { return _N; }

	/**
	 * @brief Get the nullspace of this and the previous tasks. Concretely, it
	 * is the task nullspace multiplied by the nullspace of the previous tasks
	 *
	 */
	MatrixXd getTaskAndPreviousNullspace() const override {
		return _N * _N_prec;
	}

	void setDesiredPosition(const Vector3d& desired_position) {
		_desired_position = desired_position;
	}
	const Vector3d& getDesiredPosition() const { return _desired_position; }

	void setDesiredVelocity(const Vector3d& desired_velocity) {
		_desired_velocity = desired_velocity;
	}
	const Vector3d& getDesiredVelocity() const { return _desired_velocity; }

	void setDesiredAcceleration(const Vector3d& desired_acceleration) {
		_desired_acceleration = desired_acceleration;
	}
	const Vector3d& getDesiredAcceleration() const {
		return _desired_acceleration;
	}

	// Gains for motion controller
	void setPosControlGains(const PIDGains& gains) {
		setPosControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setPosControlGains(double kp_pos, double kv_pos, double ki_pos = 0);
	void setPosControlGains(const Vector3d& kp_pos, const Vector3d& kv_pos,
							const Vector3d& ki_pos = Vector3d::Zero());
	vector<PIDGains> getPosControlGains() const;

	// internal otg functions
	/**
	 * @brief 	Enables the internal otg for position and orientation with
	 * acceleration limited trajectory, given the input numbers. By default,
	 * this one is enabled with linear velocity and acceleration limits of 0.3
	 * and 1.0 respectively, and angular velocity and acceleration limits of
	 * pi/3 and pi respectively
	 *
	 * @param max_linear_velelocity
	 * @param max_linear_acceleration
	 * @param max_angular_velocity
	 * @param max_angular_acceleration
	 */
	void enableInternalOtgAccelerationLimited(
		const double max_linear_velelocity,
		const double max_linear_acceleration, const double max_angular_velocity,
		const double max_angular_acceleration);

	/**
	 * @brief 	Enables the internal otg for position and orientation with jerk
	 * limited trajectory, given the input numbers
	 *
	 * @param max_linear_velelocity
	 * @param max_linear_acceleration
	 * @param max_linear_jerk
	 * @param max_angular_velocity
	 * @param max_angular_acceleration
	 * @param max_angular_jerk
	 */
	void enableInternalOtgJerkLimited(const double max_linear_velelocity,
									  const double max_linear_acceleration,
									  const double max_linear_jerk,
									  const double max_angular_velocity,
									  const double max_angular_acceleration,
									  const double max_angular_jerk);

	void disableInternalOtg() { _use_internal_otg_flag = false; }

	// Velocity saturation flag and saturation values
	void enableVelocitySaturation(const double linear_vel_sat = 0.3);
	void disableVelocitySaturation() { _use_velocity_saturation_flag = false; }

	bool getVelocitySaturationEnabled() const {
		return _use_velocity_saturation_flag;
	}
	double getLinearSaturationVelocity() const {
		return _linear_saturation_velocity;
	}

	//------------------------------------------------
	// Methods
	//------------------------------------------------

	// -------- core methods --------

	/**
	 * @brief      update the task model (jacobians, task inertia and nullspace
	 *             matrices)
	 * @details    This function updates the jacobian, projected jacobian, task
	 *             inertia matrix (Lambda), dynamically consistent inverse of
	 *             the Jacobian (Jbar) and nullspace matrix of the task N. This
	 *             function uses the robot model and assumes it has been
	 *             updated. There is no use to calling it if the robot
	 *             kinematics or dynamics have not been updated since the last
	 *             call. This function takes the N_prec matrix as a parameter
	 *             which is the product of the nullspace matrices of the higher
	 *             priority tasks. The N matrix will be the matrix to use as
	 *             N_prec for the subsequent tasks. In order to get the
	 *             nullspace matrix of this task alone, one needs to compute _N
	 * * _N_prec.inverse().
	 *
	 * @param      N_prec  The nullspace matrix of all the higher priority
	 *                     tasks. If this is the highest priority task, use
	 *                     identity of size n*n where n in the number of DoF of
	 *                     the robot.
	 */
	void updateTaskModel(const MatrixXd& N_prec) override;

	/**
	 * @brief      Computes the torques associated with this task.
	 * @details    Computes the torques taking into account the last model
	 *             update and updated values for the robot joint
	 *             positions/velocities assumes the desired orientation and
	 *             angular velocity has been updated
	 *
	 */
	VectorXd computeTorques() override;

	/**
	 * @brief      reinitializes the desired state to the current robot
	 *             configuration as well as the integrator terms
	 */
	void reInitializeTask() override;

	/**
	 * @brief      Checks if the desired position is reached op to a certain
	 * tolerance
	 *
	 * @param[in]  tolerance  The tolerance
	 * @param[in]  verbose    display info or not
	 *
	 * @return     true of the position error is smaller than the tolerance
	 */
	bool goalPositionReached(const double tolerance,
							 const bool verbose = false);

	/**
	 * @brief Set the Dynamic Decoupling Type. See the definition of the
	 * DynamicDecouplingType enum for more details
	 *
	 *
	 * @param type
	 */
	void setDynamicDecouplingType(const DynamicDecouplingType type) {
		_dynamic_decoupling_type = type;
	}

	// ------- helper methods -------

	/**
	 * @brief      Resets all the integrated errors used in I terms
	 */
	void resetIntegrators();

	/**
	 * @brief      Resets the integrated errors used in I terms for linear part
	 *             of task (position_integrated_error and
	 *             force_integrated_error)
	 */
	void resetIntegratorsLinear();

	MatrixXd getProjectedJacobian() const { return _projected_jacobian; }
	VectorXd getControlForces() const { return _task_force; };
	VectorXd getUnitControlForces() const { return  _unit_mass_force; }; 
	MatrixXd getLambdaMatrix() const { return _Lambda; };  

private:
	/**
	 * @brief Initial setup of the task, called in the constructor to avoid
	 * duplicated code
	 *
	 */
	void initialSetup();

	// desired pose defaults to the configuration when the task is created
	Vector3d _desired_position;			 // in robot frame
	Vector3d _desired_velocity;			 // in robot frame
	Vector3d _desired_acceleration;

	// gains for motion controller
	// defaults to isptropic 50 for p gains, 14 for d gains and 0 for i gains
	Matrix3d _kp_pos;
	Matrix3d _kv_pos;
	Matrix3d _ki_pos;

	// gains for the closed loop force controller
	// by default, the force controller is open loop
	// to set the behavior to closed loop controller, use the functions
	// setClosedLoopForceControl and setClosedLoopMomentControl. the closed loop
	// force controller is a PI controller with feedforward force and velocity
	// based damping. gains default to isotropic 1 for p gains, 0.7 for i gains
	// and 10 for d gains
	Matrix3d _kp_force, _kp_moment;
	Matrix3d _kv_force, _kv_moment;
	Matrix3d _ki_force, _ki_moment;

	// desired force and moment for the force part of the controller
	// defaults to Zero
	Vector3d _desired_force;   // robot frame
	Vector3d _desired_moment;  // robot frame

	// velocity saturation is off by default
	bool _use_velocity_saturation_flag;
	double _linear_saturation_velocity;
	double _angular_saturation_velocity;

	// internal otg using ruckig, on by default with acceleration limited
	// trajectory
	bool _use_internal_otg_flag;
	std::unique_ptr<OTG_6dof_cartesian> _otg;

	Eigen::VectorXd _task_force;
	Eigen::MatrixXd _N_prec;

	// internal variables, not to be touched by the user
	string _link_name;
	Affine3d _compliant_frame;	// in link_frame
	bool _is_force_motion_parametrization_in_compliant_frame;

	// motion quantities
	Vector3d _current_position;		// robot frame
	Matrix3d _current_orientation;	// robot frame

	Vector3d _current_velocity;			 // robot frame
	Vector3d _current_angular_velocity;	 // robot frame

	Vector3d _orientation_error;			 // robot frame
	Vector3d _integrated_orientation_error;	 // robot frame
	Vector3d _integrated_position_error;	 // robot frame

	// force quantities
	Affine3d _T_control_to_sensor;

	Vector3d _sensed_force;	  // robot frame
	Vector3d _sensed_moment;  // robot frame

	Vector3d _integrated_force_error;	// robot frame
	Vector3d _integrated_moment_error;	// robot frame

	int _force_space_dimension, _moment_space_dimension;
	Vector3d _force_or_motion_axis, _moment_or_rotmotion_axis;

	bool _closed_loop_force_control;
	bool _closed_loop_moment_control;
	double _k_ff;

	// POPC for closed loop force control
	std::unique_ptr<POPCExplicitForceControl> _POPC_force;

	// linear control inputs
	Vector3d _linear_motion_control;
	Vector3d _linear_force_control;

	// control parameters
	bool _are_pos_gains_isotropic;	// defaults to true
	bool _are_ori_gains_isotropic;	// defaults to true

	// dynamic decoupling type, defaults to BOUNDED_INERTIA_ESTIMATES
	DynamicDecouplingType _dynamic_decoupling_type;

	// model quantities
	MatrixXd _jacobian;
	MatrixXd _projected_jacobian;
	MatrixXd _Lambda, _Lambda_modified;
	MatrixXd _Jbar;
	MatrixXd _N;
	double _e_max, _e_min;  // range of eigenvalues to smooth Lambda 
	bool _use_lambda_smoothing_flag;

	MatrixXd _current_task_range;
	int _pos_range, _ori_range;

	Matrix<double, 6, 6> _partial_task_projection;

	VectorXd _unit_mass_force;
	VectorXd _impedance_force;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_MOTIONFORCETASK_TASK_H_ */
#endif