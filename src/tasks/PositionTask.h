/*
 * PositionTask.h
 *
 *      This class creates a position controller for a robotic manipulator using operational space formulation and an underlying PID compensator.
 *      It requires a robot model parsed from a urdf file to a Sai2Model object, as well as the definition of a control frame
 *      as a link at which the frame is attached, and an affine transform that determines the position and orientation of the control frame in this link
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_POSITION_TASK_H_
#define SAI2_PRIMITIVES_POSITION_TASK_H_

#include "Sai2Model.h"
#include <Eigen/Dense>
#include <string>
#include <chrono>
#include <queue> 

#ifdef USING_OTG
	#include "trajectory_generation/OTG.h"
#endif

using namespace Eigen;
using namespace std;

namespace Sai2Primitives
{

class PositionTask : public TemplateTask
{

enum DynamicDecouplingType
{
	FULL_DYNAMIC_DECOUPLING,            // use the real Lambda matrix
	IMPEDANCE,                          // use Identity for the mass matrix
};

public:

	/**
	 * @brief      Constructor that takes an Affine3d matrix for definition of
	 *             the control frame
	 *
	 * @param      robot          A pointer to a Sai2Model object for the robot
	 *                            that is to be controlled
	 * @param      link_name      The name of the link in the robot at which to
	 *                            attach the control frame
	 * @param      control_frame  The position and orientation of the control
	 *                            frame in local link coordinates
	 * @param[in]  loop_time      time taken by a control loop. Used only in trajectory generation
	 */
	PositionTask(Sai2Model::Sai2Model* robot, 
		         const string link_name, 
		         const Affine3d control_frame = Affine3d::Identity(),
		         const double loop_time = 0.001);

	/**
	 * @brief      Constructor that takes a Vector3d for definition of the
	 *             control frame position and a Matrix3d for the frame
	 *             orientation
	 *
	 * @param      robot        A pointer to a Sai2Model object for the robot
	 *                          that is to be controlled
	 * @param      link_name    The name of the link in the robot at which to
	 *                          attach the control frame
	 * @param      pos_in_link  The position the control frame in local link
	 *                          coordinates
	 * @param      rot_in_link  The orientation of the control frame in local
	 *                          link coordinates
	 * @param[in]  loop_time    time taken by a control loop. Used only in trajectory generation
	 */
	PositionTask(Sai2Model::Sai2Model* robot, 
		         const string link_name, 
				 const Vector3d pos_in_link = Vector3d::Zero(), 
				 const Matrix3d rot_in_link = Matrix3d::Identity(),
				 const double loop_time = 0.001);

	/**
	 * @brief update the task model (jacobians, task inertia and nullspace matrices)
	 * @details This function updates the jacobian, projected jacobian, task inertia matrix (Lambda), dynamically consistent inverse of the Jacobian (Jbar)
	 * and nullspace matrix of the task N. This function uses the robot model and assumes it has been updated. 
	 * There is no use to calling it if the robot kinematics or dynamics have not been updated since the last call.
	 * This function takes the N_prec matrix as a parameter which is the product of the nullspace matrices of the higher priority tasks.
	 * The N matrix will be the matrix to use as N_prec for the subsequent tasks.
	 * In order to get the nullspace matrix of this task alone, one needs to compute _N * _N_prec.inverse().	
	 * 
	 * @param N_prec The nullspace matrix of all the higher priority tasks. If this is the highest priority task, use identity of size n*n where n in the number of DoF of the robot.
	 */
	virtual void updateTaskModel(const MatrixXd N_prec);

	/**
	 * @brief Computes the torques associated with this task.
	 * @details Computes the torques taking into account the last model update and updated values for the robot joint positions/velocities
	 * assumes the desired position and velocity has been updated
	 * 
	 * @param task_joint_torques the vector to be filled with the new joint torques to apply for the task
	 */
	virtual void computeTorques(VectorXd& task_joint_torques);

	/**
	 * @brief      reinitializes the desired state to the current robot
	 *             configuration as well as the integrator terms
	 */
	void reInitializeTask();

	/**
	 * @brief      Checks if the desired position is reached op to a certain tolerance
	 *
	 * @param[in]  tolerance  The tolerance
	 * @param[in]  verbose    display info or not
	 *
	 * @return     true of the position error is smaller than the tolerance
	 */
	bool goalPositionReached(const double tolerance, const bool verbose = false);
	
	// ---------- set dynamic decoupling type for the controller  ----------------
	void setDynamicDecouplingFull();
	void setDynamicDecouplingNone();

	void setNonIsotropicGains(const Matrix3d& frame, const Vector3d& kp, const Vector3d& kv, const Vector3d& ki);

	void setIsotropicGains(const double kp, const double kv, const double ki);

	// -------- force control related methods --------

	/**
	 * @brief      Sets the force sensor frame.
	 *
	 * @param[in]  link_name               The link name on which the sensor is attached
	 * @param[in]  transformation_in_link  The transformation in link of the sensor
	 */
	void setForceSensorFrame(const std::string link_name, const Affine3d transformation_in_link);

	/**
	 * @brief      Updates the velues of the sensed force from the sensor
	 * @details    Assumes that the sensor is attached to the same link as the
	 *             control frame and that the setSensorFrame finction has been
	 *             called. The force values given to this function are assumed
	 *             to be in the force sensor frame (values taken directly from
	 *             the force sensor) These values are supposed to be the forces
	 *             that the sensor applies to the environment (so the opposite
	 *             of what the sensor feels)
	 *
	 * @param      sensed_force_sensor_frame   The sensed force as the force
	 *                                         that the sensor applies to the
	 *                                         environment in sensor frame
	 */
	void updateSensedForceAndMoment(const Vector3d sensed_force_sensor_frame);


	/**
	 * @brief      Sets the force controlled axis for a hybrid position force
	 *             controller with 1 DoF force and 2 DoF motion
	 * @details    This is the function to use in order to get the controller to
	 *             behave as a Hybrid Force/Motion controller with 1 Dof force.
	 *             The motion is controlled orthogonally to the force. It can be
	 *             called anytime to change the behavior of the controller and
	 *             reset the integral terms.
	 *
	 * @param      force_axis  The axis in robot frame coordinates along which
	 *                         the controller behaves as a force controller.
	 */
	void setForceAxis(const Vector3d force_axis);

	/**
	 * @brief      Updates the force controlled axis for a hybrid position force
	 *             controller with 1 DoF force and 2 DoF motion
	 * @details    Use this function in situations when the force axis needs to
	 *             be updated (as an estimated normal for example) over time.
	 *             This does not reset the integral terms. In setting up the
	 *             controller for the first time, prefer setForceAxis.
	 *
	 * @param      force_axis  The axis in robot frame coordinates along which
	 *                         the controller behaves as a force controller.
	 */
	void updateForceAxis(const Vector3d force_axis);

	/**
	 * @brief      Sets the motion controlled axis for a hybrid position force
	 *             controller with 2 DoF force and 1 DoF motion
	 * @details    This is the function to use in order to get the controller to
	 *             behave as a Hybrid Force/Motion controller with 2 Dof force.
	 *             The motion is controlled along one axis and the force is
	 *             controlled orthogonally to the motion It can be called
	 *             anytime to change the behavior of the controller and reset
	 *             the integral terms.
	 *
	 * @param[in]  motion_axis  The motion axis
	 * @param      force_axis  The axis in robot frame coordinates along which the
	 *                         controller behaves as a motion controller.
	 */
	void setLinearMotionAxis(const Vector3d motion_axis);

	/**
	 * @brief      Sets the motion controlled axis for a hybrid position force
	 *             controller with 2 DoF force and 1 DoF motion
	 * @details    Use this function in situations when the motion axis needs to
	 *             be updated over time. This does not reset the integral terms.
	 *             In setting up the controller for the first time, prefer
	 *             setMotionAxis.
	 *
	 * @param[in]  motion_axis  The motion axis
	 * @param      force_axis  The axis in robot frame coordinates along which the
	 *                         controller behaves as a motion controller.
	 */
	void updateLinearMotionAxis(const Vector3d motion_axis);

	/**
	 * @brief      Sets the the task as a full 3DoF force controller
	 * @details    This is the function to use in order to get the controller to
	 *             behave as pure Force controller with 3 Dof force. It can be
	 *             called anytime to change the behavior of the controller and
	 *             reset the integral terms.
	 */
	void setFullForceControl();

	/**
	 * @brief      Sets the task as a full 3DoF motion controller (default)
	 * @details    This is the function to use in order to get the controller to
	 *             behave as pure Motion controller with 3 Dof linear motion. It
	 *             is de default behavior of the controller. It can be called
	 *             anytime to change the behavior of the controller and reset
	 *             the integral terms.
	 */
	void setFullLinearMotionControl();

	/**
	 * @brief      Changes the behavior to closed/open loop force control for
	 *             the force controlled directions in the controller. Default is
	 *             open loop.
	 */
	void setClosedLoopForceControl();
	void setOpenLoopForceControl();

	/**
	 * @brief      Enables or disables the passivity based stability for the closed loop
	 *             force control (enabled by default)
	 */
	void enablePassivity();
	void disablePassivity();

	/**
	 * @brief      Resets all the integrated errors used in I terms of the force
	 *             and motion controllers
	 */
	void resetIntegrators();

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

	// inputs to be defined by the user
	Vector3d _desired_position;       // defaults to the current position when the task is created
	Vector3d _desired_velocity;       // defaults to Zero
	Vector3d _desired_acceleration;   // defaults to Zero

	Vector3d _desired_force;

	double _kp;                          // defaults to 50.0 
	double _kv;                          // defaults to 14.0
	double _ki;                          // defaults to 0.0

	double _kp_force;           // defaults to 0.7
	double _ki_force;			// defaults to 1.3
	double _kv_force;			// defaults to 10.0

	bool _use_velocity_saturation_flag;  // defaults to false
	double _saturation_velocity;         // defaults to 0.3 m/s

// trajectory generation via interpolation using Reflexxes Library
#ifdef USING_OTG
	bool _use_interpolation_flag;        // defaults to true

	// default limits for trajectory generation (same in all directions) :
	// Velocity      - 0.3   m/s
	// Acceleration  - 1.0   m/s^2
	// Jerk          - 3.0   m/s^3
#endif

	// internal variables, not to be touched by the user
	string _link_name;
	Affine3d _control_frame;

	// motion quantities
	Vector3d _current_position;
	Vector3d _current_velocity;

	Vector3d _integrated_position_error;

	Matrix3d _sigma_motion;
	
	Vector3d _motion_control;    // the motion control force

	// force quantities
	Affine3d _T_control_to_sensor;

	Vector3d _sensed_force;

	Vector3d _integrated_force_error;
	Matrix3d _sigma_force;

	Vector3d _force_control;    // the force control force

	bool _closed_loop_force_control;

	// passivity related variables
	bool _passivity_enabled;
	double _passivity_observer;
	double _E_correction;
	double _stored_energy_PO;
	queue<double> _PO_buffer_window;
	const int _PO_window_size = 250;

	const int _PO_max_counter = 50;
	int _PO_counter = _PO_max_counter;
	double _vc_squared_sum = 0;

	Vector3d _vc;
	double _Rc;
	double _k_ff;

	// control parameters
	bool _use_isotropic_gains;                 // defaults to true
	Matrix3d _kp_mat;
	Matrix3d _kv_mat;
	Matrix3d _ki_mat;

	int _dynamic_decoupling_type = FULL_DYNAMIC_DECOUPLING;

	// model quantities
	MatrixXd _jacobian;
	MatrixXd _projected_jacobian;
	MatrixXd _Lambda, _Lambda_modified;
	MatrixXd _Jbar;
	MatrixXd _N;

	MatrixXd _URange;
	int _unconstrained_dof;

	bool _first_iteration;

	Vector3d _unit_mass_force;

	VectorXd _step_desired_position;
	VectorXd _step_desired_velocity;
	VectorXd _step_desired_acceleration;

#ifdef USING_OTG
	double _loop_time;
	OTG* _otg;           // default limits: Velocity - 0.3 m/s, acceleration - 1 m/s^2, jerk 3 m/s^3
#endif

};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_POSITION_TASK_H_ */
#endif