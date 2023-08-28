/*
 * MotionForceTask.h
 *
 *      This class creates a 6Dof position + orientation hybrid controller for a robotic manipulator using operational space formulation and an underlying PID compensator.
 *      If used for hybrid position force control, assumes a force sensor is attached to the same link as the control frame and the force sensed values are given in sensor frame.
 *      Besides, the force sensed and moment sensed are assumed to be the force and moment that the robot applies to the environment.
 *      It requires a robot model parsed from a urdf file to a Sai2Model object, as well as the definition of a control frame
 *      as a link at which the frame is attached, and an affine transform that determines the position and orientation of the control frame in this link
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_MOTIONFORCETASK_TASK_H_
#define SAI2_PRIMITIVES_MOTIONFORCETASK_TASK_H_

#include "Sai2Model.h"
#include <helper_modules/POPCExplicitForceControl.h>
#include <Eigen/Dense>
#include <string>
// #include <chrono>
#include <memory>
// #include <queue> 

#ifdef USING_OTG
	#include "trajectory_generation/OTG_posori.h"
#endif

using namespace Eigen;
using namespace std;

namespace Sai2Primitives
{

struct PIDGains {
	double kp;
	double kv;
	double ki;

	PIDGains(double kp, double kv, double ki) : kp(kp), kv(kv), ki(ki) {}
};

class MotionForceTask
{

enum DynamicDecouplingType
{
	FULL_DYNAMIC_DECOUPLING,            // use the real Lambda matrix
	PARTIAL_DYNAMIC_DECOUPLING,         // Use Lambda for position part, Identity for orientation and Zero for cross coupling
	IMPEDANCE,                          // use Identity for the mass matrix
	BOUNDED_INERTIA_ESTIMATES,           // Use a Lambda computed from a saturated joint space mass matrix
};

public:

	//------------------------------------------------
	// Constructors
	//------------------------------------------------

	/**
	 * @brief      Constructor that takes an Affine3d matrix for definition of
	 *             the control frame. Creates a full position controller by
	 *             default.
	 *
	 * @param      robot          A pointer to a Sai2Model object for the robot
	 *                            that is to be controlled
	 * @param      link_name      The name of the link in the robot at which to
	 *                            attach the control frame
	 * @param      control_frame  The position and orientation of the control
	 *                            frame in local link coordinates
	 * @param[in]  loop_timestep      time taken by a control loop. Used in
	 *                            trajectory generation and integral control
	 */
	MotionForceTask(Sai2Model::Sai2Model* robot, 
		            const string link_name, 
		            const Affine3d control_frame = Affine3d::Identity(),
		            const double loop_timestep = 0.001);

	/**
	 * @brief      Constructor that takes a Vector3d for definition of the
	 *             control frame position and a Matrix3d for the frame
	 *             orientation. Creates a full position controller by default.
	 *
	 * @param      robot        A pointer to a Sai2Model object for the robot
	 *                          that is to be controlled
	 * @param      link_name    The name of the link in the robot at which to
	 *                          attach the control frame
	 * @param      pos_in_link  The position the control frame in local link
	 *                          coordinates
	 * @param      rot_in_link  The orientation of the control frame in local
	 *                          link coordinates
	 * @param[in]  loop_timestep    time taken by a control loop. Used in
	 *                          trajectory generation and integral control
	 */
	MotionForceTask(Sai2Model::Sai2Model* robot, 
		            const string link_name, 
		            const Vector3d pos_in_link, 
		            const Matrix3d rot_in_link = Matrix3d::Identity(),
		            const double loop_timestep = 0.001);

	//------------------------------------------------
	// Getters Setters
	//------------------------------------------------

	const Vector3d& getCurrentPosition() const { return _current_position; }
	const Vector3d& getCurrentVelocity() const { return _current_velocity; }

	const Matrix3d& getCurrentOrientation() const { return _current_orientation; }
	const Vector3d& getCurrentAngularVelocity() const { return _current_angular_velocity; }

	const Vector3d& getSensedForce() const {return _sensed_force;}

	const MatrixXd& getN() const {return _N;}

	void setDesiredPosition(const Vector3d& desired_position) {
		_desired_position = desired_position;
	}
	const Vector3d& getDesiredPosition() const { return _desired_position; }

	void setDesiredOrientation(const Matrix3d& desired_orientation) {
		_desired_orientation = desired_orientation;
	}
	Matrix3d getDesiredOrientation() const { return _desired_orientation; }

	void setDesiredVelocity(const Vector3d& desired_velocity) {
		_desired_velocity = desired_velocity;
	}
	const Vector3d& getDesiredVelocity() const { return _desired_velocity; }

	void setDesiredAngularVelocity(const Vector3d& desired_angvel) {
		_desired_angular_velocity = desired_angvel;
	}
	const Vector3d& getDesiredAngularVelocity() const {
		return _desired_angular_velocity;
	}

	void setDesiredAcceleration(const Vector3d& desired_acceleration) {
		_desired_acceleration = desired_acceleration;
	}
	const Vector3d& getDesiredAcceleration() const { return _desired_acceleration; }

	void setDesiredAngularAcceleration(const Vector3d& desired_angaccel) {
		_desired_angular_acceleration = desired_angaccel;
	}
	const Vector3d& getDesiredAngularAcceleration() const {
		return _desired_angular_acceleration;
	}

	// Gains for motion controller
	void setPosControlGains(const PIDGains& gains) {
		setPosControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setPosControlGains(double kp_pos, double kv_pos, double ki_pos = 0) {
		_kp_pos = kp_pos;
		_kv_pos = kv_pos;
		_ki_pos = ki_pos;
	}
	PIDGains getPosControlGains() const {
		return PIDGains(_kp_pos, _kv_pos, _ki_pos);
	}

	void setOriControlGains(const PIDGains& gains) {
		setOriControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setOriControlGains(double kp_ori, double kv_ori, double ki_ori = 0) {
		_kp_ori = kp_ori;
		_kv_ori = kv_ori;
		_ki_ori = ki_ori;
	}
	PIDGains getOriControlGains() const {
		return PIDGains(_kp_ori, _kv_ori, _ki_ori);
	}

	void setForceControlGains(const PIDGains& gains) {
		setForceControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setForceControlGains(double kp_force, double kv_force,
							  double ki_force) {
		_kp_force = kp_force;
		_kv_force = kv_force;
		_ki_force = ki_force;
	}
	PIDGains getForceControlGains() const {
		return PIDGains(_kp_force, _kv_force, _ki_force);
	}

	void setMomentControlGains(const PIDGains& gains) {
		setMomentControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setMomentControlGains(double kp_moment, double kv_moment,
							   double ki_moment) {
		_kp_moment = kp_moment;
		_kv_moment = kv_moment;
		_ki_moment = ki_moment;
	}
	PIDGains getMomentControlGains() const {
		return PIDGains(_kp_moment, _kv_moment, _ki_moment);
	}

	void setDesiredForce(const Vector3d& desired_force) {
		_desired_force = desired_force;
	}
	const Vector3d& getDesiredForce() const { return _desired_force; }

	void setDesiredMoment(const Vector3d& desired_moment) {
		_desired_moment = desired_moment;
	}
	const Vector3d& getDesiredMoment() const { return _desired_moment; }

	// Velocity saturation flag and saturation values
	void enableVelocitySaturation(double linear_vel_sat,
								  double angular_vel_sat) {
		_use_velocity_saturation_flag = true;
		_linear_saturation_velocity = linear_vel_sat;
		_angular_saturation_velocity = angular_vel_sat;
	}
	void disableVelocitySaturation() { _use_velocity_saturation_flag = false; }

	bool getVelocitySaturationEnabled() const {
		return _use_velocity_saturation_flag;
	}
	double getLinearSaturationVelocity() const {
		return _linear_saturation_velocity;
	}
	double getAngularSaturationVelocity() const {
		return _angular_saturation_velocity;
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
	 *             nullspace matrix of this task alone, one needs to compute _N *
	 *             _N_prec.inverse().
	 *
	 * @param      N_prec  The nullspace matrix of all the higher priority
	 *                     tasks. If this is the highest priority task, use
	 *                     identity of size n*n where n in the number of DoF of
	 *                     the robot.
	 */
	void updateTaskModel(const MatrixXd N_prec);

	/**
	 * @brief      Computes the torques associated with this task.
	 * @details    Computes the torques taking into account the last model
	 *             update and updated values for the robot joint
	 *             positions/velocities assumes the desired orientation and
	 *             angular velocity has been updated
	 *
	 * @param      task_joint_torques  the vector to be filled with the new
	 *                                 joint torques to apply for the task
	 */
	VectorXd computeTorques();

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
	
	/**
	 * @brief      Checks if the desired orientation has reched the goal up to a tolerance
	 *
	 * @param[in]  tolerance  The tolerance
	 * @param[in]  verbose    display info or not
	 *
	 * @return     true if the norm of the orientation error is smaller than the tolerance
	 */
	bool goalOrientationReached(const double tolerance, const bool verbose = false);


	// ---------- set dynamic decoupling type for the controller  ----------------
	void setDynamicDecouplingFull();
	void setDynamicDecouplingPartial();
	void setDynamicDecouplingNone();
	void setDynamicDecouplingBIE();

	void setNonIsotropicGainsPosition(const Matrix3d& frame, const Vector3d& kp, 
		const Vector3d& kv, const Vector3d& ki);
	void setNonIsotropicGainsOrientation(const Matrix3d& frame, const Vector3d& kp, 
		const Vector3d& kv, const Vector3d& ki);

	void setIsotropicGainsPosition(const double kp, const double kv, const double ki);
	void setIsotropicGainsOrientation(const double kp, const double kv, const double ki);

	// -------- force control related methods --------

	/**
	 * @brief      Sets the force sensor frame.
	 *
	 * @param[in]  link_name               The link name on which the sensor is attached
	 * @param[in]  transformation_in_link  The transformation in link of the sensor
	 */
	void setForceSensorFrame(const string link_name, const Affine3d transformation_in_link);

	/**
	 * @brief      Updates the velues of the sensed force and sensed moment from
	 *             sensor values
	 * @details    Assumes that the sensor is attached to the same link as the
	 *             control frame and that the setSensorFrame finction has been called. 
	 *             The force and moment values given to
	 *             this function are assumed to be in the force sensor frame
	 *             (values taken directly from the force sensor) These values
	 *             are supposed to be the forces that the sensor applies to the
	 *             environment (so the opposite of what the sensor feels)
	 *
	 * @param      sensed_force_sensor_frame   The sensed force as the force
	 *                                         that the sensor applies to the
	 *                                         environment in sensor frame
	 * @param      sensed_moment_sensor_frame  The sensed moment as the moment
	 *                                         that the sensor applies to the
	 *                                         environment in sensor frame
	 */
	void updateSensedForceAndMoment(const Vector3d sensed_force_sensor_frame, 
								    const Vector3d sensed_moment_sensor_frame);

	/**
	 * @brief      Sets the force controlled axis for a hybrid position force
	 *             controller with 1 DoF force and 2 DoF motion
	 * @details    This is the function to use in order to get the position part
	 *             of the controller behave as a Hybrid Force/Motion controller
	 *             with 1 Dof force. The motion is controlled orthogonally to
	 *             the force. It can be called anytime to change the behavior of
	 *             the controller and reset the integral terms.
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
	 * @details    This is the function to use in order to get the position part
	 *             of the controller behave as a Hybrid Force/Motion controller
	 *             with 2 Dof force. The motion is controlled along one axis and
	 *             the force is controlled orthogonally to the motion It can be
	 *             called anytime to change the behavior of the controller and
	 *             reset the integral terms.
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
	 * @brief      Sets the linear part of the task as a full 3DoF force
	 *             controller
	 * @details    This is the function to use in order to get the position part
	 *             of the controller behave as pure Force controller with 3 Dof
	 *             force. It can be called anytime to change the behavior of the
	 *             controller and reset the integral terms.
	 */
	void setFullForceControl();

	/**
	 * @brief      Sets the linear part of the task as a full 3DoF motion
	 *             controller
	 * @details    This is the function to use in order to get the position part
	 *             of the controller behave as pure Motion controller with 3 Dof
	 *             linear motion. It is de default behavior of the controller.
	 *             It can be called anytime to change the behavior of the
	 *             controller and reset the integral terms.
	 */
	void setFullLinearMotionControl();

	/**
	 * @brief      Sets the moment controlled axis for a hybrid orientation
	 *             moment controller with 1 DoF moment and 2 DoF orientation
	 * @details    This is the function to use in order to get the orientation
	 *             part of the controller behave as a Hybrid Force/Motion
	 *             controller with 1 Dof moment. The rotational motion is
	 *             controlled orthogonally to the moment. 
	 *             
	 *             It can be called anytime to change the behavior of the controller 
	 *             and reset the integral terms.
	 *
	 * @param      moment_axis  The axis in robot frame coordinates along
	 *                          which the controller behaves as a moment
	 *                          controller.
	 */
	void setMomentAxis(const Vector3d moment_axis);

	/**
	 * @brief      Sets the moment controlled axis for a hybrid orientation
	 *             moment controller with 1 DoF moment and 2 DoF orientation
	 * @details    Use this function in situations when the moment axis needs to
	 *             be updated over time. This does not reset the integral terms.
	 *             
	 *             In setting up the controller for the first time, prefer
	 *             setMomentAxis.
	 *
	 * @param      moment_axis  The axis in robot frame coordinates along
	 *                          which the controller behaves as a moment
	 *                          controller.
	 */
	void updateMomentAxis(const Vector3d moment_axis);

	/**
	 * @brief      Sets the angular movement controlled axis for a hybrid
	 *             orientation moment controller with 2 DoF moment and 1 DoF
	 *             motion
	 * @details    This is the function to use in order to get the orientation
	 *             part of the controller behave as a Hybrid Force/Motion
	 *             controller with 2 Dof moment. The rotational motion is
	 *             controlled along one axis and the moment is controlled
	 *             orthogonally to the motion.
	 *             
	 *             It can be called anytime to change
	 *             the behavior of the controller and reset the integral terms.
	 *
	 * @param[in]  motion_axis  The motion axis
	 * @param      force_axis  The axis in robot frame coordinates along which the
	 *                         controller behaves as a rotational motion controller.
	 */
	void setAngularMotionAxis(const Vector3d motion_axis);

	/**
	 * @brief      Sets the angular movement controlled axis for a hybrid
	 *             orientation moment controller with 2 DoF moment and 1 DoF
	 *             motion
	 * @details    Use this function in situations when the angular motion axis
	 *             needs to be updated over time. This does not reset the
	 *             integral terms. 
	 *             
	 *             In setting up the controller for the first
	 *             time, prefer setAngularMotionAxis.
	 *
	 * @param[in]  motion_axis  The motion axis
	 * @param      force_axis  The axis in robot frame coordinates along which the
	 *                         controller behaves as a rotational motion controller.
	 */
	void updateAngularMotionAxis(const Vector3d motion_axis);

	/**
	 * @brief      Sets the angular part of the task as a full 3DoF moment
	 *             controller
	 * @details    This is the function to use in order to get the orientation
	 *             part of the controller behave as pure moment controller with
	 *             3 Dof moment. It can be called anytime to change the behavior
	 *             of the controller and reset the integral terms.
	 */
	void setFullMomentControl();

	/**
	 * @brief      Sets the angular part of the task as a full 3DoF motion
	 *             controller
	 * @details    This is the function to use in order to get the orientation
	 *             part of the controller behave as pure Motion controller with
	 *             3 Dof angular motion. It is de default behavior of the
	 *             controller. It can be called anytime to change the behavior
	 *             of the controller and reset the integral terms.
	 */
	void setFullAngularMotionControl();

	/**
	 * @brief      Changes the behavior to closed loop force/moment control for the
	 *             force controlled directions in the linear/angular parts of the
	 *             controller
	 */
	void setClosedLoopForceControl();
	void setOpenLoopForceControl();
	void setClosedLoopMomentControl();
	void setOpenLoopMomentControl();

	/**
	 * @brief      Enables or disables the passivity based stability for the closed loop
	 *             force control (enabled by default)
	 */
	void enablePassivity();
	void disablePassivity();

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

	/**
	 * @brief      Resets the integrated errors used in I terms for angular part
	 *             of task (orientation_integrated_error and
	 *             moment_integrated_error)
	 */
	void resetIntegratorsAngular();

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

	// inputs to be defined by the user
private:
	// desired pose defaults to the configuration when the task is created
	Vector3d _desired_position;           // in robot frame
	Matrix3d _desired_orientation;        // in robot frame
	Vector3d _desired_velocity;           // in robot frame
	Vector3d _desired_angular_velocity;   // in robot frame
	Vector3d _desired_acceleration;
	Vector3d _desired_angular_acceleration;

	// gains for motion controller
	// defaults to 50 for p gains, 14 for d gains and 0 fir i gains
	double _kp_pos, _kp_ori;
	double _kv_pos, _kv_ori;
	double _ki_pos, _ki_ori;

	// gains for the closed loop force controller
	// by default, the force controller is open loop
	// to set the behavior to closed loop controller, use the functions setClosedLoopForceControl and setClosedLoopMomentControl.
	// the closed loop force controller is a PI controller with feedforward force and velocity based damping.
	// gains default to 1 for p gains, 0.7 for i gains and 10 for d gains
	double _kp_force, _kp_moment;
	double _kv_force, _kv_moment;
	double _ki_force, _ki_moment;
	
	// desired force and moment for the force part of the controller
	// defaults to Zero
	Vector3d _desired_force;   // robot frame
	Vector3d _desired_moment;  // robot frame

	// velocity saturation is off by default
	bool _use_velocity_saturation_flag;
	double _linear_saturation_velocity;   // defaults to 0.3 m/s
	double _angular_saturation_velocity;  // defaults to PI/3 Rad/s

// trajectory generation via interpolation using Reflexxes Library
// on by defalut
#ifdef USING_OTG
	bool _use_interpolation_flag; 

	// default limits for trajectory generation (same in all directions) :
	// Linear Velocity       - 0.3   m/s
	// Linear Acceleration   - 1.0   m/s^2
	// Linear Jerk           - 3.0   m/s^3
	// Angular Velocity      - PI/3  Rad/s
	// Angular Acceleration  - PI    Rad/s^2
	// Angular Jerk          - 3PI   Rad/s^3
#endif

	Sai2Model::Sai2Model* _robot;
	double _loop_timestep;

	Eigen::VectorXd _task_force;
	Eigen::MatrixXd _N_prec;

	// internal variables, not to be touched by the user
	string _link_name;
	Affine3d _control_frame;   // in link_frame

	// motion quantities
	Vector3d _current_position;      // robot frame
	Matrix3d _current_orientation;   // robot frame

	Vector3d _current_velocity;           // robot frame
	Vector3d _current_angular_velocity;   // robot frame

	Vector3d _orientation_error;               // robot frame
	Vector3d _integrated_orientation_error;    // robot frame
	Vector3d _integrated_position_error;       // robot frame
	
	Matrix3d _sigma_position;        // robot frame
	Matrix3d _sigma_orientation;     // robot frame

	// force quantities
	Affine3d _T_control_to_sensor;  

	Vector3d _sensed_force;    // robot frame
	Vector3d _sensed_moment;   // robot frame

	Vector3d _integrated_force_error;    // robot frame
	Vector3d _integrated_moment_error;   // robot frame

	Matrix3d _sigma_force;     // robot frame
	Matrix3d _sigma_moment;    // robot frame

	bool _closed_loop_force_control;
	bool _closed_loop_moment_control;
	double _k_ff;

	// POPC for closed loop force control
	std::unique_ptr<POPCExplicitForceControl> _POPC_force;

	// linear control inputs
	Vector3d _linear_motion_control;
	Vector3d _linear_force_control;

	// control parameters
	bool _use_isotropic_gains_position;                 // defaults to true
	bool _use_isotropic_gains_orientation;              // defaults to true
	Matrix3d _kp_pos_mat, _kp_ori_mat;
	Matrix3d _kv_pos_mat, _kv_ori_mat;
	Matrix3d _ki_pos_mat, _ki_ori_mat;

	int _dynamic_decoupling_type = BOUNDED_INERTIA_ESTIMATES;

	// model quantities
	MatrixXd _jacobian;
	MatrixXd _projected_jacobian;
	MatrixXd _prev_projected_jacobian;
	MatrixXd _Lambda, _Lambda_modified;
	MatrixXd _Jbar;
	MatrixXd _N;

	MatrixXd _URange_pos;
	MatrixXd _URange_ori;
	MatrixXd _URange;
	int _pos_dof, _ori_dof;

	VectorXd _unit_mass_force;

	Vector3d _step_desired_position;
	Vector3d _step_desired_velocity;
	Matrix3d _step_desired_orientation;
	Vector3d _step_orientation_error;
	Vector3d _step_desired_angular_velocity;
	Vector3d _step_desired_acceleration;
	Vector3d _step_desired_angular_acceleration;

#ifdef USING_OTG
	OTG_posori* _otg;
#endif

};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_MOTIONFORCETASK_TASK_H_ */
#endif