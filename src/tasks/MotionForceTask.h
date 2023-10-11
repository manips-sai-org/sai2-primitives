/*
 * MotionForceTask.h
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

#ifndef SAI2_PRIMITIVES_MOTIONFORCETASK_TASK_H_
#define SAI2_PRIMITIVES_MOTIONFORCETASK_TASK_H_

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

class MotionForceTask : public TemplateTask {
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
	MotionForceTask(
		std::shared_ptr<Sai2Model::Sai2Model>& robot, const string& link_name,
		const Affine3d& compliant_frame = Affine3d::Identity(),
		const std::string& task_name = "motion_force_task",
		const bool is_force_motion_parametrization_in_compliant_frame = false,
		const double loop_timestep = 0.001);

	MotionForceTask(
		std::shared_ptr<Sai2Model::Sai2Model>& robot, const string& link_name,
		std::vector<Vector3d> controlled_directions_translation,
		std::vector<Vector3d> controlled_directions_rotation,
		const Affine3d& compliant_frame = Affine3d::Identity(),
		const std::string& task_name = "partial_motion_force_task",
		const bool is_force_motion_parametrization_in_compliant_frame = false,
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
	 * @brief Get the Current Orientation
	 *
	 * @return const Matrix3d& current orientation of the control frame
	 */
	const Matrix3d& getCurrentOrientation() const {
		return _current_orientation;
	}
	/**
	 * @brief Get the Current Angular Velocity
	 *
	 * @return const Vector3d& current angular velocity of the control frame
	 */
	const Vector3d& getCurrentAngularVelocity() const {
		return _current_angular_velocity;
	}

	/**
	 * @brief Get the Sensed Force resolved at the control frame
	 *
	 * @return const Vector3d& current sensed force in the control frame
	 */
	const Vector3d& getSensedForce() const { return _sensed_force; }

	/**
	 * @brief Get the Sensed Moment resolved at the control frame
	 *
	 * @return const Vector3d& current sensed moment in the control frame
	 */
	const Vector3d& getSensedMoment() const { return _sensed_moment; }

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
	const Vector3d& getDesiredAcceleration() const {
		return _desired_acceleration;
	}

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
	void setPosControlGains(double kp_pos, double kv_pos, double ki_pos = 0);
	void setPosControlGains(const Vector3d& kp_pos, const Vector3d& kv_pos,
							const Vector3d& ki_pos = Vector3d::Zero());
	vector<PIDGains> getPosControlGains() const;

	void setOriControlGains(const PIDGains& gains) {
		setOriControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setOriControlGains(double kp_ori, double kv_ori, double ki_ori = 0);
	void setOriControlGains(const Vector3d& kp_ori, const Vector3d& kv_ori,
							const Vector3d& ki_ori = Vector3d::Zero());
	vector<PIDGains> getOriControlGains() const;

	void setForceControlGains(const PIDGains& gains) {
		setForceControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setForceControlGains(double kp_force, double kv_force,
							  double ki_force) {
		_kp_force = kp_force * Matrix3d::Identity();
		_kv_force = kv_force * Matrix3d::Identity();
		_ki_force = ki_force * Matrix3d::Identity();
	}
	vector<PIDGains> getForceControlGains() const {
		return vector<PIDGains>(
			1, PIDGains(_kp_force(0, 0), _kv_force(0, 0), _ki_force(0, 0)));
	}

	void setMomentControlGains(const PIDGains& gains) {
		setMomentControlGains(gains.kp, gains.kv, gains.ki);
	}
	void setMomentControlGains(double kp_moment, double kv_moment,
							   double ki_moment) {
		_kp_moment = kp_moment * Matrix3d::Identity();
		_kv_moment = kv_moment * Matrix3d::Identity();
		_ki_moment = ki_moment * Matrix3d::Identity();
	}
	vector<PIDGains> getMomentControlGains() const {
		return vector<PIDGains>(
			1, PIDGains(_kp_moment(0, 0), _kv_moment(0, 0), _ki_moment(0, 0)));
	}

	/**
	 * @brief Set the Desired Force in robot base frame
	 *
	 * @param desired_force
	 */
	void setDesiredForce(const Vector3d& desired_force) {
		_desired_force = desired_force;
	}

	/**
	 * @brief Get the Desired Force in robot base frame
	 *
	 * @return const Vector3d& desired force in robot base frame
	 */
	Vector3d getDesiredForce() const;

	/**
	 * @brief Set the Desired Moment in robot base frame
	 *
	 * @param desired_moment
	 */
	void setDesiredMoment(const Vector3d& desired_moment) {
		_desired_moment = desired_moment;
	}

	/**
	 * @brief Get the Desired Moment in robot base frame
	 *
	 * @return const Vector3d& desired moment in robot base frame
	 */
	Vector3d getDesiredMoment() const;

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
	void enableVelocitySaturation(const double linear_vel_sat = 0.3,
								  const double angular_vel_sat = M_PI / 3);
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
	 * @brief      Checks if the desired orientation has reched the goal up to a
	 * tolerance
	 *
	 * @param[in]  tolerance  The tolerance
	 * @param[in]  verbose    display info or not
	 *
	 * @return     true if the norm of the orientation error is smaller than the
	 * tolerance
	 */
	bool goalOrientationReached(const double tolerance,
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

	// -------- force control related methods --------

	/**
	 * @brief      Sets the force sensor frame.
	 *
	 * @param[in]  link_name               The link name on which the sensor is
	 * attached
	 * @param[in]  transformation_in_link  The transformation in link of the
	 * sensor
	 */
	void setForceSensorFrame(const string link_name,
							 const Affine3d transformation_in_link);

	/**
	 * @brief      Updates the velues of the sensed force and sensed moment from
	 *             sensor values
	 * @details    Assumes that the sensor is attached to the same link as the
	 *             control frame and that the setSensorFrame finction has been
	 * called. The force and moment values given to this function are assumed to
	 * be in the force sensor frame (values taken directly from the force
	 * sensor) These values are supposed to be the forces that the sensor
	 * applies to the environment (so the opposite of what the sensor feels)
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
	 * @brief Parametrizes the force space and motion space for translational
	 * control The first argument is the dimension of the force space (between
	 * 0 and 3, 0 meaning the whole translation is controlled in motion and 3
	 * meaning the whole translation is controlled in force) and the second
	 * argument is the axis defining the space of dimension one (unused if the
	 * first parameter is 0 or 3, and representing the direction of the force
	 * space is the first parameter is 1, or the direction of the motion space
	 * if the first parameter is 2)
	 *
	 * @param force_space_dimension  between 0 and 3
	 * @param force_or_motion_single_axis non singular axis (unused if
	 * force_space_dimension is 0 or 3)
	 */
	void parametrizeForceMotionSpaces(
		const int force_space_dimension,
		const Vector3d& force_or_motion_single_axis = Vector3d::Zero());

	/**
	 * @brief Parametrizes the moment space and rotational motion space.
	 * The first argument is the dimension of the moment space (between
	 * 0 and 3, 0 meaning the whole rotation is controlled in motion and 3
	 * meaning the whole rotation is controlled in moments) and the second
	 * argument is the axis defining the space of dimension one (unused if the
	 * first parameter is 0 or 3, and representing the direction of the moment
	 * space is the first parameter is 1, or the direction of the rotational
	 * motion space if the first parameter is 2)
	 *
	 * @param moment_space_dimension betawwn 0 and 3
	 * @param moment_or_rot_motion_single_axis non singular axis (unused if
	 * moment_space_dimension is 0 or 3)
	 */
	void parametrizeMomentRotMotionSpaces(
		const int moment_space_dimension,
		const Vector3d& moment_or_rot_motion_single_axis = Vector3d::Zero());

	Matrix3d sigmaForce() const;
	Matrix3d sigmaPosition() const;
	Matrix3d sigmaMoment() const;
	Matrix3d sigmaOrientation() const;

	/**
	 * @brief      Changes the behavior to closed loop force/moment control for
	 * the force controlled directions in the linear/angular parts of the
	 *             controller
	 */
	void setClosedLoopForceControl(
		const bool closed_loop_force_control = true) {
		_closed_loop_force_control = closed_loop_force_control;
		resetIntegratorsLinear();
	}
	void setClosedLoopMomentControl(
		const bool closed_loop_moment_control = true) {
		_closed_loop_moment_control = closed_loop_moment_control;
		resetIntegratorsAngular();
	}

	/**
	 * @brief      Enables or disables the passivity based stability for the
	 * closed loop force control (enabled by default)
	 */
	void enablePassivity() { _POPC_force->enable(); }
	void disablePassivity() { _POPC_force->disable(); }

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

	Matrix3d posSelectionProjector() const {
		return _partial_task_projection.block<3, 3>(0, 0);
	}

	Matrix3d oriSelectionProjector() const {
		return _partial_task_projection.block<3, 3>(3, 3);
	}

private:
	/**
	 * @brief Initial setup of the task, called in the constructor to avoid
	 * duplicated code
	 *
	 */
	void initialSetup();

	// desired pose defaults to the configuration when the task is created
	Vector3d _desired_position;			 // in robot frame
	Matrix3d _desired_orientation;		 // in robot frame
	Vector3d _desired_velocity;			 // in robot frame
	Vector3d _desired_angular_velocity;	 // in robot frame
	Vector3d _desired_acceleration;
	Vector3d _desired_angular_acceleration;

	// gains for motion controller
	// defaults to isptropic 50 for p gains, 14 for d gains and 0 for i gains
	Matrix3d _kp_pos, _kp_ori;
	Matrix3d _kv_pos, _kv_ori;
	Matrix3d _ki_pos, _ki_ori;

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

	MatrixXd _current_task_range;
	int _pos_range, _ori_range;

	Matrix<double, 6, 6> _partial_task_projection;

	VectorXd _unit_mass_force;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_MOTIONFORCETASK_TASK_H_ */
#endif