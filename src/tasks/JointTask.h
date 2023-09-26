/*
 * JointTask.h
 *
 *      This class creates a joint controller for a robotic manipulator using
 * dynamic decoupling and an underlying PID compensator. It requires a robot
 * model parsed from a urdf file to a Sai2Model object. It does not support
 * spherical joints
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_JOINT_TASK_H_
#define SAI2_PRIMITIVES_JOINT_TASK_H_

#include <helper_modules/OTG_joints.h>

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <string>

#include "Sai2Model.h"
#include "TemplateTask.h"

using namespace Eigen;
namespace Sai2Primitives {

class JointTask : public TemplateTask {
public:
	enum DynamicDecouplingType {
		FULL_DYNAMIC_DECOUPLING,	// use the real Mass matrix
		BOUNDED_INERTIA_ESTIMATES,	// use a Mass matrix computed from
									// saturating the minimal values of the Mass
									// Matrix
		IMPEDANCE,					// use Identity for the Mass matrix
	};

	/**
	 * @brief      Constructor for a full joint task
	 *
	 * @param      robot      A pointer to a Sai2Model object for the robot that
	 *                        is to be controlled
	 * @param[in]  loop_timestep  time taken by a control loop. Used only in
	 * trajectory generation
	 */
	JointTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
			  const double loop_timestep = 0.001);

	/**
	 * @brief Constor for a partial joint task, taking as parameter the joint
	 * selection matrix. Ths joint selection matrix is the constant Jacobian
	 * mapping from the full joint space to the controlled task space, where
	 * each row is an independent degree of freedom in the robot joint space.
	 * Typically, it will be a subset of identity matrix rows
	 *
	 * @param robot
	 * @param joint_selection_matrix
	 * @param loop_timestep
	 */
	JointTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
			  const MatrixXd& joint_selection_matrix,
			  const double loop_timestep = 0.001);

	/**
	 * @brief      update the task model (only _N_prec for a joint task)
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
	 *             positions/velocities assumes the desired position and
	 *             velocity has been updated
	 *
	 * @param      task_joint_torques  the vector to be filled with the new
	 *                                 joint torques to apply for the task
	 */
	VectorXd computeTorques() override;

	/**
	 * @brief      reinitializes the desired state to the current robot
	 *             configuration as well as the integrator terms
	 */
	void reInitializeTask();

	/**
	 * @brief Get the Joint Selection Matrix. Will be Identity for a full joint
	 * task, and for a partial joint task, it is the constant Jacobian mapping
	 * from the full joint space to the controlled task space
	 *
	 * @return const MatrixXd Joint Selection Matrix that projects the full
	 * joint space to the controlled task space
	 */
	const MatrixXd getJointSelectionMatrix() const { return _joint_selection; }

	/**
	 * @brief Get the Current Position
	 *
	 * @return const VectorXd&
	 */
	const VectorXd& getCurrentPosition() { return _current_position; }

	/**
	 * @brief Set the Desired Position
	 *
	 * @param desired_position
	 */
	void setDesiredPosition(const VectorXd& desired_position);

	/**
	 * @brief Get the Desired Position
	 *
	 * @return desired position as a VectorXd
	 */
	const VectorXd& getDesiredPosition() const { return _desired_position; }

	/**
	 * @brief Set the Desired Velocity
	 *
	 * @param desired_velocity
	 */
	void setDesiredvelocity(const VectorXd& desired_velocity);

	/**
	 * @brief Get the Desired Velocity
	 *
	 * @return desired velocity as a VectorXd
	 */
	const VectorXd& getDesiredvelocity() const { return _desired_velocity; }

	/**
	 * @brief Set the Desired Acceleration
	 *
	 * @param desired_acceleration
	 */
	void setDesiredAcceleration(const VectorXd& desired_acceleration);

	/**
	 * @brief Get the Desired Acceleration
	 *
	 * @return desired acceleration as a VectorXd
	 */
	const VectorXd& getDesiredAcceleration() const {
		return _desired_acceleration;
	}

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

	/**
	 * @brief Set non isotropic gains
	 *
	 * @param kp
	 * @param kv
	 * @param ki
	 */
	void setGains(const VectorXd& kp, const VectorXd& kv, const VectorXd& ki);

	/**
	 * @brief Set non isotropic gains with ki = 0
	 *
	 * @param kp
	 * @param kv
	 */
	void setGains(const VectorXd& kp, const VectorXd& kv) {
		setGains(kp, kv, VectorXd::Zero(getConstRobotModel()->dof()));
	}

	/**
	 * @brief Set isotropic gains
	 *
	 * @param kp
	 * @param kv
	 * @param ki
	 */
	void setGains(const double kp, const double kv, const double ki = 0);

	/**
	 * @brief Enables the internal trajectory generation and sets the max
	 * velocity and acceleration (different for each joint)
	 *
	 * @param max_velocity
	 * @param max_acceleration
	 */
	void enableInternalOtgAccelerationLimited(const VectorXd& max_velocity,
											  const VectorXd& max_acceleration);

	/**
	 * @brief Enables the internal trajectory generation and sets the max
	 * velocity and acceleration (same for all joints)
	 *
	 * @param max_velocity
	 * @param max_acceleration
	 */
	void enableInternalOtgAccelerationLimited(const double max_velocity,
											  const double max_acceleration) {
		enableInternalOtgAccelerationLimited(
			max_velocity * VectorXd::Ones(getConstRobotModel()->dof()),
			max_acceleration * VectorXd::Ones(getConstRobotModel()->dof()));
	}

	/**
	 * @brief      Enables the internal trajectory generation and sets the max
	 * velocity, acceleration and jerk (different for each joint)
	 *
	 * @param[in]  max_velocity      The maximum velocity
	 * @param[in]  max_acceleration  The maximum acceleration
	 * @param[in]  max_jerk          The maximum jerk
	 */
	void enableInternalOtgJerkLimited(const VectorXd& max_velocity,
									  const VectorXd& max_acceleration,
									  const VectorXd& max_jerk);

	/**
	 * @brief 	Enables the internal trajectory generation and sets the max
	 * velocity, acceleration and jerk (same for all joints)
	 *
	 * @param max_velocity
	 * @param max_acceleration
	 * @param max_jerk
	 */
	void enableInternalOtgJerkLimited(const double max_velocity,
									  const double max_acceleration,
									  const double max_jerk) {
		enableInternalOtgJerkLimited(
			max_velocity * VectorXd::Ones(getConstRobotModel()->dof()),
			max_acceleration * VectorXd::Ones(getConstRobotModel()->dof()),
			max_jerk * VectorXd::Ones(getConstRobotModel()->dof()));
	}

	/**
	 * @brief      Disables the internal trajectory generation and uses the
	 * desired position, velocity and acceleration directly
	 */
	void disableInternalOtg() { _use_internal_otg_flag = false; }

	/**
	 * @brief      Enables the velocity saturation and sets the saturation
	 * velocity (different for each joint)
	 *
	 * @param[in]  saturation_velocity  The saturation velocity
	 */
	void enableVelocitySaturation(const VectorXd& saturation_velocity);

	/**
	 * @brief      Enables the velocity saturation and sets the saturation
	 * velocity (same for all joints)
	 *
	 * @param[in]  saturation_velocity  The saturation velocity
	 */
	void enableVelocitySaturation(const double saturation_velocity) {
		enableVelocitySaturation(saturation_velocity *
								 VectorXd::Ones(getConstRobotModel()->dof()));
	}

	/**
	 * @brief      Disables the velocity saturation
	 */
	void disableVelocitySaturation() {
		_use_velocity_saturation_flag = false;
	}

	/**
	 * @brief      Sets the dynamic decoupling type. See the enum for more info
	 * on what each type does
	 */
	void setDynamicDecouplingType(const DynamicDecouplingType& type) {
		_dynamic_decoupling_type = type;
	}

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

private:
	/**
	 * @brief      Initializes the task. Automatically called by the constructor
	 */
	void initialSetup();

	// desired controller state
	VectorXd _desired_position;
	VectorXd _desired_velocity;
	VectorXd _desired_acceleration;

	// current state from robot model
	VectorXd _current_position;
	VectorXd _current_velocity;

	// state variables for the integrator
	VectorXd _integrated_position_error;

	// controller gains
	bool _are_gains_isotropic;
	MatrixXd _kp;  // 50 by default on all axes
	MatrixXd _kv;  // 14 by default on all axes
	MatrixXd _ki;  // 0 by default on all axes

	// velocity saturation related variables
	bool _use_velocity_saturation_flag;	 // disabled by default
	VectorXd _saturation_velocity;

	// internal trajectory generation. Defaults to a velocity and acceleration
	// limited trajectory generation, with max velocity being PI/3 and max
	// acceleration being PI on all axes
	bool _use_internal_otg_flag;  // defaults to true
	shared_ptr<OTG_joints> _otg;

	// model related variables
	int _task_dof;
	MatrixXd _N_prec;			   // nullspace of the previous tasks
	MatrixXd _M_partial;		   // mass matrix for the partial joint task
	MatrixXd _M_partial_modified;  // modified mass matrix according to the
								   // decoupling type
	DynamicDecouplingType
		_dynamic_decoupling_type;  // defaults to BOUNDED_INERTIA_ESTIMATES. See
								   // the enum for more details

	MatrixXd _joint_selection;	// selection matrix for the joint task, defaults
								// to Identity
	MatrixXd _projected_jacobian;
	MatrixXd _Jbar;
	MatrixXd _N;
	MatrixXd _URange;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_JOINT_TASK_H_ */
#endif