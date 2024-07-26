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
#include "helper_modules/Sai2PrimitivesCommonDefinitions.h"

using namespace Eigen;
namespace Sai2Primitives {

class JointTask : public TemplateTask {
public:
	struct DefaultParameters {
		static constexpr double kp = 50.0;
		static constexpr double kv = 14.0;
		static constexpr double ki = 0.0;
		static constexpr DynamicDecouplingType dynamic_decoupling_type =
			DynamicDecouplingType::BOUNDED_INERTIA_ESTIMATES;
		static constexpr double bie_threshold = 0.1;
		static constexpr bool use_internal_otg = true;
		static constexpr bool internal_otg_jerk_limited = false;
		static constexpr double otg_max_velocity = M_PI / 3.0;
		static constexpr double otg_max_acceleration = 2.0 * M_PI;
		static constexpr double otg_max_jerk = 10.0 * M_PI;
		static constexpr bool use_velocity_saturation = false;
		static constexpr double saturation_velocity = M_PI / 3.0;
	};

	/**
	 * @brief      Constructor for a full joint task
	 *
	 * @param      robot      A pointer to a Sai2Model object for the robot that
	 *                        is to be controlled
	 * @param[in]  task_name  The task name
	 * @param[in]  loop_timestep  time taken by a control loop. Used only in
	 * trajectory generation
	 */
	JointTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
			  const std::string& task_name = "joint_task",
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
	 * @param task_name
	 * @param loop_timestep
	 */
	JointTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
			  const MatrixXd& joint_selection_matrix,
			  const std::string& task_name = "partial_joint_task",
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
	 *             positions/velocities
	 *
	 * @param      task_joint_torques  the vector to be filled with the new
	 *                                 joint torques to apply for the task
	 */
	VectorXd computeTorques() override;

	/**
	 * @brief      reinitializes the desired and goal states to the current
	 * robot configuration as well as the integrator terms
	 */
	void reInitializeTask() override;

	/**
	 * @brief Get the Joint Selection Matrix. Will be Identity for a full joint
	 * task, and for a partial joint task, it is the constant Jacobian mapping
	 * from the full joint space to the controlled task space
	 *
	 * @return const MatrixXd Joint Selection Matrix that projects the full
	 * joint space to the controlled task space
	 */
	const MatrixXd getJointSelectionMatrix() const { return _joint_selection; }

	int getTaskDof() const { return _task_dof; }
	bool isFullJointTask() const { return _task_dof == getConstRobotModel()->dof(); }

	/**
	 * @brief Get the Current Position
	 *
	 * @return const VectorXd&
	 */
	const VectorXd& getCurrentPosition() { return _current_position; }

	/**
	 * @brief Set the Goal Position
	 *
	 * @param goal_position
	 */
	void setGoalPosition(const VectorXd& goal_position);

	/**
	 * @brief Get the Goal Position
	 *
	 * @return goal position as a VectorXd
	 */
	const VectorXd& getGoalPosition() const { return _goal_position; }

	/**
	 * @brief Get the Current Velocity
	 *
	 * @return const VectorXd&
	 */
	const VectorXd& getCurrentVelocity() { return _current_velocity; }

	/**
	 * @brief Set the Goal Velocity
	 *
	 * @param goal_velocity
	 */
	void setGoalVelocity(const VectorXd& goal_velocity);

	/**
	 * @brief Get the Goal Velocity
	 *
	 * @return goal velocity as a VectorXd
	 */
	const VectorXd& getGoalVelocity() const { return _goal_velocity; }

	/**
	 * @brief Set the Goal Acceleration
	 *
	 * @param goal_acceleration
	 */
	void setGoalAcceleration(const VectorXd& goal_acceleration);

	/**
	 * @brief Get the Goal Acceleration
	 *
	 * @return goal acceleration as a VectorXd
	 */
	const VectorXd& getGoalAcceleration() const { return _goal_acceleration; }

	/**
	 * @brief Get the desired position which is the output of OTG if enabled, or
	 * otherwise equal to the goal position
	 */
	const VectorXd& getDesiredPosition() const { return _desired_position; }

	/**
	 * @brief Get the desired velocity which is the output of OTG if enabled, or
	 * otherwise equal to the goal velocity
	 */
	const VectorXd& getDesiredVelocity() const { return _desired_velocity; }

	/**
	 * @brief Get the desired acceleration which is the output of OTG if
	 * enabled, or otherwise equal to the goal acceleration
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
	 * @brief Get the Nullspace projector of the previous tasks
	 *
	 * @return Eigen::MatrixXd
	 */
	MatrixXd getPreviousTasksNullspace() const override { return _N_prec; }

	/**
	 * @brief Get the nullspace of this and the previous tasks. Concretely, it
	 * is the task nullspace multiplied by the nullspace of the previous tasks
	 *
	 */
	MatrixXd getTaskAndPreviousNullspace() const override {
		return _N * _N_prec;
	}

	/**
	 * @brief Set gains from vectors. The vectors must be either all of size 1
	 * (in which case isotropic gains will be set) or of size robot_dof (for non
	 * isotropic gains).
	 *
	 * @param kp
	 * @param kv
	 * @param ki
	 */
	void setGains(const VectorXd& kp, const VectorXd& kv, const VectorXd& ki);

	/**
	 * @brief Set the gains from vectors with only kp and kv (ki will be set to
	 * 0)
	 *
	 * @param kp
	 * @param kv
	 */
	void setGains(const VectorXd& kp, const VectorXd& kv) {
		setGains(kp, kv, VectorXd::Zero(kp.size()));
	}

	/**
	 * @brief Set isotropic gains from double values
	 *
	 * @param kp
	 * @param kv
	 * @param ki
	 */
	void setGains(const double kp, const double kv, const double ki = 0);

	void setGainsUnsafe(const VectorXd& kp, const VectorXd& kv,
						const VectorXd& ki);

	vector<PIDGains> getGains() const;

	/**
	 * @brief Enables the internal trajectory generation and sets the max
	 * velocity and acceleration (different for each joint if the vectors are of
	 * size robot_dof, otherwise the same for all joints)
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
			max_velocity * VectorXd::Ones(_task_dof),
			max_acceleration * VectorXd::Ones(_task_dof));
	}

	/**
	 * @brief      Enables the internal trajectory generation and sets the max
	 * velocity, acceleration and jerk (different for each joint if the vectors
	 * are of size robot_dof, otherwise the same for all joints)
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
			max_velocity * VectorXd::Ones(_task_dof),
			max_acceleration * VectorXd::Ones(_task_dof),
			max_jerk * VectorXd::Ones(_task_dof));
	}

	/**
	 * @brief      Disables the internal trajectory generation and uses the
	 * goal position, velocity and acceleration directly
	 */
	void disableInternalOtg() { _use_internal_otg_flag = false; }

	bool getInternalOtgEnabled() const { return _use_internal_otg_flag; }

	const OTG_joints& getInternalOtg() const { return *_otg; }

	/**
	 * @brief      Enables the velocity saturation and sets the saturation
	 * velocity (different for each joint if the vectors are of size robot_dof,
	 * otherwise the same for all joints)
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
	void enableVelocitySaturation(const double saturation_velocity);

	/**
	 * @brief      Disables the velocity saturation
	 */
	void disableVelocitySaturation() { _use_velocity_saturation_flag = false; }

	bool getVelocitySaturationEnabled() const {
		return _use_velocity_saturation_flag;
	}

	VectorXd getVelocitySaturationMaxVelocity() const {
		return _saturation_velocity;
	}

	/**
	 * @brief      Sets the dynamic decoupling type. See the enum for more info
	 * on what each type does
	 */
	void setDynamicDecouplingType(const DynamicDecouplingType& type) {
		_dynamic_decoupling_type = type;
	}

	/**
	 * @brief Set the Bounded Inertia Estimate Threshold
	 * 
	 * @param threshold 
	 */
	void setBoundedInertiaEstimateThreshold(const double threshold) {
		if(threshold < 0) {
			_bie_threshold = 0;
		} else {
			_bie_threshold = threshold;
		}
	}

	/**
	 * @brief Get the Bounded Inertia Estimate Threshold value
	 * 
	 * @return double 
	 */
	double getBoundedInertiaEstimateThreshold() const {
		return _bie_threshold;
	}

	/**
	 * @brief	   Returns whether current position is within a tolerance to the goal
	*/
	bool goalPositionReached(const double& tol = 1e-2);

	/**
	 * @brief	Reset integrator error  
	*/
	void resetIntegrators() {
		_integrated_position_error.setZero();
	}

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

private:
	/**
	 * @brief      Initializes the task. Automatically called by the constructor
	 */
	void initialSetup();

	// The goal state of the task is set by the user
	VectorXd _goal_position;
	VectorXd _goal_velocity;
	VectorXd _goal_acceleration;

	// the desired state is the one used in the control equations. It is equal
	// to the output of the OTG interpolation if enabled, and otherwise equal to
	// the goal state
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
	double _bie_threshold;		   // threshold for the bounded inertia estimate
								   // decoupling type

	MatrixXd _joint_selection;	// selection matrix for the joint task, defaults
								// to Identity
	MatrixXd _projected_jacobian;
	MatrixXd _N;
	MatrixXd _current_task_range;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_JOINT_TASK_H_ */
#endif