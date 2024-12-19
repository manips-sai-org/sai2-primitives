/*
 * JointLimitAvoidanceTask.h
 *
 *      This class creates a joint limit avoidance task controller for position and velocity limits
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI_PRIMITIVES_JOINT_LIMIT_AVOIDANCE_TASK_H_
#define SAI_PRIMITIVES_JOINT_LIMIT_AVOIDANCE_TASK_H_

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>

#include "SaiModel.h"
#include "TemplateTask.h"
#include "helper_modules/SaiPrimitivesCommonDefinitions.h"

using namespace Eigen;
namespace SaiPrimitives {

class JointLimitAvoidanceTask : public TemplateTask {
public:
	struct DefaultParameters {
		static constexpr bool enabled = true;
		static constexpr double kv = 20.0;
		static constexpr double position_z1_to_limit = 9 * M_PI / 180.0;
		static constexpr double position_z2_to_limit = 6 * M_PI / 180.0;
		static constexpr double velocity_z1_to_limit = 0.5;
		static constexpr double velocity_z2_to_limit = 0.3;
		static constexpr double max_torque_ratio_pos_limit = 1.0;
		static constexpr double max_torque_ratio_vel_limit = 0.05;
	};

	enum LimitStatus { OFF, POS_Z1, POS_Z2, VEL_Z1, VEL_Z2 };
	enum LimitDirection { POSITIVE, NEGATIVE };

	JointLimitAvoidanceTask(
		std::shared_ptr<SaiModel::SaiModel>& robot,
		const std::string& task_name = "joint_limit_avoidance_task");

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
	 * @return Eigen::VectorXd the joint task torques
	 */
	VectorXd computeTorques() override;

	/**
	 * @brief Computes the joint torques associated with this control task.
	 *
	 * @param tau_tasks the control torques from all the non constraint tasks
	 * @return Eigen::VectorXd the joint task torques
	 */
	VectorXd computeTorques(const Eigen::VectorXd& tau_tasks) override;

	/**
	 * @brief      reinitializes the desired and goal states to the current
	 * robot configuration as well as the integrator terms
	 */
	void reInitializeTask() override;

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

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

private:
	/**
	 * @brief      Initializes the task. Automatically called by the constructor
	 */
	void initialSetup();

	void computeJointSelectionMatrix();
	void updateLimitStatus();

	void verifyValidityPerJoint();


	VectorXd _torque_limit;
	VectorXd _position_limit;
	VectorXd _velocity_limit;

	bool _enabled;
	std::vector<LimitStatus> _limit_status;
	std::vector<LimitDirection> _limit_direction;
	std::vector<double> _limit_value;
	std::vector<double> _torque_limit_value;
	double _kv;
	double _position_z1_to_limit;
	double _position_z2_to_limit;
	double _velocity_z1_to_limit;
	double _velocity_z2_to_limit;
	double _max_torque_ratio_pos_limit;
	double _max_torque_ratio_vel_limit;
	std::map<std::string, bool> _joint_pos_limit_valid;
	std::map<std::string, bool> _joint_vel_limit_valid;

	// model related variables
	int _active_constraints;
	MatrixXd _N_prec;
	MatrixXd _M_partial;

	MatrixXd _joint_selection;
	MatrixXd _projected_jacobian;
	MatrixXd _N, _N_unconstrained;
	MatrixXd _current_task_range;

};

} /* namespace SaiPrimitives */

#endif /* SAI_PRIMITIVES_JOINT_LIMIT_AVOIDANCE_TASK_H_ */