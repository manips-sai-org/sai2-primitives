/*
 * TemplateTask.h
 *
 *      Template task for Sai2 tasks
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_TEMPLATE_TASK_H_
#define SAI2_PRIMITIVES_TEMPLATE_TASK_H_

#include <Sai2Model.h>

#include <Eigen/Dense>
#include <memory>

namespace Sai2Primitives {

enum TaskType {
	UNDEFINED,
	JOINT_TASK,
	MOTION_FORCE_TASK,
};

class TemplateTask {
public:
	TemplateTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
				 const std::string& task_name, const TaskType task_type,
				 const double loop_timestep)
		: _robot(robot),
		  _task_name(task_name),
		  _task_type(task_type),
		  _loop_timestep(loop_timestep) {}

	/**
	 * @brief update the task model (only _N_prec for a joint task)
	 *
	 * @param N_prec The nullspace matrix of all the higher priority tasks. If
	 * this is the highest priority task, use identity of size n*n where n in
	 * the number of DoF of the robot.
	 */
	virtual void updateTaskModel(const Eigen::MatrixXd& N_prec) = 0;

	/**
	 * @brief Computes the joint torques associated with this control task.
	 *
	 * @return Eigen::VectorXd the joint task torques
	 */
	virtual Eigen::VectorXd computeTorques() = 0;

	/**
	 * @brief Computes the joint torques associated with this control task, and
	 * feedforward compensates the disturbances due to the previous tasks.
	 *
	 * @param tau_prec the control torques from the frevious tasks in the hierarchy
	 * @return Eigen::VectorXd the joint task torques
	 */
	virtual Eigen::VectorXd computeTorques(const Eigen::VectorXd& tau_prec) = 0;

	/**
	 * @brief Re initializes the task by setting the desired state to the
	 * current state.
	 *
	 */
	virtual void reInitializeTask() = 0;

	/**
	 * @brief Get the Nullspace projector of this task only (N associated with
	 * the jacobian or constrained jacobian of this task)
	 *
	 * @return Eigen::MatrixXd
	 */
	virtual Eigen::MatrixXd getTaskNullspace() const = 0;

	/**
	 * @brief Get the Nullspace projector of the previous tasks (N_prec
	 * associated with all the previous tasks)
	 *
	 * @return Eigen::MatrixXd
	 */
	virtual Eigen::MatrixXd getPreviousTasksNullspace() const = 0;

	/**
	 * @brief Get the Nullspace projector of this task and the previous tasks (N * N_prec)
	 *
	 * @return Eigen::MatrixXd
	 */
	virtual Eigen::MatrixXd getTaskAndPreviousNullspace() const = 0;

	/**
	 * @brief gets a const reference to the internal robot model
	 *
	 * @return const std::shared_ptr<Sai2Model::Sai2Model>
	 */
	const std::shared_ptr<Sai2Model::Sai2Model>& getConstRobotModel() const {
		return _robot;
	}

	/**
	 * @brief returns the loop timestep of the task
	 *
	 */
	const double& getLoopTimestep() const { return _loop_timestep; }

	/**
	 * @brief returns the task type
	 *
	 */
	const TaskType& getTaskType() const { return _task_type; }

	/**
	 * @brief returns the task name
	 *
	 */
	const std::string& getTaskName() const { return _task_name; }

private:
	std::shared_ptr<Sai2Model::Sai2Model> _robot;
	double _loop_timestep;

	TaskType _task_type;
	std::string _task_name;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_TEMPLATE_TASK_H_ */
#endif