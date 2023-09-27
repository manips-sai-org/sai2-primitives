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

class TemplateTask {
public:
	TemplateTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
				 const double loop_timestep)
		: _robot(robot), _loop_timestep(loop_timestep) {}

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
	 * @brief Re initializes the task by setting the desired state to the current state.
	 * 
	 */
	virtual void reInitializeTask() = 0;

    /**
     * @brief Get the Nullspace projector of this task
     * 
     * @return Eigen::MatrixXd
     */
    virtual Eigen::MatrixXd getTaskNullspace() const = 0;

    /**
     * @brief Get the Nullspace projector of this task and the previous tasks
     * 
     * @return Eigen::MatrixXd
     */
	virtual Eigen::MatrixXd getTaskAndPreviousNullspace() const =0;


	/**
	 * @brief gets a const reference to the internal robot model
	 *
	 * @return const std::shared_ptr<Sai2Model::Sai2Model>
	 */
	const std::shared_ptr<Sai2Model::Sai2Model> getConstRobotModel() const {
		return _robot;
	}

	/**
	 * @brief returns the loop timestep of the task
	 *
	 */
	double getLoopTimestep() const { return _loop_timestep; }

private:
	std::shared_ptr<Sai2Model::Sai2Model> _robot;
	double _loop_timestep;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_TEMPLATE_TASK_H_ */
#endif