/*
 * TemplateTask.h
 *
 *      Template task for Sai2 tasks
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_TEMPLATE_TASK_H_
#define SAI2_PRIMITIVES_TEMPLATE_TASK_H_

#include "Sai2Model.h"
#include <Eigen/Dense>
#include <string>
#include <chrono>

namespace Sai2Primitives
{

class TemplateTask
{
public:

	/**
	 * @brief update the task model (only _N_prec for a joint task)
	 * 
	 * @param N_prec The nullspace matrix of all the higher priority tasks. If this is the highest priority task, use identity of size n*n where n in the number of DoF of the robot.
	 */
	virtual void updateTaskModel(const Eigen::MatrixXd N_prec) = 0;

	/**
	 * @brief Computes the torques associated with this task.
	 * @details Computes the torques taking into account the last model update and updated values for the robot joint positions/velocities
	 * assumes the desired position and velocity has been updated
	 * 
	 * @param task_joint_torques the vector to be filled with the new joint torques to apply for the task
	 */
	virtual void computeTorques(Eigen::VectorXd& task_joint_torques) = 0;


	Sai2Model::Sai2Model* _robot;
	double _loop_timestep;

	Eigen::VectorXd _task_force;

	Eigen::MatrixXd _N_prec;
};


} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_TEMPLATE_TASK_H_ */
#endif