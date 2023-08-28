/*
 * OrientationTask.h
 *
 *      This class creates an orientation controller for a robotic manipulator using operational space formulation and an underlying PID compensator.
 *      It requires a robot model parsed from a urdf file to a Sai2Model object, as well as the definition of a control frame
 *      as a link at which the frame is attached, and an affine transform that determines the position and orientation of the control frame in this link
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_ORIENTATION_TASK_H_
#define SAI2_PRIMITIVES_ORIENTATION_TASK_H_

#include "Sai2Model.h"
#include <Eigen/Dense>
#include <string>
#include <chrono>

#ifdef USING_OTG
	#include "trajectory_generation/OTG_ori.h"
#endif

namespace Sai2Primitives
{

class OrientationTask : public TemplateTask
{
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
	OrientationTask(Sai2Model::Sai2Model* robot, 
		            const std::string link_name, 
		            const Eigen::Affine3d control_frame = Eigen::Affine3d::Identity(),
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
	 * @param[in]  loop_time    time taken by a control loop. Used only in
	 *                          trajectory generation
	 */
	OrientationTask(Sai2Model::Sai2Model* robot, 
		            const std::string link_name, 
		            const Eigen::Vector3d pos_in_link = Eigen::Vector3d::Zero(), 
		            const Eigen::Matrix3d rot_in_link = Eigen::Matrix3d::Identity(),
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
	virtual void updateTaskModel(const Eigen::MatrixXd N_prec);

	/**
	 * @brief Computes the torques associated with this task.
	 * @details Computes the torques taking into account the last model update and updated values for the robot joint positions/velocities
	 * assumes the desired orientation and angular velocity has been updated
	 * 
	 * @param task_joint_torques the vector to be filled with the new joint torques to apply for the task
	 */
	virtual void computeTorques(Eigen::VectorXd& task_joint_torques);

	/**
	 * @brief      reinitializes the desired state to the current robot
	 *             configuration as well as the integrator terms
	 */
	void reInitializeTask();

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

	// inputs to be defined by the user
	Eigen::Matrix3d _desired_orientation;       // defaults to the current orientation when the task is created
	Eigen::Vector3d _desired_angular_velocity;  // defaults to Zero
	Eigen::Vector3d _desired_angular_acceleration;  // defaults to Zero

	double _kp;    // defaults to 50.0 
	double _kv;    // defaults to 14.0
	double _ki;    // defaults to 0.0

	bool _use_velocity_saturation_flag;  // defaults to false
	double _saturation_velocity;         // defaults to PI/3 Rad/s (60 degrees per second)

// trajectory generation via interpolation using Reflexxes Library
#ifdef USING_OTG
	bool _use_interpolation_flag;   // defaults to true

	// default limits for trajectory generation (same in all directions) :
	// Velocity      - PI/3  Rad/s
	// Acceleration  - PI    Rad/s^2
	// Jerk          - 3PI   Rad/s^3
#endif

	// internal variables, not to be touched by the user
	std::string _link_name;
	Eigen::Affine3d _control_frame;

	Eigen::Matrix3d _current_orientation;
	Eigen::Vector3d _current_angular_velocity;

	Eigen::Vector3d _orientation_error;
	Eigen::Vector3d _integrated_orientation_error;

	Eigen::MatrixXd _jacobian;
	Eigen::MatrixXd _projected_jacobian;
	Eigen::MatrixXd _Lambda;
	Eigen::MatrixXd _Jbar;
	Eigen::MatrixXd _N;

	Eigen::Matrix3d _step_desired_orientation;
	Eigen::Vector3d _step_orientation_error;
	Eigen::Vector3d _step_desired_angular_velocity;
	Eigen::Vector3d _step_desired_angular_acceleration;

#ifdef USING_OTG
	double _loop_time;
	OTG_ori* _otg;       
#endif

};


} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_ORIENTATION_TASK_H_ */
#endif