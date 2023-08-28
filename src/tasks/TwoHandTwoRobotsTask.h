/*
 * TwoHandTwoRobotsTask.h
 *
 *      This class creates a 6Dof position + orientation hybrid controller for a robotic manipulator using operational space formulation and an underlying PID compensator.
 *      If used for hybrid position force control, assumes a force sensor is attached to the same link as the control frame and the force sensed values are given in sensor frame.
 *      Besides, the force sensed and moment sensed are assumed to be the force and moment that the robot applies to the environment.
 *      It requires a robot model parsed from a urdf file to a Sai2Model object, as well as the definition of a control frame
 *      as a link at which the frame is attached, and an affine transform that determines the position and orientation of the control frame in this link
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_TWOHANDTWOROBOTS_TASK_H_
#define SAI2_PRIMITIVES_TWOHANDTWOROBOTS_TASK_H_

#include "Sai2Model.h"
#include <Eigen/Dense>
#include <string>
#include <chrono>

#ifdef USING_OTG
	#include "trajectory_generation/OTG.h"
	#include "trajectory_generation/OTG_ori.h"
#endif

namespace Sai2Primitives
{

class TwoHandTwoRobotsTask
{
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
	 * @param[in]  loop_time      time taken by a control loop. Used only in
	 *                            trajectory generation
	 */
	TwoHandTwoRobotsTask(Sai2Model::Sai2Model* robot_arm_1,
						 Sai2Model::Sai2Model* robot_arm_2, 
		            const std::string link_name_1, 
					const std::string link_name_2, 
		            const Eigen::Affine3d control_frame_1 = Eigen::Affine3d::Identity(),
		            const Eigen::Affine3d control_frame_2 = Eigen::Affine3d::Identity(),
		            const double loop_time = 0.001);

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
	 * @param[in]  loop_time    time taken by a control loop. Used only in
	 *                          trajectory generation
	 */
	TwoHandTwoRobotsTask(Sai2Model::Sai2Model* robot_arm_1,
						 Sai2Model::Sai2Model* robot_arm_2, 
		            const std::string link_name_1, 
		            const std::string link_name_2, 
		            const Eigen::Vector3d pos_in_link_1, 
		            const Eigen::Vector3d pos_in_link_2, 
		            const Eigen::Matrix3d rot_in_link_1 = Eigen::Matrix3d::Identity(),
		            const Eigen::Matrix3d rot_in_link_2 = Eigen::Matrix3d::Identity(),
		            const double loop_time = 0.001);


	//------------------------------------------------
	// Methods
	//------------------------------------------------

	// -------- core methods --------

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
	virtual void updateTaskModel(const Eigen::MatrixXd N_prec_1, const Eigen::MatrixXd N_prec_2);

	/**
	 * @brief Computes the torques associated with this task.
	 * @details Computes the torques taking into account the last model update and updated values for the robot joint positions/velocities
	 * assumes the desired orientation and angular velocity has been updated
	 * 
	 * @param task_joint_torques the vector to be filled with the new joint torques to apply for the task
	 */
	virtual void computeTorques(Eigen::VectorXd& task_joint_torques_1, Eigen::VectorXd& task_joint_torques_2);

	/**
	 * @brief      reinitializes the desired state to the current robot
	 *             configuration as well as the integrator terms. 
	 *             Also reinitialize the control frame to the center of the grasp points, 
	 *             and the inertial properties to zero, and the object controller to be on motion for all axes.
	 */
	void reInitializeTask();

	/**
	 * @brief      Sets the object mass properties, and compute the transformation between 
	 *             the frame of the grasp matrix and the object inertial frame
	 *
	 * @param[in]  object_mass     The object mass
	 * @param[in]  T_world_com     The position of the object inertial frame in world frame
	 * @param[in]  object_inertia  The object inertia tensor in its own inertial frame
	 */
	void setObjectMassPropertiesAndInitialInertialFrameLocation(double object_mass, 
			Eigen::Affine3d T_world_com,
			Eigen::Matrix3d object_inertia);

	// /**
	//  * @brief      Sets the control frame for the object. It will be placed on the object at the specified location and orientation
	//  * 			   when the function is called, and then it will follow the motion of the object assuming it is rigidly attached to the arms.
	//  * 			   Also sets the desired position and orientation to current ones.
	//  *
	//  * @param[in]  T_world_controlpoint  Location of the control frame in world frame.
	//  */
	void setControlFrameLocationInitial(Eigen::Affine3d T_world_controlpoint);

	// //////////////////////////// Object force control related functions /////////////////////////////////////

	void setForceSensorFrames(const std::string link_name_1, const Eigen::Affine3d sensor_in_link_r1, 
										const std::string link_name_2, const Eigen::Affine3d sensor_in_link_r2);

	void updateSensedForcesAndMoments(const Eigen::Vector3d sensed_force_sensor_frame_r1,
										const Eigen::Vector3d sensed_moment_sensor_frame_r1,
										const Eigen::Vector3d sensed_force_sensor_frame_r2,
										const Eigen::Vector3d sensed_moment_sensor_frame_r2);

	void setForceAxis(const Eigen::Vector3d force_axis);
	void updateForceAxis(const Eigen::Vector3d force_axis);

	void setLinearMotionAxis(const Eigen::Vector3d linear_motion_axis);
	void updateLinearMotionAxis(const Eigen::Vector3d linear_motion_axis);

	void setMomentAxis(const Eigen::Vector3d moment_axis);
	void updateMomentAxis(const Eigen::Vector3d moment_axis);

	void setAngularMotionAxis(const Eigen::Vector3d angular_motion_axis);
	void updateAngularMotionAxis(const Eigen::Vector3d angular_motion_axis);

	void setFullLinearMotionControl();
	void setFullForceControl();

	void setFullAngularMotionControl();
	void setFullMomentControl();

	void setClosedLoopForceControl();
	void setClosedLoopMomentControl();
	void setOpenLoopForceControl();
	void setOpenLoopMomentControl();

	void resetIntegratorsLinear();
	void resetIntegratorsAngular();
	void resetIntegrators();


	//------------------------------------------------
	// Attributes to be modified by user
	//------------------------------------------------
	Eigen::Vector3d _desired_object_position;      // world frame
	Eigen::Matrix3d _desired_object_orientation;   // world frame

	Eigen::Vector3d _desired_object_velocity;           // world frame
	Eigen::Vector3d _desired_object_angular_velocity;   // world frame

	Eigen::Vector3d _desired_object_acceleration;           // world frame
	Eigen::Vector3d _desired_object_angular_acceleration;   // world frame

	// gains for position control
	double _kp_pos, _kp_ori;
	double _kv_pos, _kv_ori;
	double _ki_pos, _ki_ori;

	Eigen::Vector3d _object_desired_force;        // world frame
	Eigen::Vector3d _object_desired_moment;       // world frame

	double _desired_internal_tension;
	Eigen::VectorXd _desired_internal_moments;    // dimension 5

	double _desired_internal_separation;
	Eigen::VectorXd _desired_internal_angles;
	double _internal_force_control_flag = false;  // if false, the relative behavior is controlled in motion and the orientation holds while the position can be controlled

	// gains for force control
	double _kp_force, _kp_moment;
	double _ki_force, _ki_moment;
	double _kv_force, _kv_moment;

	// gains for relative behavior control
	double _kp_internal_separation, _kp_internal_ori;
	double _kv_internal_separation, _kv_internal_ori;

	// velocity saturation
	bool _use_velocity_saturation_flag;     // false by default
	double _linear_saturation_velocity;     // defaults to 0.3
	double _angular_saturation_velocity;    // defaults to PI/3

	// trajectory interpolation
#ifdef USING_OTG
	bool _use_interpolation_pos_flag;   // defaults to true
	bool _use_interpolation_ori_flag;   // defaults to true

	// default values for the limits :
	// _otg_pos->setMaxVelocity(0.3);
	// _otg_pos->setMaxAcceleration(0.6);
	// _otg_pos->setMaxJerk(1.2);

	// _otg_ori->setMaxVelocity(M_PI/4);
	// _otg_ori->setMaxAcceleration(M_PI/2);
	// _otg_ori->setMaxJerk(M_PI);
#endif

	//------------------------------------------------
	// Attributes that the user should not touch
	//------------------------------------------------
	Sai2Model::Sai2Model* _robot_arm_1;
	Sai2Model::Sai2Model* _robot_arm_2;

	Eigen::MatrixXd _N_prec_1;
	Eigen::MatrixXd _N_prec_2;

	std::string _link_name_1;
	std::string _link_name_2;
	Eigen::Affine3d _control_frame_1;   // in link_frame
	Eigen::Affine3d _control_frame_2;   // in link_frame

	double _object_mass;
	Eigen::Matrix3d _object_inertia_in_control_frame;
	Eigen::Affine3d _T_com_controlpoint;

	// object control quantities
	Eigen::Vector3d _current_object_position;      // world frame
	Eigen::Matrix3d _current_object_orientation;   // world frame

	Eigen::Vector3d _current_object_velocity;           // world frame
	Eigen::Vector3d _current_object_angular_velocity;   // world frame

	Eigen::VectorXd _object_gravity;

	Eigen::Vector3d _object_orientation_error;               // world frame
	Eigen::Vector3d _integrated_object_orientation_error;    // world frame
	Eigen::Vector3d _integrated_object_position_error;       // world frame

	Eigen::Matrix3d _sigma_position;
	Eigen::Matrix3d _sigma_orientation;

	// object force control quantities
	Eigen::Affine3d _T_contact_fsensor_r1;
	Eigen::Affine3d _T_contact_fsensor_r2;

	Eigen::Vector3d _sensed_force_r1;          // world frame
	Eigen::Vector3d _sensed_force_r2;          // world frame
	Eigen::Vector3d _sensed_moment_r1;          // world frame
	Eigen::Vector3d _sensed_moment_r2;          // world frame

	Eigen::Vector3d _object_sensed_force;
	Eigen::Vector3d _object_sensed_moment;

	Eigen::Vector3d _integrated_object_force_error;    // world frame
	Eigen::Vector3d _integrated_object_moment_error;   // world frame

	bool _closed_loop_force_control;
	bool _closed_loop_moment_control;

	Eigen::Matrix3d _sigma_force;
	Eigen::Matrix3d _sigma_moment;

	// internal forces quantities
	double _sensed_internal_tension;
	Eigen::VectorXd _sensed_internal_moments;

	double _current_internal_separation;
	Eigen::VectorXd _current_internal_angles;

	Eigen::VectorXd _task_force;

	// model quantities
	std::vector<Eigen::Vector3d> _contact_locations;  // the contact points in world frame
	std::vector<Sai2Model::ContactType> _contact_types;  // only rigid contacts supported for now
	Eigen::MatrixXd _grasp_matrix;
	Eigen::MatrixXd _grasp_matrix_inverse;
	Eigen::Matrix3d _R_grasp_matrix;

	// previous robots positions and orientations
	Eigen::Vector3d _previous_position_r1;
	Eigen::Vector3d _previous_position_r2;
	Eigen::Matrix3d _previous_orientation_r1;
	Eigen::Matrix3d _previous_orientation_r2;
	Eigen::Vector3d _previous_R12_axis;

	Eigen::Vector3d _arbitrary_direction_hand1, _arbitrary_direction_hand2;
	Eigen::Vector3d _x_object_frame, _y_object_frame, _z_object_frame;
	Eigen::Affine3d _T_world_object;
	Eigen::Affine3d _T_hand1_object;
	Eigen::Affine3d _T_hand2_object;
	Eigen::MatrixXd _object_effective_inertia;
	Eigen::MatrixXd _robot_1_effective_inertia;
	Eigen::MatrixXd _robot_2_effective_inertia;
	Eigen::MatrixXd _Lambda_tot;

	Eigen::MatrixXd _J_res_int;
	Eigen::MatrixXd _Lambda_complete;

	Eigen::MatrixXd _jacobian_1;
	Eigen::MatrixXd _projected_jacobian_1;
	Eigen::MatrixXd _Lambda_1;
	Eigen::MatrixXd _Jbar_1;
	Eigen::MatrixXd _N_1;
	Eigen::MatrixXd _jacobian_2;
	Eigen::MatrixXd _projected_jacobian_2;
	Eigen::MatrixXd _Lambda_2;
	Eigen::MatrixXd _Jbar_2;
	Eigen::MatrixXd _N_2;

	// Eigen::MatrixXd _Jr, _Ji;
	// Eigen::MatrixXd _Lambda_r, _Lambda_i;

	Eigen::VectorXd _step_desired_object_position;
	Eigen::VectorXd _step_desired_object_velocity;
	Eigen::VectorXd _step_desired_object_acceleration;
	Eigen::Matrix3d _step_desired_object_orientation;
	Eigen::Vector3d _step_object_orientation_error;
	Eigen::Vector3d _step_desired_object_angular_velocity;
	Eigen::Vector3d _step_desired_object_angular_acceleration;

#ifdef USING_OTG
	double _loop_time;
	OTG* _otg_pos;
	OTG_ori* _otg_ori;
#endif

	std::chrono::high_resolution_clock::time_point _t_prev;
	std::chrono::high_resolution_clock::time_point _t_curr;
	std::chrono::duration<double> _t_diff;
	bool _first_iteration;

};

MatrixXd computeGInverseAtGeometricCenterExplicit(const Matrix3d Rg, 
		const Vector3d contact_location_1,
		const Vector3d contact_location_2);


} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_TWOHANDTWOROBOTS_TASK_H_ */
#endif