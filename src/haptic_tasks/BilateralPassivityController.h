/*
 * BilateralPassivityController.h
 *
 *      Implements time domain passivity approach for a bilateral teleoperation scheme 
 *      where the robot is controlled using a MotionForceTask from Sai2 and the haptic device
 *      is controlled using a HapticController from Sai2.
 *      
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_BILATERALPASSIVITYCONTROLLER_TASK_H_
#define SAI2_PRIMITIVES_BILATERALPASSIVITYCONTROLLER_TASK_H_


#include "tasks/MotionForceTask.h"
#include "HapticController.h"
#include <Eigen/Dense>
#include <string>
#include <chrono>
#include <queue> 

#define SAI2PRIMITIVES_BILATERAL_PASSIVITY_CONTROLLER_WINDOWED_PO_BUFFER  30

namespace Sai2Primitives
{

class BilateralPassivityController
{
public:

	BilateralPassivityController(MotionForceTask* posori_task, HapticController* haptic_task);

	~BilateralPassivityController();

	void reInitializeTask();

	void computePOPCForce(Eigen::Vector3d& haptic_damping_force_command);

	void computePOPCTorque(Eigen::Vector3d& haptic_damping_moment_command);


	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

	MotionForceTask* _posori_task;
	HapticController* _haptic_task;

	double _passivity_observer_force;
	double _stored_energy_force;
	std::queue<double> _PO_buffer_force;
	const int _PO_buffer_size_force = 30;

	double _passivity_observer_moment;
	double _stored_energy_moment;
	std::queue<double> _PO_buffer_moment;
	const int _PO_buffer_size_moment = 30;

	double _alpha_force;
	double _max_alpha_force;
	Eigen::Vector3d _damping_force;
	double _alpha_moment;
	double _max_alpha_moment;
	Eigen::Vector3d _damping_moment;

	std::chrono::high_resolution_clock::time_point _t_prev_force;
	std::chrono::high_resolution_clock::time_point _t_prev_moment;
	std::chrono::duration<double> _t_diff_force;
	std::chrono::duration<double> _t_diff_moment;
	bool _first_iteration_force;
	bool _first_iteration_moment;

};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_BILATERALPASSIVITYCONTROLLER_TASK_H_ */
#endif

