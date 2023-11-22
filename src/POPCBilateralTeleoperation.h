/*
 * POPCBilateralTeleoperation.h
 *
 *      Implements time domain passivity approach for a bilateral teleoperation
 * scheme where the robot is controlled using a MotionForceTask from Sai2 and
 * the haptic device is controlled using a HapticDeviceController from Sai2.
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_POPC_BILATERAL_TELEOPERATION_H_
#define SAI2_PRIMITIVES_POPC_BILATERAL_TELEOPERATION_H_

#include <Eigen/Dense>
#include <string>

#include "HapticDeviceController.h"
#include "tasks/MotionForceTask.h"
// #include <chrono>
#include <queue>

#define SAI2PRIMITIVES_BILATERAL_PASSIVITY_CONTROLLER_WINDOWED_PO_BUFFER 30

namespace Sai2Primitives {

class POPCBilateralTeleoperation {
public:
	POPCBilateralTeleoperation(
		const std::shared_ptr<MotionForceTask>& motion_force_task,
		const std::shared_ptr<HapticDeviceController>& haptic_task,
		const double loop_dt);

	~POPCBilateralTeleoperation() = default;

	// disallow default, copy operator and copy constructor
	POPCBilateralTeleoperation() = delete;
	POPCBilateralTeleoperation(const POPCBilateralTeleoperation&) = delete;
	POPCBilateralTeleoperation& operator=(const POPCBilateralTeleoperation&) = delete;

	void reInitialize();

	std::pair<Eigen::Vector3d, Eigen::Vector3d> computeAdditionalHapticDampingForce();


private:
	Eigen::Vector3d computePOPCForce();

	Eigen::Vector3d computePOPCTorque();

	//-----------------------------------------------
	//         Member variables
	//-----------------------------------------------

	std::shared_ptr<MotionForceTask> _motion_force_task;
	std::shared_ptr<HapticDeviceController> _haptic_controller;

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

	double _loop_dt;

	// std::chrono::high_resolution_clock::time_point _t_prev_force;
	// std::chrono::high_resolution_clock::time_point _t_prev_moment;
	// std::chrono::duration<double> _t_diff_force;
	// std::chrono::duration<double> _t_diff_moment;
	// bool _first_iteration_force;
	// bool _first_iteration_moment;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_POPC_BILATERAL_TELEOPERATION_H_ */
#endif
