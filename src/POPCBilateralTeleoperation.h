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
#include <queue>

#include "HapticDeviceController.h"
#include "tasks/MotionForceTask.h"

namespace Sai2Primitives {

class POPCBilateralTeleoperation {
public:
	/**
	 * @brief Construct a new POPCBilateralTeleoperation object
	 *
	 * @param motion_force_task the task used to control the robot
	 * @param haptic_controller the controller used to control the haptic device
	 * @param loop_dt the control loop time step
	 */
	POPCBilateralTeleoperation(
		const std::shared_ptr<MotionForceTask>& motion_force_task,
		const std::shared_ptr<HapticDeviceController>& haptic_controller,
		const double loop_dt);

	// disallow default, copy operator and copy constructor
	POPCBilateralTeleoperation() = delete;
	POPCBilateralTeleoperation(const POPCBilateralTeleoperation&) = delete;
	POPCBilateralTeleoperation& operator=(const POPCBilateralTeleoperation&) =
		delete;

	/**
	 * @brief Reinitialize the passivity observer. called automatically when the
	 * haptic controller type changes to motion-motion.
	 *
	 */
	void reInitialize();

	/**
	 * @brief Compute the additional damping force and moment to be applied to
	 * the haptic device. If the haptic controller type is not motion-motion,
	 * this will return zero. Otherwise, the damping value will depend on the
	 * passivity observer value computed internally. If orientation
	 * teleoperation is disabled in the haptic controller, then the returned
	 * damping moment will be zero
	 *
	 * @return std::pair<Eigen::Vector3d, Eigen::Vector3d> the additional
	 * damping force and moment. The first element is the force and the second
	 * is the moment. Those values need to be added to the commanded force and
	 * moment sent to the haptic device.
	 */
	std::pair<Eigen::Vector3d, Eigen::Vector3d>
	computeAdditionalHapticDampingForce();

private:
	/**
	 * @brief Computes the passivity observer and controller for the linear part
	 * of the teleoperation and returns the damping force
	 *
	 * @return Eigen::Vector3d the damping force
	 */
	Eigen::Vector3d computePOPCForce();

	/**
	 * @brief Computes the passivity observer and controller for the angular
	 * part of the teleoperation and returns the damping moment
	 *
	 * @return Eigen::Vector3d the damping moment
	 */
	Eigen::Vector3d computePOPCTorque();

	// internal pointers to the motion force task and haptic device controller
	std::shared_ptr<MotionForceTask> _motion_force_task;
	std::shared_ptr<HapticDeviceController> _haptic_controller;

	// passivity observer variables
	double _passivity_observer_force;
	std::queue<double> _PO_buffer_force;
	double _passivity_observer_moment;
	std::queue<double> _PO_buffer_moment;

	// maximum damping values
	double _max_damping_force;
	double _max_damping_moment;

	// control loop time step
	double _loop_dt;

	// latest haptic controller type to know when a switch occured
	HapticControlType _latest_haptic_controller_type;
};

} /* namespace Sai2Primitives */

/* SAI2_PRIMITIVES_POPC_BILATERAL_TELEOPERATION_H_ */
#endif
