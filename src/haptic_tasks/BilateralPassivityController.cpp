/*
 * BilateralPassivityController.cpp
 *
 *      Author: Mikael Jorda
 */

#include "BilateralPassivityController.h"

namespace Sai2Primitives
{

BilateralPassivityController::BilateralPassivityController(MotionForceTask* posori_task, HapticController* haptic_task)
{
	_posori_task = posori_task;
	_haptic_task = haptic_task;

	_max_alpha_force = 30.0;
	_max_alpha_moment = 0.04;

	reInitializeTask();
}

void BilateralPassivityController::reInitializeTask()
{

	_passivity_observer_force = 0;
	_stored_energy_force = 0;
	_PO_buffer_force = std::queue<double>();

	_passivity_observer_moment = 0;
	_stored_energy_moment = 0;
	_PO_buffer_moment = std::queue<double>();

	_alpha_force = 0;
	_damping_force.setZero();
	_alpha_moment = 0;
	_damping_moment.setZero();

	_first_iteration_force = true;
	_first_iteration_moment = true;
}

void BilateralPassivityController::computePOPCForce(Eigen::Vector3d& haptic_damping_force_command)
{
	// compute dt for passivity observer
	std::chrono::high_resolution_clock::time_point t_curr = std::chrono::high_resolution_clock::now();
	if(_first_iteration_force)
	{
		_t_prev_force = t_curr;
		_first_iteration_force = false;
	}
	std::chrono::duration<double> t_diff = t_curr - _t_prev_force;
	double dt = t_diff.count();

	// compute stored energy
	Eigen::Vector3d dx = _posori_task->_sigma_position * (_posori_task->_desired_position - _posori_task->_current_position);
	_stored_energy_force = 0.5 * _posori_task->_kp_pos * dx.squaredNorm(); 
	_stored_energy_force = 0;

	// compute power input and output
	// upit mass force already contains the projection into the motion space
	double power_output_robot_side = _posori_task->_current_velocity.dot(_posori_task->_unit_mass_force.head(3));

	Eigen::Vector3d device_force_in_motion_space = _haptic_task->_sigma_position * _haptic_task->_commanded_force_device;
	double power_output_haptic_side = _haptic_task->_current_trans_velocity_device.dot(device_force_in_motion_space) -
		_posori_task->_kp_pos * _haptic_task->_current_trans_velocity_device_RobFrame.dot(dx);

	double total_power_input = (- power_output_haptic_side - power_output_robot_side) * dt;

	// compute passivity observer
	_PO_buffer_force.push(total_power_input);
	_passivity_observer_force += total_power_input;

	// compute the passivity controller
	if(_passivity_observer_force + _stored_energy_force < 0.0 && !_first_iteration_force)
	{
		double vh_norm_square = _haptic_task->_current_trans_velocity_device.squaredNorm();
		
		// if velocity of haptic device is too low, we cannot dissipate through gamping
		if(vh_norm_square < 1e-6)
		{
			vh_norm_square = 1e-6;
		}

		_alpha_force = -(_passivity_observer_force + _stored_energy_force)/(vh_norm_square * dt);

		// limit the amount of damping otherwise the real hardware has issues
		if(_alpha_force > _max_alpha_force)
		{
			_alpha_force = _max_alpha_force;
		}

		Eigen::Matrix3d sigma_damping = Eigen::Matrix3d::Identity();

		// Eigen::Vector3d force_direction = _haptic_task->_commanded_force_device;
		// if(force_direction.norm() > 0.01)
		// {
		// 	sigma_damping = force_direction * force_direction.transpose() / force_direction.squaredNorm();
		// }


		_damping_force = - _alpha_force * sigma_damping * _haptic_task->_current_trans_velocity_device;
		_damping_force = _haptic_task->_sigma_position * _damping_force;

		double passivity_observer_correction = dt * _haptic_task->_current_trans_velocity_device.dot(_damping_force);
		_passivity_observer_force -= passivity_observer_correction;
		_PO_buffer_force.back() -= passivity_observer_correction;
	}
	else
	{
		// no passivity controller correction
		_alpha_force = 0;
		_damping_force.setZero();

		while(_PO_buffer_force.size() > _PO_buffer_size_force)
		{
			// do not reset if it would make your system think it is going to be active
			if(_passivity_observer_force > _PO_buffer_force.front())
			{
				if(_PO_buffer_force.front() > 0) // only forget dissipated energy
				{
					_passivity_observer_force -= _PO_buffer_force.front();
				}
				_PO_buffer_force.pop();
			}
			else
			{
				break;
			}
		}
	}


	haptic_damping_force_command = _damping_force;

	_t_prev_force = t_curr;

}
	
void BilateralPassivityController::computePOPCTorque(Eigen::Vector3d& haptic_damping_moment_command)
{
	// compute dt for passivity observer
	std::chrono::high_resolution_clock::time_point t_curr = std::chrono::high_resolution_clock::now();
	if(_first_iteration_moment)
	{
		_t_prev_moment = t_curr;
		_first_iteration_moment = false;
	}
	std::chrono::duration<double> t_diff = t_curr - _t_prev_moment;
	double dt = t_diff.count();

	// compute stored energy
	Eigen::Vector3d ori_error = - _posori_task->_sigma_orientation * _posori_task->_orientation_error;
	_stored_energy_moment = 0.5 * _posori_task->_kp_ori * ori_error.squaredNorm(); 
	// _stored_energy_moment = 0;

	// compute power input and output
	// upit mass force already contains the projection into the motion space
	double power_output_robot_side = _posori_task->_current_angular_velocity.dot(_posori_task->_unit_mass_force.tail(3));

	Eigen::Vector3d device_moment_in_motion_space = _haptic_task->_sigma_orientation * _haptic_task->_commanded_torque_device;
	double power_output_haptic_side = _haptic_task->_current_rot_velocity_device.dot(device_moment_in_motion_space) -
		_posori_task->_kp_ori * _haptic_task->_current_rot_velocity_device_RobFrame.dot(ori_error);

	double total_power_input = (- power_output_haptic_side - power_output_robot_side) * dt;

	// compute passivity observer
	_PO_buffer_moment.push(total_power_input);
	_passivity_observer_moment += total_power_input;

	// compute the passivity controller
	if(_passivity_observer_moment + _stored_energy_moment < 0.0 && !_first_iteration_moment)
	{
		double vh_norm_square = _haptic_task->_current_rot_velocity_device.squaredNorm();
		
		// if velocity of haptic device is too low, we cannot dissipate through gamping
		if(vh_norm_square < 1e-6)
		{
			vh_norm_square = 1e-6;
		}

		_alpha_moment = -(_passivity_observer_moment + _stored_energy_moment)/(vh_norm_square * dt);

		// limit the amount of damping otherwise the real hardware has issues
		if(_alpha_moment > _max_alpha_moment)
		{
			_alpha_moment = _max_alpha_moment;
		}

		_damping_moment = - _alpha_moment * _haptic_task->_current_rot_velocity_device;

		double passivity_observer_correction = dt * _haptic_task->_current_rot_velocity_device.dot(_damping_moment);
		_passivity_observer_moment -= passivity_observer_correction;
		_PO_buffer_moment.back() -= passivity_observer_correction;
	}
	else
	{
		// no passivity controller correction
		_alpha_moment = 0;
		_damping_moment.setZero();

		while(_PO_buffer_moment.size() > _PO_buffer_size_moment)
		{
			// do not reset if it would make your system think it is going to be active
			if(_passivity_observer_moment > _PO_buffer_moment.front())
			{
				if(_PO_buffer_moment.front() > 0) // only forget dissipated energy
				{
					_passivity_observer_moment -= _PO_buffer_moment.front();
				}
				_PO_buffer_moment.pop();
			}
			else
			{
				break;
			}
		}
	}

	haptic_damping_moment_command = _damping_moment;

	_t_prev_moment = t_curr;
}


} /* namespace Sai2Primitives */