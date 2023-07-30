/*
 * JointTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "JointTask.h"

#include <stdexcept>


namespace Sai2Primitives
{


JointTask::JointTask(Sai2Model::Sai2Model* robot,
			const double loop_timestep)
{
	_loop_timestep = loop_timestep;
	_robot = robot;
	int dof = _robot->dof();

	_current_position = _robot->q();
	_current_velocity.setZero(dof);

	// default values for gains and velocity saturation
	_kp = 50.0;
	_kv = 14.0;
	_ki = 0.0;
	_use_velocity_saturation_flag = false;
	_saturation_velocity = M_PI/3.0*Eigen::VectorXd::Ones(dof);

	_use_isotropic_gains = true;
	_kp_mat = Eigen::MatrixXd::Zero(dof,dof);
	_kv_mat = Eigen::MatrixXd::Zero(dof,dof);
	_ki_mat = Eigen::MatrixXd::Zero(dof,dof);

	// initialize matrices sizes
	_N_prec = Eigen::MatrixXd::Identity(dof,dof);
	_M_modified = Eigen::MatrixXd::Zero(dof,dof);

#ifdef USING_OTG 
	_use_interpolation_flag = true;
	_loop_time = loop_time;
	_otg = new OTG(_current_position, _loop_time);

	_otg->setMaxVelocity(M_PI/3);
	_otg->setMaxAcceleration(M_PI);
	_otg->setMaxJerk(3*M_PI);
#endif
	reInitializeTask();
}

void JointTask::reInitializeTask()
{
	int dof = _robot->dof();

	_desired_position = _robot->q();
	_desired_velocity.setZero(dof);
	_desired_acceleration.setZero(dof);

	_step_desired_position = _desired_position;
	_step_desired_velocity = _desired_velocity;
	_step_desired_acceleration = _desired_acceleration;

	_task_force.setZero();
	_integrated_position_error.setZero(dof);

#ifdef USING_OTG 
	_otg->reInitialize(_current_position);
#endif
}

void JointTask::setDynamicDecouplingFull()
{
	_dynamic_decoupling_type = FULL_DYNAMIC_DECOUPLING;
}

void JointTask::setDynamicDecouplingBIE()
{
	_dynamic_decoupling_type = BOUNDED_INERTIA_ESTIMATES;
}

void JointTask::setDynamicDecouplingNone()
{
	_dynamic_decoupling_type = IMPEDANCE;
}

void JointTask::setNonIsotropicGains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv, const Eigen::VectorXd& ki)
{
	int dof = _robot->dof();

	if(kp.size() != dof || kv.size() != dof || ki.size() != dof)
	{
		throw std::invalid_argument("size of gain vector inconsistent with number of robot joints in JointTask::useNonIsotropicGains\n");
	}

	_use_isotropic_gains = false;

	_kp_mat.setZero(dof,dof);
	_kv_mat.setZero(dof,dof);
	_ki_mat.setZero(dof,dof);

	for(int i=0 ; i<dof ; i++)
	{
		_kp_mat(i,i) = kp(i);
		_kv_mat(i,i) = kv(i);
		_ki_mat(i,i) = ki(i);
	}
}
	
void JointTask::setIsotropicGains(const double kp, const double kv, const double ki)
{
	_use_isotropic_gains = true;

	_kp = kp;
	_kv = kv;
	_ki = ki;
}


void JointTask::updateTaskModel(const Eigen::MatrixXd N_prec)
{
	if(N_prec.rows() != N_prec.cols())
	{
		throw std::invalid_argument("N_prec matrix not square in JointTask::updateTaskModel\n");
	}
	if(N_prec.rows() != _robot->dof())
	{
		throw std::invalid_argument("N_prec matrix size not consistent with robot dof in JointTask::updateTaskModel\n");
	}

	_N_prec = N_prec;

	switch(_dynamic_decoupling_type)
	{
		case FULL_DYNAMIC_DECOUPLING :
		{
			_M_modified = _robot->M();
			break;
		}

		case BOUNDED_INERTIA_ESTIMATES :
		{
			_M_modified = _robot->M();
			for(int i=0 ; i<_robot->dof() ; i++)
			{
				if(_M_modified(i,i) < 0.1)
				{
					_M_modified(i,i) = 0.1;
				}
			}
			break;
		}

		case IMPEDANCE :
		{
			_M_modified = Eigen::MatrixXd::Identity(_robot->dof(), _robot->dof());
			break;
		}

		default :
		{
			_M_modified = _robot->M();
			break;		
		}
	}
}


Eigen::VectorXd JointTask::computeTorques()
{
	int dof = _robot->dof();
	VectorXd task_joint_torques = VectorXd::Zero(dof);

	// update matrix gains
	if(_use_isotropic_gains)
	{
		_kp_mat = _kp * Eigen::MatrixXd::Identity(dof,dof);
		_kv_mat = _kv * Eigen::MatrixXd::Identity(dof,dof);
		_ki_mat = _ki * Eigen::MatrixXd::Identity(dof,dof);
	}

	// update constroller state
	_current_position = _robot->q();
	_current_velocity = _robot->dq();
	_step_desired_position = _desired_position;
	_step_desired_velocity = _desired_velocity;
	_step_desired_acceleration = _desired_acceleration;

	// compute next state from trajectory generation
#ifdef USING_OTG
	if(_use_interpolation_flag)
	{
		_otg->setGoalPositionAndVelocity(_desired_position, _desired_velocity);
		_otg->computeNextState(_step_desired_position, _step_desired_velocity, _step_desired_acceleration);
	}
#endif

	// compute error for I term
	_integrated_position_error += (_current_position - _step_desired_position) * _loop_timestep;

	// compute task force (with velocity saturation if asked)
	if(_use_velocity_saturation_flag)
	{
		_step_desired_velocity = -_kp_mat*_kv_mat.inverse() * (_current_position - _step_desired_position) - _ki_mat*_kv_mat.inverse() * _integrated_position_error;
		for(int i=0; i<_robot->dof(); i++)
		{
			if(_step_desired_velocity(i) > _saturation_velocity(i))
			{
				_step_desired_velocity(i) = _saturation_velocity(i);
			}
			else if(_step_desired_velocity(i) < -_saturation_velocity(i))
			{
				_step_desired_velocity(i) = -_saturation_velocity(i);
			}
		}
		_task_force = (_step_desired_acceleration -_kv_mat*(_current_velocity - _step_desired_velocity));
	}
	else
	{
		_task_force = (_step_desired_acceleration -_kp_mat*(_current_position - _step_desired_position) - _kv_mat * (_current_velocity - _step_desired_velocity) - _ki_mat * _integrated_position_error);
	}

	_task_force = _M_modified * _task_force;

	// compute task torques
	task_joint_torques = _N_prec.transpose() * _task_force;
	return task_joint_torques;
}

} /* namespace Sai2Primitives */

