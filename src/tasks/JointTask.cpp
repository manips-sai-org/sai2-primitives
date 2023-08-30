/*
 * JointTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "JointTask.h"

#include <stdexcept>

using namespace Eigen;
namespace Sai2Primitives {

JointTask::JointTask(std::shared_ptr<Sai2Model::Sai2Model> robot,
					 const double loop_timestep) {
	_loop_timestep = loop_timestep;
	_robot = robot;
	const int dof = _robot->dof();

	setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

	// default values for gains and velocity saturation
	_are_gains_isotropic = true;
	setGains(50.0, 14.0, 0.0);

	_use_velocity_saturation_flag = false;
	_saturation_velocity = VectorXd::Zero(dof);

	// initialize matrices sizes
	_N_prec = MatrixXd::Identity(dof, dof);
	_M_modified = MatrixXd::Identity(dof, dof);

	_use_internal_otg_flag = true;
	_otg = make_shared<OTG_joints>(robot->q(), loop_timestep);
	_otg->setMaxVelocity(M_PI / 3);
	_otg->setMaxAcceleration(M_PI);
	_otg->disableJerkLimits();

	reInitializeTask();
}

void JointTask::reInitializeTask() {
	int dof = _robot->dof();

	_current_position = _robot->q();
	_current_velocity.setZero(dof);

	_desired_position = _robot->q();
	_desired_velocity.setZero(dof);
	_desired_acceleration.setZero(dof);

	_integrated_position_error.setZero(dof);

	_otg->reInitialize(_current_position);
}

void JointTask::setGains(const VectorXd& kp, const VectorXd& kv,
						 const VectorXd& ki) {
	int dof = _robot->dof();

	if (kp.size() != dof || kv.size() != dof || ki.size() != dof) {
		throw std::invalid_argument(
			"size of gain vector inconsistent with number of robot joints in "
			"JointTask::useNonIsotropicGains\n");
	}
	if (kp.maxCoeff() < 0 || kv.maxCoeff() < 0 || ki.maxCoeff() < 0) {
		throw std::invalid_argument(
			"gains must be positive or zero in "
			"JointTask::useNonIsotropicGains\n");
	}
	if (kv.maxCoeff() < 1e-3 && _use_velocity_saturation_flag) {
		throw std::invalid_argument(
			"cannot set singular kv if using velocity saturation in "
			"JointTask::useNonIsotropicGains\n");
	}

	_are_gains_isotropic = false;
	_kp = kp;
	_kv = kv;
	_ki = ki;
}

void JointTask::setGains(const double kp, const double kv, const double ki) {
	if (kp < 0 || kv < 0 || ki < 0) {
		throw std::invalid_argument(
			"gains must be positive or zero in JointTask::useIsotropicGains\n");
	}
	if (kv < 1e-3 && _use_velocity_saturation_flag) {
		throw std::invalid_argument(
			"cannot set singular kv if using velocity saturation in "
			"JointTask::useIsotropicGains\n");
	}

	_are_gains_isotropic = true;
	int dof = _robot->dof();
	_kp = kp * MatrixXd::Identity(dof, dof);
	_kv = kv * MatrixXd::Identity(dof, dof);
	_ki = ki * MatrixXd::Identity(dof, dof);
}

void JointTask::updateTaskModel(const MatrixXd& N_prec) {
	if (N_prec.rows() != N_prec.cols()) {
		throw std::invalid_argument(
			"N_prec matrix not square in JointTask::updateTaskModel\n");
	}
	if (N_prec.rows() != _robot->dof()) {
		throw std::invalid_argument(
			"N_prec matrix size not consistent with robot dof in "
			"JointTask::updateTaskModel\n");
	}

	_N_prec = N_prec;

	switch (_dynamic_decoupling_type) {
		case FULL_DYNAMIC_DECOUPLING: {
			_M_modified = _robot->M();
			break;
		}

		case BOUNDED_INERTIA_ESTIMATES: {
			_M_modified = _robot->M();
			for (int i = 0; i < _robot->dof(); i++) {
				if (_M_modified(i, i) < 0.1) {
					_M_modified(i, i) = 0.1;
				}
			}
			break;
		}

		case IMPEDANCE: {
			_M_modified = MatrixXd::Identity(_robot->dof(), _robot->dof());
			break;
		}

		default: {
			_M_modified = _robot->M();
			break;
		}
	}
}

VectorXd JointTask::computeTorques() {
	int dof = _robot->dof();
	VectorXd task_joint_torques = VectorXd::Zero(dof);

	// update constroller state
	_current_position = _robot->q();
	_current_velocity = _robot->dq();

	VectorXd tmp_desired_position = _desired_position;
	VectorXd tmp_desired_velocity = _desired_velocity;
	VectorXd tmp_desired_acceleration = _desired_acceleration;

	// compute next state from trajectory generation
	if (_use_internal_otg_flag) {
		_otg->setGoalPositionAndVelocity(_desired_position, _desired_velocity);
		_otg->update();

		tmp_desired_position = _otg->getNextPosition();
		tmp_desired_velocity = _otg->getNextVelocity();
		tmp_desired_acceleration = _otg->getNextAcceleration();
	}

	// compute error for I term
	_integrated_position_error +=
		(_current_position - tmp_desired_position) * _loop_timestep;

	// compute task force (with velocity saturation if asked)
	if (_use_velocity_saturation_flag) {
		tmp_desired_velocity =
			-_kp * _kv.inverse() * (_current_position - tmp_desired_position) -
			_ki * _kv.inverse() * _integrated_position_error;
		for (int i = 0; i < _robot->dof(); i++) {
			if (tmp_desired_velocity(i) > _saturation_velocity(i)) {
				tmp_desired_velocity(i) = _saturation_velocity(i);
			} else if (tmp_desired_velocity(i) < -_saturation_velocity(i)) {
				tmp_desired_velocity(i) = -_saturation_velocity(i);
			}
		}
		task_joint_torques = -_kv * (_current_velocity - tmp_desired_velocity);
	} else {
		task_joint_torques = -_kp * (_current_position - tmp_desired_position) -
							 _kv * (_current_velocity - tmp_desired_velocity) -
							 _ki * _integrated_position_error;
	}

	task_joint_torques = _robot->M() * tmp_desired_acceleration +
						 _M_modified * task_joint_torques;

	// return projected task torques
	return _N_prec.transpose() * task_joint_torques;
}

void JointTask::enableInternalOtgAccelerationLimited(
	const VectorXd& max_velocity, const VectorXd& max_acceleration) {
	if (max_velocity.size() != _robot->dof() ||
		max_acceleration.size() != _robot->dof()) {
		throw std::invalid_argument(
			"max velocity or max acceleration vector size not consistent with "
			"robot dof in "
			"JointTask::enableInternalOtgAccelerationLimited\n");
	}
	_use_internal_otg_flag = true;
	_otg->reInitialize(_current_position);
	_otg->setMaxVelocity(max_velocity);
	_otg->setMaxAcceleration(max_acceleration);
	_otg->disableJerkLimits();
}

void JointTask::enableInternalOtgJerkLimited(const VectorXd& max_velocity,
											 const VectorXd& max_acceleration,
											 const VectorXd& max_jerk) {
	if (max_velocity.size() != _robot->dof() ||
		max_acceleration.size() != _robot->dof() ||
		max_jerk.size() != _robot->dof()) {
		throw std::invalid_argument(
			"max velocity, max acceleration or max jerk vector size not "
			"consistent with robot dof in "
			"JointTask::enableInternalOtgJerkLimited\n");
	}
	_use_internal_otg_flag = true;
	_otg->reInitialize(_current_position);
	_otg->setMaxVelocity(max_velocity);
	_otg->setMaxAcceleration(max_acceleration);
	_otg->setMaxJerk(max_jerk);
}

void JointTask::enableVelocitySaturation(const VectorXd& saturation_velocity) {
	if(saturation_velocity.size() != _robot->dof()) {
		throw std::invalid_argument(
			"saturation velocity vector size not consistent with robot dof in "
			"JointTask::enableVelocitySaturation\n");
	}
	_use_velocity_saturation_flag = true;
	_saturation_velocity = saturation_velocity;
}

} /* namespace Sai2Primitives */
