/*
 * JointTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "JointTask.h"

#include <stdexcept>

using namespace Eigen;
namespace Sai2Primitives {

JointTask::JointTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
					 const std::string& task_name, const double loop_timestep)
	: TemplateTask(robot, task_name, TaskType::JOINT_TASK, loop_timestep) {
	// selection for full joint task
	_joint_selection = MatrixXd::Identity(getConstRobotModel()->dof(), getConstRobotModel()->dof());
	_is_partial_joint_task = false;

	initialSetup();
}

JointTask::JointTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
					 const MatrixXd& joint_selection_matrix,
					 const std::string& task_name,
					 const double loop_timestep)
	: TemplateTask(robot, task_name, TaskType::JOINT_TASK, loop_timestep) {
	// selection for partial joint task
	if (joint_selection_matrix.cols() != getConstRobotModel()->dof()) {
		throw std::invalid_argument(
			"joint selection matrix size not consistent with robot dof in "
			"JointTask constructor\n");
	}
	// find rank of joint selection matrix
	FullPivLU<MatrixXd> lu(joint_selection_matrix);
	if (lu.rank() != joint_selection_matrix.rows()) {
		throw std::invalid_argument(
			"joint selection matrix is not full rank in JointTask "
			"constructor\n");
	}
	_joint_selection = joint_selection_matrix;
	_is_partial_joint_task = true;

	initialSetup();
}

void JointTask::initialSetup() {
	const int robot_dof = getConstRobotModel()->dof();
	_task_dof = _joint_selection.rows();
	setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

	// default values for gains and velocity saturation
	_are_gains_isotropic = true;
	setGains(50.0, 14.0, 0.0);

	_use_velocity_saturation_flag = false;
	_saturation_velocity = VectorXd::Zero(_task_dof);

	// initialize matrices sizes
	_N_prec = MatrixXd::Identity(robot_dof, robot_dof);
	_M_partial = MatrixXd::Identity(_task_dof, _task_dof);
	_M_partial_modified = MatrixXd::Identity(_task_dof, _task_dof);
	_projected_jacobian = _joint_selection;
	_N = MatrixXd::Zero(robot_dof, robot_dof);
	_current_task_range = MatrixXd::Identity(_task_dof, _task_dof);

	_use_internal_otg_flag = true;
	_otg =
		make_shared<OTG_joints>(_joint_selection * getConstRobotModel()->q(), getLoopTimestep());
	_otg->setMaxVelocity(M_PI / 3);
	_otg->setMaxAcceleration(M_PI);
	_otg->disableJerkLimits();

	reInitializeTask();
}

void JointTask::reInitializeTask() {
	const int robot_dof = getConstRobotModel()->dof();

	_current_position = _joint_selection * getConstRobotModel()->q();
	_current_velocity.setZero(_task_dof);

	_desired_position = _current_position;
	_desired_velocity.setZero(_task_dof);
	_desired_acceleration.setZero(_task_dof);

	_integrated_position_error.setZero(_task_dof);

	_otg->reInitialize(_current_position);
}

void JointTask::setDesiredPosition(const VectorXd& desired_position) {
	if (desired_position.size() != _task_dof) {
		throw std::invalid_argument(
			"desired position vector size not consistent with task dof in "
			"JointTask::setDesiredPosition\n");
	}
	_desired_position = desired_position;
}

void JointTask::setDesiredvelocity(const VectorXd& desired_velocity) {
	if (desired_velocity.size() != _task_dof) {
		throw std::invalid_argument(
			"desired velocity vector size not consistent with task dof in "
			"JointTask::setDesiredvelocity\n");
	}
	_desired_velocity = desired_velocity;
}

void JointTask::setDesiredAcceleration(const VectorXd& desired_acceleration) {
	if (desired_acceleration.size() != _task_dof) {
		throw std::invalid_argument(
			"desired acceleration vector size not consistent with task dof in "
			"JointTask::setDesiredAcceleration\n");
	}
	_desired_acceleration = desired_acceleration;
}

void JointTask::setGains(const VectorXd& kp, const VectorXd& kv,
						 const VectorXd& ki) {
	if (kp.size() != _task_dof || kv.size() != _task_dof ||
		ki.size() != _task_dof) {
		throw std::invalid_argument(
			"size of gain vector inconsistent with number of task dofs in "
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
	_kp = kp.asDiagonal();
	_kv = kv.asDiagonal();
	_ki = ki.asDiagonal();
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
	_kp = kp * MatrixXd::Identity(_task_dof, _task_dof);
	_kv = kv * MatrixXd::Identity(_task_dof, _task_dof);
	_ki = ki * MatrixXd::Identity(_task_dof, _task_dof);
}

void JointTask::updateTaskModel(const MatrixXd& N_prec) {
	const int robot_dof = getConstRobotModel()->dof();
	if (N_prec.rows() != N_prec.cols()) {
		throw std::invalid_argument(
			"N_prec matrix not square in JointTask::updateTaskModel\n");
	}
	if (N_prec.rows() != robot_dof) {
		throw std::invalid_argument(
			"N_prec matrix size not consistent with robot dof in "
			"JointTask::updateTaskModel\n");
	}

	_N_prec = N_prec;
	_projected_jacobian = _joint_selection * _N_prec;

	if (_is_partial_joint_task) {
		_current_task_range = Sai2Model::matrixRangeBasis(_projected_jacobian);
		if (_current_task_range.norm() == 0) {
			// there is no controllable degree of freedom for the task, just
			// return should maybe print a warning here
			_N.setIdentity(robot_dof, robot_dof);
			return;
		}

		Sai2Model::OpSpaceMatrices op_space_matrices =
			getConstRobotModel()->operationalSpaceMatrices(
				_current_task_range.transpose() * _projected_jacobian);
		_M_partial = op_space_matrices.Lambda;
		_N = op_space_matrices.N;
	} else {
		_current_task_range = MatrixXd::Identity(_task_dof, _task_dof);
		_M_partial = getConstRobotModel()->M();
		_N = MatrixXd::Zero(robot_dof, robot_dof);
	}

	switch (_dynamic_decoupling_type) {
		case FULL_DYNAMIC_DECOUPLING: {
			_M_partial_modified = _M_partial;
			break;
		}

		case BOUNDED_INERTIA_ESTIMATES: {
			MatrixXd M_BIE = getConstRobotModel()->M();
			for (int i = 0; i < getConstRobotModel()->dof(); i++) {
				if (M_BIE(i, i) < 0.1) {
					M_BIE(i, i) = 0.1;
				}
			}
			if (_is_partial_joint_task) {
				MatrixXd M_inv_BIE = M_BIE.inverse();
				_M_partial_modified =
					(_current_task_range.transpose() * _projected_jacobian *
					 M_inv_BIE * _projected_jacobian.transpose() *
					 _current_task_range)
						.inverse();
			} else {
				_M_partial_modified = M_BIE;
			}
			break;
		}

		case IMPEDANCE: {
			_M_partial_modified =
				MatrixXd::Identity(_current_task_range.cols(), _current_task_range.cols());
			break;
		}

		default: {
			// should not happen
			throw std::invalid_argument(
				"Dynamic decoupling type not recognized in "
				"JointTask::updateTaskModel\n");
			break;
		}
	}
}

VectorXd JointTask::computeTorques() {
	VectorXd partial_joint_task_torques = VectorXd::Zero(_task_dof);
	_projected_jacobian = _joint_selection * _N_prec;

	// update constroller state
	_current_position = _joint_selection * getConstRobotModel()->q();
	_current_velocity = _projected_jacobian * getConstRobotModel()->dq();

	if(_current_task_range.norm() == 0)
	{
		// there is no controllable degree of freedom for the task, just return
		// zero torques. should maybe print a warning here
		return partial_joint_task_torques;
	}

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
		(_current_position - tmp_desired_position) * getLoopTimestep();

	// compute task force (with velocity saturation if asked)
	if (_use_velocity_saturation_flag) {
		tmp_desired_velocity =
			-_kp * _kv.inverse() * (_current_position - tmp_desired_position) -
			_ki * _kv.inverse() * _integrated_position_error;
		for (int i = 0; i < getConstRobotModel()->dof(); i++) {
			if (tmp_desired_velocity(i) > _saturation_velocity(i)) {
				tmp_desired_velocity(i) = _saturation_velocity(i);
			} else if (tmp_desired_velocity(i) < -_saturation_velocity(i)) {
				tmp_desired_velocity(i) = -_saturation_velocity(i);
			}
		}
		partial_joint_task_torques =
			-_kv * (_current_velocity - tmp_desired_velocity);
	} else {
		partial_joint_task_torques =
			-_kp * (_current_position - tmp_desired_position) -
			_kv * (_current_velocity - tmp_desired_velocity) -
			_ki * _integrated_position_error;
	}

	VectorXd partial_joint_task_torques_in_range_space =
		_M_partial * _current_task_range.transpose() * tmp_desired_acceleration +
		_M_partial_modified * _current_task_range.transpose() * partial_joint_task_torques;

	// return projected task torques
	return _projected_jacobian.transpose() * _current_task_range *
		   partial_joint_task_torques_in_range_space;
}

void JointTask::enableInternalOtgAccelerationLimited(
	const VectorXd& max_velocity, const VectorXd& max_acceleration) {
	if (max_velocity.size() != _task_dof ||
		max_acceleration.size() != _task_dof) {
		throw std::invalid_argument(
			"max velocity or max acceleration vector size not consistent with "
			"task dof in JointTask::enableInternalOtgAccelerationLimited\n");
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
	if (max_velocity.size() != _task_dof ||
		max_acceleration.size() != _task_dof || max_jerk.size() != _task_dof) {
		throw std::invalid_argument(
			"max velocity, max acceleration or max jerk vector size not "
			"consistent with task dof in "
			"JointTask::enableInternalOtgJerkLimited\n");
	}
	_use_internal_otg_flag = true;
	_otg->reInitialize(_current_position);
	_otg->setMaxVelocity(max_velocity);
	_otg->setMaxAcceleration(max_acceleration);
	_otg->setMaxJerk(max_jerk);
}

void JointTask::enableVelocitySaturation(const VectorXd& saturation_velocity) {
	if (saturation_velocity.size() != _task_dof) {
		throw std::invalid_argument(
			"saturation velocity vector size not consistent with task dof in "
			"JointTask::enableVelocitySaturation\n");
	}
	_use_velocity_saturation_flag = true;
	_saturation_velocity = saturation_velocity;
}

} /* namespace Sai2Primitives */
