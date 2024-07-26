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
	_joint_selection = MatrixXd::Identity(getConstRobotModel()->dof(),
										  getConstRobotModel()->dof());
	initialSetup();
}

JointTask::JointTask(std::shared_ptr<Sai2Model::Sai2Model>& robot,
					 const MatrixXd& joint_selection_matrix,
					 const std::string& task_name, const double loop_timestep)
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

	initialSetup();
}

void JointTask::initialSetup() {
	const int robot_dof = getConstRobotModel()->dof();
	_task_dof = _joint_selection.rows();
	setDynamicDecouplingType(DefaultParameters::dynamic_decoupling_type);
	setBoundedInertiaEstimateThreshold(DefaultParameters::bie_threshold);
	_current_position = _joint_selection * getConstRobotModel()->q();

	// default values for gains and velocity saturation
	setGains(DefaultParameters::kp, DefaultParameters::kv,
			 DefaultParameters::ki);

	if (DefaultParameters::use_velocity_saturation) {
		enableVelocitySaturation(DefaultParameters::saturation_velocity);
	} else {
		disableVelocitySaturation();
	}

	// initialize matrices sizes
	_N_prec = MatrixXd::Identity(robot_dof, robot_dof);
	_M_partial = MatrixXd::Identity(_task_dof, _task_dof);
	_M_partial_modified = MatrixXd::Identity(_task_dof, _task_dof);
	_projected_jacobian = _joint_selection;
	_N = MatrixXd::Zero(robot_dof, robot_dof);
	_current_task_range = MatrixXd::Identity(_task_dof, _task_dof);

	// initialize internal otg
	_otg = make_shared<OTG_joints>(_joint_selection * getConstRobotModel()->q(),
								   getLoopTimestep());
	if (DefaultParameters::use_internal_otg) {
		if (DefaultParameters::internal_otg_jerk_limited) {
			enableInternalOtgJerkLimited(
				DefaultParameters::otg_max_velocity,
				DefaultParameters::otg_max_acceleration,
				DefaultParameters::otg_max_jerk);
		} else {
			enableInternalOtgAccelerationLimited(
				DefaultParameters::otg_max_velocity,
				DefaultParameters::otg_max_acceleration);
		}
	} else {
		disableInternalOtg();
	}

	reInitializeTask();
}

void JointTask::reInitializeTask() {
	const int robot_dof = getConstRobotModel()->dof();

	_current_position = _joint_selection * getConstRobotModel()->q();
	_current_velocity.setZero(_task_dof);

	_goal_position = _current_position;
	_desired_position = _current_position;
	_goal_velocity.setZero(_task_dof);
	_goal_acceleration.setZero(_task_dof);
	_desired_velocity.setZero(_task_dof);
	_desired_acceleration.setZero(_task_dof);

	_integrated_position_error.setZero(_task_dof);

	_otg->reInitialize(_current_position);
}

void JointTask::setGoalPosition(const VectorXd& goal_position) {
	if (goal_position.size() != _task_dof) {
		throw std::invalid_argument(
			"goal position vector size not consistent with task dof in "
			"JointTask::setGoalPosition\n");
	}
	_goal_position = goal_position;
}

void JointTask::setGoalVelocity(const VectorXd& goal_velocity) {
	if (goal_velocity.size() != _task_dof) {
		throw std::invalid_argument(
			"goal velocity vector size not consistent with task dof in "
			"JointTask::setGoalVelocity\n");
	}
	_goal_velocity = goal_velocity;
}

void JointTask::setGoalAcceleration(const VectorXd& goal_acceleration) {
	if (goal_acceleration.size() != _task_dof) {
		throw std::invalid_argument(
			"goal acceleration vector size not consistent with task dof in "
			"JointTask::setGoalAcceleration\n");
	}
	_goal_acceleration = goal_acceleration;
}

void JointTask::setGainsUnsafe(const VectorXd& kp, const VectorXd& kv,
							   const VectorXd& ki) {
	if (kp.size() == 1 && kv.size() == 1 && ki.size() == 1) {
		_are_gains_isotropic = true;
		_kp = kp(0) * MatrixXd::Identity(_task_dof, _task_dof);
		_kv = kv(0) * MatrixXd::Identity(_task_dof, _task_dof);
		_ki = ki(0) * MatrixXd::Identity(_task_dof, _task_dof);
		return;
	}

	if (kp.size() != _task_dof || kv.size() != _task_dof ||
		ki.size() != _task_dof) {
		throw std::invalid_argument(
			"size of gain vectors inconsistent with number of task dofs in "
			"JointTask::setGains\n");
	}
	_are_gains_isotropic = false;
	_kp = kp.asDiagonal();
	_kv = kv.asDiagonal();
	_ki = ki.asDiagonal();
}

void JointTask::setGains(const VectorXd& kp, const VectorXd& kv,
						 const VectorXd& ki) {
	if (kp.size() == 1 && kv.size() == 1 && ki.size() == 1) {
		setGains(kp(0), kv(0), ki(0));
		return;
	}

	if (kp.size() != _task_dof || kv.size() != _task_dof ||
		ki.size() != _task_dof) {
		throw std::invalid_argument(
			"size of gain vectors inconsistent with number of task dofs in "
			"JointTask::setGains\n");
	}
	if (kp.maxCoeff() < 0 || kv.maxCoeff() < 0 || ki.maxCoeff() < 0) {
		throw std::invalid_argument(
			"gains must be positive or zero in "
			"JointTask::setGains\n");
	}
	// TODO: print warning if kv is too small
	// if (kv.maxCoeff() < 1e-3 && _use_velocity_saturation_flag) {
	// 	throw std::invalid_argument(
	// 		"cannot set singular kv if using velocity saturation in "
	// 		"JointTask::setGains\n");
	// }

	_are_gains_isotropic = false;
	_kp = kp.asDiagonal();
	_kv = kv.asDiagonal();
	_ki = ki.asDiagonal();
}

void JointTask::setGains(const double kp, const double kv, const double ki) {
	if (kp < 0 || kv < 0 || ki < 0) {
		throw std::invalid_argument(
			"gains must be positive or zero in JointTask::setGains\n");
	}
	// TODO: print warning if kv is too small
	// if (kv < 1e-3 && _use_velocity_saturation_flag) {
	// 	throw std::invalid_argument(
	// 		"cannot set singular kv if using velocity saturation in "
	// 		"JointTask::setGains\n");
	// }

	_are_gains_isotropic = true;
	_kp = kp * MatrixXd::Identity(_task_dof, _task_dof);
	_kv = kv * MatrixXd::Identity(_task_dof, _task_dof);
	_ki = ki * MatrixXd::Identity(_task_dof, _task_dof);
}

vector<PIDGains> JointTask::getGains() const {
	if (_are_gains_isotropic) {
		return vector<PIDGains>(1, PIDGains(_kp(0, 0), _kv(0, 0), _ki(0, 0)));
	}
	vector<PIDGains> gains = {};
	for (int i = 0; i < _task_dof; i++) {
		gains.push_back(PIDGains(_kp(i, i), _kv(i, i), _ki(i, i)));
	}
	return gains;
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

	_current_task_range = Sai2Model::matrixRangeBasis(_projected_jacobian);
	if (_current_task_range.norm() == 0) {
		// there is no controllable degree of freedom for the task, just
		// return should maybe print a warning here
		_N = Eigen::MatrixXd::Identity(robot_dof, robot_dof);
		return;
	}

	Sai2Model::OpSpaceMatrices op_space_matrices =
		getConstRobotModel()->operationalSpaceMatrices(
			_current_task_range.transpose() * _projected_jacobian);
	_M_partial = op_space_matrices.Lambda;
	_N = op_space_matrices.N;

	switch (_dynamic_decoupling_type) {
		case FULL_DYNAMIC_DECOUPLING: {
			_M_partial_modified = _M_partial;
			break;
		}

		case BOUNDED_INERTIA_ESTIMATES: {
			MatrixXd M_BIE = getConstRobotModel()->M();
			for (int i = 0; i < getConstRobotModel()->dof(); i++) {
				if (M_BIE(i, i) < _bie_threshold) {
					M_BIE(i, i) = _bie_threshold;
				}
			}
			MatrixXd M_inv_BIE = M_BIE.inverse();
			_M_partial_modified =
				(_current_task_range.transpose() * _projected_jacobian *
				 M_inv_BIE * _projected_jacobian.transpose() *
				 _current_task_range)
					.inverse();
			break;
		}

		case IMPEDANCE: {
			_M_partial_modified = MatrixXd::Identity(
				_current_task_range.cols(), _current_task_range.cols());
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

	if (_current_task_range.norm() == 0) {
		// there is no controllable degree of freedom for the task, just return
		// zero torques. should maybe print a warning here
		return VectorXd::Zero(getConstRobotModel()->dof());
	}

	_desired_position = _goal_position;
	_desired_velocity = _goal_velocity;
	_desired_acceleration = _goal_acceleration;

	// compute next state from trajectory generation
	if (_use_internal_otg_flag) {
		_otg->setGoalPositionAndVelocity(_goal_position, _goal_velocity);
		_otg->update();

		_desired_position = _otg->getNextPosition();
		_desired_velocity = _otg->getNextVelocity();
		_desired_acceleration = _otg->getNextAcceleration();
	}

	// compute error for I term
	_integrated_position_error +=
		(_current_position - _desired_position) * getLoopTimestep();

	// compute task force (with velocity saturation if asked)
	if (_use_velocity_saturation_flag) {
		const MatrixXd kv_inverse = Sai2Model::computePseudoInverse(_kv);
		_desired_velocity =
			-_kp * kv_inverse * (_current_position - _desired_position) -
			_ki * kv_inverse * _integrated_position_error;
		for (int i = 0; i < getConstRobotModel()->dof(); i++) {
			if (_desired_velocity(i) > _saturation_velocity(i)) {
				_desired_velocity(i) = _saturation_velocity(i);
			} else if (_desired_velocity(i) < -_saturation_velocity(i)) {
				_desired_velocity(i) = -_saturation_velocity(i);
			}
		}
		partial_joint_task_torques =
			-_kv * (_current_velocity - _desired_velocity);
	} else {
		partial_joint_task_torques =
			-_kp * (_current_position - _desired_position) -
			_kv * (_current_velocity - _desired_velocity) -
			_ki * _integrated_position_error;
	}

	VectorXd partial_joint_task_torques_in_range_space =
		_M_partial * _current_task_range.transpose() * _desired_acceleration +
		_M_partial_modified * _current_task_range.transpose() *
			partial_joint_task_torques;

	// return projected task torques
	return _projected_jacobian.transpose() * _current_task_range *
		   partial_joint_task_torques_in_range_space;
}

void JointTask::enableInternalOtgAccelerationLimited(
	const VectorXd& max_velocity, const VectorXd& max_acceleration) {
	if (max_velocity.size() == 1 && max_acceleration.size() == 1 && _task_dof != 1) {
		enableInternalOtgAccelerationLimited(max_velocity(0),
											 max_acceleration(0));
		return;
	}

	if (max_velocity.size() != _task_dof ||
		max_acceleration.size() != _task_dof) {
		throw std::invalid_argument(
			"max velocity or max acceleration vector size not consistent with "
			"task dof in JointTask::enableInternalOtgAccelerationLimited\n");
	}
	if (!_use_internal_otg_flag || _otg->getJerkLimitEnabled()) {
		_otg->reInitialize(_current_position);
	}
	_otg->setMaxVelocity(max_velocity);
	_otg->setMaxAcceleration(max_acceleration);
	_otg->disableJerkLimits();
	_use_internal_otg_flag = true;
}

void JointTask::enableInternalOtgJerkLimited(const VectorXd& max_velocity,
											 const VectorXd& max_acceleration,
											 const VectorXd& max_jerk) {
	if (max_velocity.size() == 1 && max_acceleration.size() == 1 &&
		max_jerk.size() == 1 && _task_dof != 1) {
		enableInternalOtgJerkLimited(max_velocity(0), max_acceleration(0),
									 max_jerk(0));
		return;
	}
	if (max_velocity.size() != _task_dof ||
		max_acceleration.size() != _task_dof || max_jerk.size() != _task_dof) {
		throw std::invalid_argument(
			"max velocity, max acceleration or max jerk vector size not "
			"consistent with task dof in "
			"JointTask::enableInternalOtgJerkLimited\n");
	}
	if (!_use_internal_otg_flag || !_otg->getJerkLimitEnabled()) {
		_otg->reInitialize(_current_position);
	}
	_otg->setMaxVelocity(max_velocity);
	_otg->setMaxAcceleration(max_acceleration);
	_otg->setMaxJerk(max_jerk);
	_use_internal_otg_flag = true;
}

void JointTask::enableVelocitySaturation(const double saturation_velocity) {
	if (saturation_velocity <= 0) {
		throw std::invalid_argument(
			"saturation velocity must be positive in "
			"JointTask::enableVelocitySaturation\n");
	}
	_use_velocity_saturation_flag = true;
	_saturation_velocity = VectorXd::Constant(_task_dof, saturation_velocity);
}

void JointTask::enableVelocitySaturation(const VectorXd& saturation_velocity) {
	if (saturation_velocity.size() == 1) {
		enableVelocitySaturation(saturation_velocity(0));
		return;
	}
	if (saturation_velocity.size() != _task_dof) {
		throw std::invalid_argument(
			"saturation velocity vector size not consistent with task dof in "
			"JointTask::enableVelocitySaturation\n");
	}
	if (saturation_velocity.minCoeff() <= 0) {
		throw std::invalid_argument(
			"saturation velocity must be positive in "
			"JointTask::enableVelocitySaturation\n");
	}
	_use_velocity_saturation_flag = true;
	_saturation_velocity = saturation_velocity;
}

bool JointTask::goalPositionReached(const double& tol) {
	double squared_error =
		(_current_position - _goal_position).transpose() * _current_task_range *
		_current_task_range.transpose() * (_current_position - _goal_position);
	if (std::sqrt(squared_error) < tol) {
		return true;
	} else {
		return false;
	}
}

} /* namespace Sai2Primitives */