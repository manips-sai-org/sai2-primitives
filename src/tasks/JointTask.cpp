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
	_is_partial_joint_task = false;
	_joint_selection = MatrixXd::Identity(getConstRobotModel()->dof(),
										  getConstRobotModel()->dof());
	_is_partial_joint_task = false;

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
	_is_partial_joint_task = true;
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
	_otg = make_shared<OTG_joints>(_joint_selection * getConstRobotModel()->q(),
								   getLoopTimestep());
	_otg->setMaxVelocity(M_PI / 3);
	_otg->setMaxAcceleration(M_PI);
	_otg->disableJerkLimits();

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
	if (kv.maxCoeff() < 1e-3 && _use_velocity_saturation_flag) {
		throw std::invalid_argument(
			"cannot set singular kv if using velocity saturation in "
			"JointTask::setGains\n");
	}

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
	if (kv < 1e-3 && _use_velocity_saturation_flag) {
		throw std::invalid_argument(
			"cannot set singular kv if using velocity saturation in "
			"JointTask::setGains\n");
	}

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
		_N.setZero(robot_dof, robot_dof);
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
		return partial_joint_task_torques;
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
		_desired_velocity =
			-_kp * _kv.inverse() * (_current_position - _desired_position) -
			_ki * _kv.inverse() * _integrated_position_error;
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
		_M_partial * _current_task_range.transpose() *
			_desired_acceleration +
		_M_partial_modified * _current_task_range.transpose() *
			partial_joint_task_torques;

	// return projected task torques
	return _projected_jacobian.transpose() * _current_task_range *
		   partial_joint_task_torques_in_range_space;
}

void JointTask::enableInternalOtgAccelerationLimited(
	const VectorXd& max_velocity, const VectorXd& max_acceleration) {
	if (max_velocity.size() == 1 && max_acceleration.size() == 1) {
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
	_otg->setMaxVelocity(max_velocity);
	_otg->setMaxAcceleration(max_acceleration);
	_otg->disableJerkLimits();
	if (!_use_internal_otg_flag) {
		_otg->reInitialize(_current_position);
	}
	_use_internal_otg_flag = true;
}

void JointTask::enableInternalOtgJerkLimited(const VectorXd& max_velocity,
											 const VectorXd& max_acceleration,
											 const VectorXd& max_jerk) {
	if (max_velocity.size() == 1 && max_acceleration.size() == 1 &&
		max_jerk.size() == 1) {
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
	_otg->setMaxVelocity(max_velocity);
	_otg->setMaxAcceleration(max_acceleration);
	_otg->setMaxJerk(max_jerk);
	if (!_use_internal_otg_flag) {
		_otg->reInitialize(_current_position);
	}
	_use_internal_otg_flag = true;
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
	_use_velocity_saturation_flag = true;
	_saturation_velocity = saturation_velocity;
}

} /* namespace Sai2Primitives */