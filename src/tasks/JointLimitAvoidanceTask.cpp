/*
 * JointLimitAvoidanceTaskTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "JointLimitAvoidanceTask.h"

#include <algorithm>
#include <stdexcept>

using namespace Eigen;
namespace SaiPrimitives {

namespace {
double computeBlendingCoefficient(
	double z, double z1, double z2,
	JointLimitAvoidanceTask::LimitDirection direction) {
	if (direction == JointLimitAvoidanceTask::LimitDirection::NEGATIVE) {
		if (z >= z1) {
			return 0;
		}
		if (z <= z2) {
			return 1;
		}
		return (z1 - z) / (z1 - z2);
	} else {
		if (z <= z1) {
			return 0;
		}
		if (z >= z2) {
			return 1;
		}
		return (z - z1) / (z2 - z1);
	}
}

}  // namespace

JointLimitAvoidanceTask::JointLimitAvoidanceTask(
	std::shared_ptr<SaiModel::SaiModel>& robot, const std::string& task_name)
	: TemplateTask(robot, task_name, TaskType::JOINT_LIMIT_AVOIDANCE_TASK,
				   0.001) {
	// selection for full joint task
	_joint_selection = MatrixXd::Identity(0, getConstRobotModel()->dof());
	initialSetup();
}

void JointLimitAvoidanceTask::initialSetup() {
	const int robot_dof = getConstRobotModel()->dof();
	_active_constraints = _joint_selection.rows();

	_limit_status = vector<LimitStatus>(robot_dof, LimitStatus::OFF);
	_limit_direction =
		vector<LimitDirection>(robot_dof, LimitDirection::POSITIVE);
	_limit_value = vector<double>(robot_dof, 0);
	_torque_limit_value = vector<double>(robot_dof, 0);
	_enabled = DefaultParameters::enabled;
	_kv = DefaultParameters::kv;
	_position_z1_to_limit = DefaultParameters::position_z1_to_limit;
	_position_z2_to_limit = DefaultParameters::position_z2_to_limit;
	_velocity_z1_to_limit = DefaultParameters::velocity_z1_to_limit;
	_velocity_z2_to_limit = DefaultParameters::velocity_z2_to_limit;
	_max_torque_ratio_pos_limit = DefaultParameters::max_torque_ratio_pos_limit;
	_max_torque_ratio_vel_limit = DefaultParameters::max_torque_ratio_vel_limit;

	if (_position_z1_to_limit <= 0 || _position_z2_to_limit <= 0 ||
		_velocity_z1_to_limit <= 0 || _velocity_z2_to_limit <= 0) {
		throw std::invalid_argument(
			"JointLimitAvoidanceTask: negative limit avoidance parameters. z1 "
			"to limit and z2 to limit must be positive for position and "
			"velocity\n");
	}
	if (_position_z1_to_limit <= _position_z2_to_limit ||
		_velocity_z1_to_limit <= _velocity_z2_to_limit) {
		throw std::invalid_argument(
			"JointLimitAvoidanceTask: invalid limit avoidance parameters, z1 "
			"needs to be bigger than z2\n");
	}

	// initialize matrices sizes
	_N_prec = MatrixXd::Identity(robot_dof, robot_dof);
	_M_partial = MatrixXd::Identity(_active_constraints, _active_constraints);
	_projected_jacobian = _joint_selection;
	_N = MatrixXd::Zero(robot_dof, robot_dof);
	_current_task_range =
		MatrixXd::Identity(_active_constraints, _active_constraints);

	reInitializeTask();
}

void JointLimitAvoidanceTask::reInitializeTask() {
	computeJointSelectionMatrix();
}

void JointLimitAvoidanceTask::updateTaskModel(const MatrixXd& N_prec) {
	const int robot_dof = getConstRobotModel()->dof();
	if (N_prec.rows() != N_prec.cols()) {
		throw std::invalid_argument(
			"N_prec matrix not square in "
			"JointLimitAvoidanceTask::updateTaskModel\n");
	}
	if (N_prec.rows() != robot_dof) {
		throw std::invalid_argument(
			"N_prec matrix size not consistent with robot dof in "
			"JointLimitAvoidanceTask::updateTaskModel\n");
	}

	if (!_enabled) {
		_N = Eigen::MatrixXd::Identity(robot_dof, robot_dof);
		_N_prec = Eigen::MatrixXd::Identity(robot_dof, robot_dof);
		return;
	}

	_N_prec = N_prec;
	computeJointSelectionMatrix();
	_projected_jacobian = _joint_selection * _N_prec;

	MatrixXd unconstrained_jacobian_task_range = SaiModel::matrixRangeBasis(
		_joint_selection);
	if (unconstrained_jacobian_task_range.norm() == 0) {
		_N_unconstrained = MatrixXd::Identity(robot_dof, robot_dof);
	} else {
		_N_unconstrained = getConstRobotModel()->nullspaceMatrix(
			unconstrained_jacobian_task_range.transpose() * _joint_selection);
	}

	_current_task_range = SaiModel::matrixRangeBasis(_projected_jacobian);
	if (_current_task_range.norm() == 0) {
		// there is no controllable degree of freedom for the task, just
		// return should maybe print a warning here
		_N = Eigen::MatrixXd::Identity(robot_dof, robot_dof);
		return;
	}

	SaiModel::OpSpaceMatrices op_space_matrices =
		getConstRobotModel()->operationalSpaceMatrices(
			_current_task_range.transpose() * _projected_jacobian);
	_M_partial = op_space_matrices.Lambda;
	_N = op_space_matrices.N;
}

void JointLimitAvoidanceTask::updateLimitStatus() {
	for (const auto& joint_limit : getConstRobotModel()->jointLimits()) {
		int index = joint_limit.joint_index;
		_limit_status[index] = LimitStatus::OFF;
		_limit_direction[index] = LimitDirection::POSITIVE;
		_limit_value[index] = 0;
		_torque_limit_value[index] = 0;
		double q = getConstRobotModel()->q()(index);
		double dq = getConstRobotModel()->dq()(index);

		bool pos_limit_valid =
			joint_limit.position_upper - joint_limit.position_lower >
			2 * _position_z1_to_limit;
		bool vel_limit_valid = joint_limit.velocity > 2 * _velocity_z1_to_limit;
		if (!pos_limit_valid) {
			std::cout << "JointLimitAvoidanceTask: joint "
					  << joint_limit.joint_name
					  << " has position limits to close to each other "
						 "considering the desired buffer zones. ignoring this "
						 "joint for position limits\n";
		}
		if (!vel_limit_valid) {
			std::cout << "JointLimitAvoidanceTask: joint "
					  << joint_limit.joint_name
					  << " has velocity limits too low considering the desired "
						 "buffer zones. ignoring this joint for velocity "
						 "limits\n";
		}

		if (pos_limit_valid &&
			joint_limit.position_upper != std::numeric_limits<double>::max()) {
			if (q > joint_limit.position_upper - _position_z1_to_limit) {
				_limit_direction[index] = LimitDirection::POSITIVE;
				_limit_value[index] = joint_limit.position_upper;
				_torque_limit_value[index] = joint_limit.effort;
				_limit_status[index] = LimitStatus::POS_Z1;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at positive position limit Z1" << endl;
				cout << "joint value: " << q
					 << " limit value: " << _limit_value[index]
					 << " z1 boundary: "
					 << joint_limit.position_upper - _position_z1_to_limit
					 << " z2 boundary: "
					 << joint_limit.position_upper - _position_z2_to_limit
					 << endl
					 << endl;
			}
			if (q > joint_limit.position_upper - _position_z2_to_limit) {
				_limit_status[index] = LimitStatus::POS_Z2;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at positive position limit Z2" << endl;
				cout << "joint value: " << q
					 << " limit value: " << _limit_value[index]
					 << " z2 boundary: "
					 << joint_limit.position_upper - _position_z2_to_limit
					 << endl
					 << endl;
			}
		}
		if (pos_limit_valid &&
			joint_limit.position_lower != -std::numeric_limits<double>::max()) {
			if (q < joint_limit.position_lower + _position_z1_to_limit) {
				_limit_direction[index] = LimitDirection::NEGATIVE;
				_limit_value[index] = joint_limit.position_lower;
				_torque_limit_value[index] = joint_limit.effort;
				_limit_status[index] = LimitStatus::POS_Z1;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at negative position limit Z1" << endl;
				cout << "joint value: " << q
					 << " limit value: " << _limit_value[index]
					 << " z1 boundary: "
					 << joint_limit.position_lower + _position_z1_to_limit
					 << " z2 boundary: "
					 << joint_limit.position_lower + _position_z2_to_limit
					 << endl
					 << endl;
			}
			if (q < joint_limit.position_lower + _position_z2_to_limit) {
				_limit_status[index] = LimitStatus::POS_Z2;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at negative position limit Z2" << endl;
				cout << "joint value: " << q
					 << " limit value: " << _limit_value[index]
					 << " z2 boundary: "
					 << joint_limit.position_lower + _position_z2_to_limit
					 << endl
					 << endl;
			}
		}
		if (vel_limit_valid &&
			(_limit_status[index] == LimitStatus::OFF ||
			 _limit_direction[index] == LimitDirection::NEGATIVE)) {
			if (dq > joint_limit.velocity - _velocity_z1_to_limit) {
				_limit_direction[index] = LimitDirection::POSITIVE;
				_limit_value[index] = joint_limit.velocity;
				_torque_limit_value[index] = joint_limit.effort;
				_limit_status[index] = LimitStatus::VEL_Z1;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at positive velocity limit Z1" << endl;
				cout << "joint value: " << dq
					 << " limit value: " << _limit_value[index]
					 << " z1 boundary: "
					 << joint_limit.velocity - _velocity_z1_to_limit
					 << " z2 boundary: "
					 << joint_limit.velocity - _velocity_z2_to_limit << endl
					 << endl;
			}
			if (dq > joint_limit.velocity - _velocity_z2_to_limit) {
				_limit_status[index] = LimitStatus::VEL_Z2;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at positive velocity limit Z2" << endl;
				cout << "joint value: " << dq
					 << " limit value: " << _limit_value[index]
					 << " z2 boundary: "
					 << joint_limit.velocity - _velocity_z2_to_limit << endl
					 << endl;
			}
		}
		if (vel_limit_valid &&
			(_limit_status[index] == LimitStatus::OFF ||
			 _limit_direction[index] == LimitDirection::POSITIVE)) {
			if (dq < -joint_limit.velocity + _velocity_z1_to_limit) {
				_limit_direction[index] = LimitDirection::NEGATIVE;
				_limit_value[index] = -joint_limit.velocity;
				_torque_limit_value[index] = joint_limit.effort;
				_limit_status[index] = LimitStatus::VEL_Z1;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at negative velocity limit Z1" << endl;
				cout << "joint value: " << dq
					 << " limit value: " << _limit_value[index]
					 << " z1 boundary: "
					 << -joint_limit.velocity + _velocity_z1_to_limit
					 << " z2 boundary: "
					 << -joint_limit.velocity + _velocity_z2_to_limit << endl
					 << endl;
			}
			if (dq < -joint_limit.velocity + _velocity_z2_to_limit) {
				_limit_status[index] = LimitStatus::VEL_Z2;
				cout << "JointLimitAvoidanceTask: joint "
					 << joint_limit.joint_name
					 << " is at negative velocity limit Z2" << endl;
				cout << "joint value: " << dq
					 << " limit value: " << _limit_value[index]
					 << " z2 boundary: "
					 << -joint_limit.velocity + _velocity_z2_to_limit << endl
					 << endl;
			}
		}
	}

	_active_constraints = getConstRobotModel()->dof() -
						  std::count(_limit_status.begin(), _limit_status.end(),
									 LimitStatus::OFF);
}

void JointLimitAvoidanceTask::computeJointSelectionMatrix() {
	updateLimitStatus();
	_joint_selection =
		MatrixXd::Zero(_active_constraints, getConstRobotModel()->dof());
	int row = 0;
	for (int i = 0; i < getConstRobotModel()->dof(); i++) {
		if (_limit_status[i] != LimitStatus::OFF) {
			_joint_selection(row, i) = 1;
			row++;
		}
	}
}

VectorXd JointLimitAvoidanceTask::computeTorques() {
	return computeTorques(Eigen::VectorXd::Zero(getConstRobotModel()->dof()));
}

VectorXd JointLimitAvoidanceTask::computeTorques(
	const Eigen::VectorXd& tau_tasks) {
	const int robot_dof = getConstRobotModel()->dof();
	if (!_enabled || _active_constraints == 0) {
		return Eigen::VectorXd::Zero(robot_dof);
	}

	VectorXd limit_avoidance_torques = VectorXd::Zero(_active_constraints);
	_projected_jacobian = _joint_selection * _N_prec;

	const VectorXd robot_q = getConstRobotModel()->q();
	const VectorXd robot_dq = getConstRobotModel()->dq();

	if (_current_task_range.norm() == 0) {
		// there is no controllable degree of freedom for the task, just
		// return zero torques. should maybe print a warning here
		return VectorXd::Zero(robot_dof);
	}

	int constraint_number = 0;
	for (int i = 0; i < robot_dof; ++i) {
		double alpha = 0;
		double tau_constraint_z1 = 0;
		double tau_constraint_z2 = 0;
		if (_limit_direction[i] == LimitDirection::POSITIVE) {
			switch (_limit_status[i]) {
				case LimitStatus::POS_Z1:
					alpha = computeBlendingCoefficient(
						robot_q(i), _limit_value[i] - _position_z1_to_limit,
						_limit_value[i] - _position_z2_to_limit,
						_limit_direction[i]);

					tau_constraint_z1 = tau_tasks(i) - _kv * robot_dq(i);

					limit_avoidance_torques[constraint_number] =
						(1 - alpha) * tau_tasks(i) + alpha * tau_constraint_z1;
					break;
				case LimitStatus::POS_Z2:
					alpha = computeBlendingCoefficient(
						robot_q(i), _limit_value[i] - _position_z2_to_limit,
						_limit_value[i], _limit_direction[i]);

					tau_constraint_z1 = tau_tasks(i) - _kv * robot_dq(i);
					tau_constraint_z2 =
						-_torque_limit_value[i] * _max_torque_ratio_pos_limit -
						_kv * robot_dq(i);
					limit_avoidance_torques[constraint_number] =
						(1 - alpha) * tau_constraint_z1 +
						alpha * tau_constraint_z2;

					break;
				case LimitStatus::VEL_Z1:
					alpha = computeBlendingCoefficient(
						robot_dq(i), _limit_value[i] - _velocity_z1_to_limit,
						_limit_value[i] - _velocity_z2_to_limit,
						_limit_direction[i]);

					tau_constraint_z1 = -_kv * robot_dq(i);

					limit_avoidance_torques[constraint_number] =
						(1 - alpha) * tau_tasks(i) + alpha * tau_constraint_z1;

					break;
				case LimitStatus::VEL_Z2:
					alpha = computeBlendingCoefficient(
						robot_dq(i), _limit_value[i] - _velocity_z2_to_limit,
						_limit_value[i], _limit_direction[i]);

					tau_constraint_z1 = -_kv * robot_dq(i);
					tau_constraint_z1 = std::max(
						std::min(tau_constraint_z1,
								 _torque_limit_value[i] *
									 _max_torque_ratio_vel_limit),
						-_torque_limit_value[i] * _max_torque_ratio_vel_limit);

					tau_constraint_z2 = -alpha * _torque_limit_value[i] *
										_max_torque_ratio_vel_limit;

					limit_avoidance_torques[constraint_number] =
						(1 - alpha) * tau_constraint_z1 +
						alpha * tau_constraint_z2;
					break;
				default:
					break;
			}
		} else if (_limit_direction[i] == LimitDirection::NEGATIVE) {
			switch (_limit_status[i]) {
				case LimitStatus::POS_Z1:
					alpha = computeBlendingCoefficient(
						robot_q(i), _limit_value[i] + _position_z1_to_limit,
						_limit_value[i] + _position_z2_to_limit,
						_limit_direction[i]);

					tau_constraint_z1 = tau_tasks(i) - _kv * robot_dq(i);
					tau_constraint_z1 = std::max(
						std::min(tau_constraint_z1,
								 _torque_limit_value[i] *
									 _max_torque_ratio_vel_limit),
						-_torque_limit_value[i] * _max_torque_ratio_vel_limit);

					limit_avoidance_torques[constraint_number] =
						alpha * tau_tasks(i) + (1 - alpha) * tau_constraint_z1;
					break;
				case LimitStatus::POS_Z2:
					alpha = computeBlendingCoefficient(
						robot_q(i), _limit_value[i] + _position_z2_to_limit,
						_limit_value[i], _limit_direction[i]);

					tau_constraint_z1 = tau_tasks(i) - _kv * robot_dq(i);
					tau_constraint_z2 =
						_torque_limit_value[i] * _max_torque_ratio_pos_limit -
						_kv * robot_dq(i);
					limit_avoidance_torques[constraint_number] =
						(1 - alpha) * tau_constraint_z1 +
						alpha * tau_constraint_z2;
					break;
				case LimitStatus::VEL_Z1:
					alpha = computeBlendingCoefficient(
						robot_dq(i), _limit_value[i] + _velocity_z1_to_limit,
						_limit_value[i] + _velocity_z2_to_limit,
						_limit_direction[i]);

					tau_constraint_z1 = -_kv * robot_dq(i);
					tau_constraint_z1 = std::max(
						std::min(tau_constraint_z1,
								 _torque_limit_value[i] *
									 _max_torque_ratio_vel_limit),
						-_torque_limit_value[i] * _max_torque_ratio_vel_limit);

					limit_avoidance_torques[constraint_number] =
						(1 - alpha) * tau_tasks(i) + alpha * tau_constraint_z1;
					break;
				case LimitStatus::VEL_Z2:
					alpha = computeBlendingCoefficient(
						robot_dq(i), _limit_value[i] + _velocity_z2_to_limit,
						_limit_value[i], _limit_direction[i]);

					tau_constraint_z1 = -_kv * robot_dq(i);
					tau_constraint_z1 = std::max(
						std::min(tau_constraint_z1,
								 _torque_limit_value[i] *
									 _max_torque_ratio_vel_limit),
						-_torque_limit_value[i] * _max_torque_ratio_vel_limit);
					tau_constraint_z2 =
						_torque_limit_value[i] * _max_torque_ratio_vel_limit;

					limit_avoidance_torques[constraint_number] =
						(1 - alpha) * tau_constraint_z1 +
						alpha * tau_constraint_z2;
					break;
				default:
					break;
			}
		}

		if (_limit_status[i] != LimitStatus::OFF) {
			constraint_number++;
		}

		if (_limit_status[i] != LimitStatus::OFF) {
			cout << "joint: " << i << " alpha: " << alpha
				 << " tau_tasks: " << tau_tasks(i)
				 << " tau_constraint_z1: " << tau_constraint_z1
				 << " tau_constraint_z2: " << tau_constraint_z2 << endl;
		}
	}

	cout << "limit avoidance torques: " << limit_avoidance_torques.transpose()
		 << endl;

	// return projected task torques
	return _projected_jacobian.transpose() * _current_task_range *
		   limit_avoidance_torques;
}

} /* namespace SaiPrimitives */