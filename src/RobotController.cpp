#include "RobotController.h"

using namespace Eigen;
using namespace std;

namespace SaiPrimitives {

RobotController::RobotController(std::shared_ptr<SaiModel::SaiModel>& robot,
								 vector<shared_ptr<TemplateTask>>& tasks)
	: _robot(robot), _enable_gravity_compensation(false) {
	if (tasks.size() == 0) {
		throw std::invalid_argument(
			"RobotController must have at least one task");
	}

	_enable_gravity_compensation =
		DefaultParameters::enable_gravity_compensation;
	_enable_joint_limit_avoidance =
		DefaultParameters::enable_joint_limit_avoidance;
	_enable_torque_saturation = DefaultParameters::enable_torque_saturation;

	_joint_limit_avoidance_task = make_unique<JointLimitAvoidanceTask>(robot);
	_N_constraints = MatrixXd::Identity(robot->dof(), robot->dof());

	bool cannot_accept_new_tasks = false;

	for (auto& task : tasks) {
		if (task->getConstRobotModel() != _robot) {
			throw std::invalid_argument(
				"All tasks must have the same robot model in RobotController");
		}
		if (task->getLoopTimestep() != tasks[0]->getLoopTimestep()) {
			throw std::invalid_argument(
				"All tasks must have the same loop timestep in "
				"RobotController");
		}
		if (std::find(_task_names.begin(), _task_names.end(),
					  task->getTaskName()) != _task_names.end()) {
			throw std::invalid_argument(
				"Tasks in RobotController must have unique names");
		}
		_task_names.push_back(task->getTaskName());
		_tasks.push_back(task);

		if (cannot_accept_new_tasks) {
			throw std::invalid_argument(
				"task [" + task->getTaskName() +
				"] cannot be added to the controller because it is in the "
				"nullspace of a full joint task");
			break;
		}

		if (task->getTaskType() == TaskType::JOINT_TASK) {
			auto joint_task = std::dynamic_pointer_cast<JointTask>(task);
			if (joint_task->isFullJointTask()) {
				cannot_accept_new_tasks = true;
			}
		}
	}

	_torque_limits =
		std::numeric_limits<double>::max() * VectorXd::Ones(robot->dof());
	for (const auto& limit : robot->jointLimits()) {
		_torque_limits[limit.joint_index] = limit.effort;
	}
}

void RobotController::updateControllerTaskModels() {
	const int dof = _robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);
	_joint_limit_avoidance_task->updateTaskModel(N_prec);
	_N_constraints = _joint_limit_avoidance_task->getTaskAndPreviousNullspace();
	for (auto& task : _tasks) {
		task->updateTaskModel(N_prec);
		N_prec = task->getTaskAndPreviousNullspace();
	}
}

Eigen::VectorXd RobotController::computeControlTorques() {
	const int dof = _robot->dof();
	VectorXd control_torques = VectorXd::Zero(dof);
	for (auto& task : _tasks) {
		control_torques += task->computeTorques(control_torques);
	}

	if (_enable_torque_saturation) {
		for (int i = 0; i < dof; i++) {
			if (control_torques[i] > _torque_limits[i]) {
				control_torques[i] = _torque_limits[i];
			} else if (control_torques[i] < -_torque_limits[i]) {
				control_torques[i] = -_torque_limits[i];
			}
		}
	}

	if (_enable_joint_limit_avoidance) {
		VectorXd joint_limit_avoidance_torques =
			_joint_limit_avoidance_task->computeTorques(control_torques);

		control_torques = joint_limit_avoidance_torques +
						  _N_constraints.transpose() * control_torques;

		if (_enable_torque_saturation) {
			for (int i = 0; i < dof; i++) {
				if (control_torques[i] > _torque_limits[i]) {
					control_torques[i] = _torque_limits[i];
				} else if (control_torques[i] < -_torque_limits[i]) {
					control_torques[i] = -_torque_limits[i];
				}
			}
		}
	}

	if (_enable_gravity_compensation) {
		control_torques += _robot->jointGravityVector();
	}
	return control_torques;
}

void RobotController::reinitializeTasks() {
	_joint_limit_avoidance_task->reInitializeTask();
	for (auto& task : _tasks) {
		task->reInitializeTask();
	}
}

std::shared_ptr<JointTask> RobotController::getJointTaskByName(
	const std::string& task_name) {
	for (auto& task : _tasks) {
		if (task->getTaskName() == task_name) {
			if (task->getTaskType() != TaskType::JOINT_TASK) {
				throw std::invalid_argument(
					"Task " + task_name +
					" is not a JointTask, and cannot be casted as such in "
					"RobotController::GetTaskByName");
			}
			return std::dynamic_pointer_cast<JointTask>(task);
		}
	}
	throw std::invalid_argument("Task " + task_name +
								" not found in RobotController::GetTaskByName");
}

std::shared_ptr<MotionForceTask> RobotController::getMotionForceTaskByName(
	const std::string& task_name) {
	for (auto& task : _tasks) {
		if (task->getTaskName() == task_name) {
			if (task->getTaskType() != TaskType::MOTION_FORCE_TASK) {
				throw std::invalid_argument("Task " + task_name +
											" is not a MotionForceTask, and "
											"cannot be casted as such in "
											"RobotController::GetTaskByName");
			}
			return std::dynamic_pointer_cast<MotionForceTask>(task);
		}
	}
	throw std::invalid_argument("Task " + task_name +
								" not found in RobotController::GetTaskByName");
}

} /* namespace SaiPrimitives */