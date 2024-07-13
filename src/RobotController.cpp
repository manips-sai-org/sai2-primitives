#include "RobotController.h"

using namespace Eigen;
using namespace std;

namespace Sai2Primitives {

RobotController::RobotController(std::shared_ptr<Sai2Model::Sai2Model>& robot,
								 vector<shared_ptr<TemplateTask>>& tasks)
	: _robot(robot), _enable_gravity_compensation(false) {
	if (tasks.size() == 0) {
		throw std::invalid_argument(
			"RobotController must have at least one task");
	}

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
}

void RobotController::updateControllerTaskModels() {
	const int dof = _robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);
	for (auto& task : _tasks) {
		task->updateTaskModel(N_prec);
		N_prec = task->getTaskAndPreviousNullspace();
	}
}

Eigen::VectorXd RobotController::computeControlTorques() {
	const int dof = _robot->dof();
	VectorXd control_torques = VectorXd::Zero(dof);
	VectorXd previous_tasks_disturbance = VectorXd::Zero(dof);
	for (auto& task : _tasks) {
		previous_tasks_disturbance = (MatrixXd::Identity(dof, dof) -
									  task->getTaskNullspace().transpose()) *
									 control_torques;
		control_torques += task->getPreviousTasksNullspace().transpose() *
							   task->computeTorques() -
						   previous_tasks_disturbance;
	}

	if (_enable_gravity_compensation) {
		control_torques += _robot->jointGravityVector();
	}
	return control_torques;
}

void RobotController::reinitializeTasks() {
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

} /* namespace Sai2Primitives */