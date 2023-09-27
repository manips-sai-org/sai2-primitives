#include "RobotController.h"

using namespace Eigen;
using namespace std;

namespace Sai2Primitives {

RobotController::RobotController(std::shared_ptr<Sai2Model::Sai2Model>& robot,
								 vector<shared_ptr<TemplateTask>>& tasks)
	: _robot(robot), _tasks(tasks), _enable_gravity_compensation(false) {
	if (_tasks.size() == 0) {
		throw std::invalid_argument("RobotController must have at least one task");
	}
	for (auto& task : _tasks) {
		if (task->getConstRobotModel() != _robot) {
			throw std::invalid_argument(
				"All tasks must have the same robot model in RobotController");
		}
		if (task->getLoopTimestep() != _tasks[0]->getLoopTimestep()) {
			throw std::invalid_argument(
				"All tasks must have the same loop timestep in RobotController");
		}
	}
	_redundancy_completion_task =
		std::make_shared<JointTask>(_robot, _tasks[0]->getLoopTimestep());
}

void RobotController::updateControllerTaskModels() {
    const int dof = _robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);
    for (auto& task : _tasks) {
		task->updateTaskModel(N_prec);
        N_prec = task->getTaskAndPreviousNullspace();
	}
	_redundancy_completion_task->updateTaskModel(N_prec);
}

Eigen::VectorXd RobotController::computeControlTorques() {
    const int dof = _robot->dof();
    VectorXd control_torques = VectorXd::Zero(dof);
    VectorXd previous_tasks_disturbance = VectorXd::Zero(dof);
    for (auto& task : _tasks) {
        previous_tasks_disturbance = (MatrixXd::Identity(dof, dof) - task->getTaskNullspace().transpose()) * control_torques;
        control_torques += task->computeTorques() - previous_tasks_disturbance;
    }
	previous_tasks_disturbance = (MatrixXd::Identity(dof, dof) - _redundancy_completion_task->getTaskNullspace().transpose()) * control_torques;
	control_torques += _redundancy_completion_task->computeTorques() - previous_tasks_disturbance;

    if (_enable_gravity_compensation) {
        control_torques += _robot->jointGravityVector();
    }
    return control_torques;
}

void RobotController::reinitializeTasks() {
	for (auto& task : _tasks) {
		task->reInitializeTask();
	}
	_redundancy_completion_task->reInitializeTask();
}

} /* namespace Sai2Primitives */