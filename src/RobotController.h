/**
 * RobotController.h
 *
 *	A controller class for a single robot that takes a list of ordered tasks and
 *compute the control torques for the robot
 *
 * Author: Mikael Jorda
 * Created: September 2023
 */

#ifndef SAI_PRIMITIVES_ROBOT_CONTROLLER_H_
#define SAI_PRIMITIVES_ROBOT_CONTROLLER_H_

#include <memory>
#include <vector>

#include "tasks/TemplateTask.h"
#include "tasks/JointTask.h"
#include "tasks/MotionForceTask.h"

namespace SaiPrimitives {

class RobotController {
public:
	RobotController(std::shared_ptr<SaiModel::SaiModel>& robot, std::vector<std::shared_ptr<TemplateTask>>& tasks);

	void updateControllerTaskModels();

	Eigen::VectorXd computeControlTorques();

	void enableGravityCompensation(const bool enable_gravity_compensation) {
		_enable_gravity_compensation = enable_gravity_compensation;
	}

	void reinitializeTasks();

	std::shared_ptr<JointTask> getJointTaskByName(const std::string& task_name);
	std::shared_ptr<MotionForceTask> getMotionForceTaskByName(const std::string& task_name);

	const std::vector<std::string>& getTaskNames() const {
		return _task_names;
	}

private:
    std::shared_ptr<SaiModel::SaiModel> _robot;
	std::vector<std::shared_ptr<TemplateTask>> _tasks;
	std::vector<std::string> _task_names;
	bool _enable_gravity_compensation;
};

} /* namespace SaiPrimitives */

#endif /* SAI_PRIMITIVES_ROBOT_CONTROLLER_H_ */