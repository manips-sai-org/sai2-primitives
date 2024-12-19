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
#include "tasks/JointLimitAvoidanceTask.h"

namespace SaiPrimitives {

class RobotController {
public:
	struct DefaultParameters {
		static constexpr bool enable_gravity_compensation = false;
		static constexpr bool enable_joint_limit_avoidance = false;
		static constexpr bool enable_torque_saturation = false;
	};


	RobotController(std::shared_ptr<SaiModel::SaiModel>& robot, std::vector<std::shared_ptr<TemplateTask>>& tasks);

	void updateControllerTaskModels();

	Eigen::VectorXd computeControlTorques();

	void enableGravityCompensation(const bool enable_gravity_compensation) {
		_enable_gravity_compensation = enable_gravity_compensation;
	}

	void enableJointLimitAvoidance(const bool enable_joint_limit_avoidance) {
		_enable_joint_limit_avoidance = enable_joint_limit_avoidance;
	}

	void enableTorqueSaturation(const bool enable_torque_saturation) {
		_enable_torque_saturation = enable_torque_saturation;
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
	std::unique_ptr<JointLimitAvoidanceTask> _joint_limit_avoidance_task;
	MatrixXd _N_constraints;
	VectorXd _torque_limits;
	bool _enable_joint_limit_avoidance;
	bool _enable_torque_saturation;
};

} /* namespace SaiPrimitives */

#endif /* SAI_PRIMITIVES_ROBOT_CONTROLLER_H_ */