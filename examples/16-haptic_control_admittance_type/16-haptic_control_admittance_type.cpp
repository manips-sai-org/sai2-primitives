#include <signal.h>

#include <iostream>
#include <mutex>
#include <string>

#include "Sai2Graphics.h"
#include "Sai2Primitives.h"
#include "Sai2Simulation.h"
#include "redis/RedisClient.h"
#include "redis/keys/chai_haptic_devices_driver.h"
#include "timer/LoopTimer.h"
bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

using namespace std;
using namespace Eigen;
using namespace Sai2Common::ChaiHapticDriverKeys;

namespace {
const string world_file = "${EXAMPLE_16_FOLDER}/world.urdf";
const string robot_file = "${EXAMPLE_16_FOLDER}/panda_arm.urdf";
const string robot_name = "PANDA";
const string link_name = "end-effector";

// mutex for control torques
mutex mtx;

// map of flags for key presses
map<int, bool> key_pressed = {
	{GLFW_KEY_P, false},
	{GLFW_KEY_L, false},
	{GLFW_KEY_O, false},
};
map<int, bool> key_was_pressed = key_pressed;

}  // namespace

// Create simulation and control function
void runSim(shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void runControl(shared_ptr<Sai2Simulation::Sai2Simulation> sim);

//// Robot global variables /////
// sensed task force from robot interaction
Vector3d sensed_force = Vector3d::Zero();
Vector3d sensed_moment = Vector3d::Zero();
// robot joint data
VectorXd robot_control_torques = Eigen::VectorXd::Zero(7);

int main() {
	Sai2Model::URDF_FOLDERS["EXAMPLE_16_FOLDER"] =
		string(EXAMPLES_FOLDER) + "/16-haptic_control_admittance_type";
	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);

	// load graphics scene
	auto graphics = make_shared<Sai2Graphics::Sai2Graphics>(world_file);

	// Run simulation and control threads
	thread sim_thread(runSim, sim);
	thread control_thread(runControl, sim);

	// graphics timer
	Sai2Common::LoopTimer graphicsTimer(30.0, 1e6);

	while (graphics->isWindowOpen()) {
		graphicsTimer.waitForNextLoop();

		for (auto& key : key_pressed) {
			key_pressed[key.first] = graphics->isKeyPressed(key.first);
		}

		graphics->updateRobotGraphics(robot_name,
									  sim->getJointPositions(robot_name));
		graphics->renderGraphicsWorld();
	}

	// stop simulation and control threads
	fSimulationRunning = false;
	sim_thread.join();
	control_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
////// Simulation thread //////
//------------------------------------------------------------------------------
void runSim(shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	// create a timer
	Sai2Common::LoopTimer simTimer(1.0 / sim->timestep(), 1e6);

	fSimulationRunning = true;

	while (fSimulationRunning) {
		simTimer.waitForNextLoop();
		{
			lock_guard<mutex> lock(mtx);
			sim->setJointTorques(robot_name, robot_control_torques);
		}
		sim->integrate();
	}

	cout << "simulation timer stats:" << endl;
	simTimer.printInfoPostRun();
}

//------------------------------------------------------------------------------
////// Control thread //////
//------------------------------------------------------------------------------
void runControl(shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	// redis client
	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();

	// load robot
	const Affine3d T_world_robot = sim->getRobotBaseTransform(robot_name);
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file);
	robot->setTRobotBase(T_world_robot);
	robot->setQ(sim->getJointPositions(robot_name));
	robot->setDq(sim->getJointVelocities(robot_name));
	robot->updateModel();

	// instructions
	cout << "\nexmaple of a force-motion controller to control a simulated robot "
			"with a haptic device.. The controller will first bring the haptic "
			"device to its home pose and then switch automatically to "
			"force-motion control"
		 << endl;
	cout << "Provided options:" << endl;
	cout << "1. Press 'p' to enable/disable plane guidance" << endl;
	cout << "2. Press 'l' to enable/disable line guidance" << endl;
	cout << "3. Press 'o' to enable/disable orientation teleoperation" << endl;

	// create robot controller
	Affine3d compliant_frame = Affine3d::Identity();
	auto motion_force_task = make_shared<Sai2Primitives::MotionForceTask>(
		robot, link_name, compliant_frame);
	motion_force_task->disableInternalOtg();
	motion_force_task->enableVelocitySaturation(0.7, M_PI);

	auto joint_task = make_shared<Sai2Primitives::JointTask>(robot);
	vector<shared_ptr<Sai2Primitives::TemplateTask>> task_list = {
		motion_force_task, joint_task};
	auto robot_controller =
		make_unique<Sai2Primitives::RobotController>(robot, task_list);

	// create haptic controller
	Sai2Primitives::HapticDeviceController::DeviceLimits device_limits(
		redis_client.getEigen(createRedisKey(MAX_STIFFNESS_KEY_SUFFIX, 0)),
		redis_client.getEigen(createRedisKey(MAX_DAMPING_KEY_SUFFIX, 0)),
		redis_client.getEigen(createRedisKey(MAX_FORCE_KEY_SUFFIX, 0)));
	Affine3d device_home_pose = Affine3d(Translation3d(0, 0, 0));
	auto haptic_controller =
		make_shared<Sai2Primitives::HapticDeviceController>(
			device_limits, robot->transformInWorld(link_name),
			device_home_pose);
	haptic_controller->setHapticControlType(
		Sai2Primitives::HapticControlType::HOMING);
	haptic_controller->disableOrientationTeleop();

	Sai2Primitives::HapticControllerInput haptic_input;
	Sai2Primitives::HapticControllerOtuput haptic_output;

	// setup redis communication
	redis_client.addToSendGroup(createRedisKey(COMMANDED_FORCE_KEY_SUFFIX, 0),
								haptic_output.device_command_force);
	redis_client.addToSendGroup(createRedisKey(COMMANDED_TORQUE_KEY_SUFFIX, 0),
								haptic_output.device_command_moment);

	redis_client.addToReceiveGroup(createRedisKey(POSITION_KEY_SUFFIX, 0),
								   haptic_input.device_position);
	redis_client.addToReceiveGroup(createRedisKey(ROTATION_KEY_SUFFIX, 0),
								   haptic_input.device_orientation);
	redis_client.addToReceiveGroup(
		createRedisKey(LINEAR_VELOCITY_KEY_SUFFIX, 0),
		haptic_input.device_linear_velocity);
	redis_client.addToReceiveGroup(
		createRedisKey(ANGULAR_VELOCITY_KEY_SUFFIX, 0),
		haptic_input.device_angular_velocity);

	// create a timer
	Sai2Common::LoopTimer controlTimer(1000.0, 1e6);

	while (fSimulationRunning) {
		// wait for next scheduled loop
		controlTimer.waitForNextLoop();

		// read robot data from simulation thread
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		robot_controller->updateControllerTaskModels();

		// read haptic device state from redis
		redis_client.receiveAllFromGroup();

		// compute haptic control
		haptic_input.robot_position = robot->positionInWorld(link_name);
		haptic_input.robot_orientation = robot->rotationInWorld(link_name);
		haptic_input.robot_linear_velocity =
			robot->linearVelocityInWorld(link_name);
		haptic_input.robot_angular_velocity =
			robot->angularVelocityInWorld(link_name);

		haptic_output = haptic_controller->computeHapticControl(haptic_input);

		redis_client.sendAllFromGroup();

		// compute robot control
		motion_force_task->setGoalPosition(haptic_output.robot_goal_position);
		motion_force_task->setGoalOrientation(
			haptic_output.robot_goal_orientation);

		{
			lock_guard<mutex> lock(mtx);
			robot_control_torques = robot_controller->computeControlTorques();
		}

		// state machine for button presses
		if (haptic_controller->getHapticControlType() ==
				Sai2Primitives::HapticControlType::HOMING &&
			haptic_controller->getHomed()) {
			haptic_controller->setHapticControlType(
				Sai2Primitives::HapticControlType::FORCE_MOTION);
			haptic_controller->setDeviceControlGains(350.0, 15.0);
		}

		if (key_pressed.at(GLFW_KEY_P) && !key_was_pressed.at(GLFW_KEY_P)) {
			if (haptic_controller->getPlaneGuidanceEnabled()) {
				cout << "disabling plane guidance" << endl;
				haptic_controller->disablePlaneGuidance();
			} else {
				cout << "enabling plane guidance" << endl;
				haptic_controller->enablePlaneGuidance();
			}
		} else if (key_pressed.at(GLFW_KEY_L) &&
				   !key_was_pressed.at(GLFW_KEY_L)) {
			if (haptic_controller->getLineGuidanceEnabled()) {
				cout << "disabling line guidance" << endl;
				haptic_controller->disableLineGuidance();
			} else {
				cout << "enabling line guidance" << endl;
				haptic_controller->enableLineGuidance();
			}
		} else if (key_pressed.at(GLFW_KEY_O) &&
				   !key_was_pressed.at(GLFW_KEY_O)) {
			if (haptic_controller->getOrientationTeleopEnabled()) {
				cout << "disabling orientation teleoperation" << endl;
				haptic_controller->disableOrientationTeleop();
			} else {
				cout << "enabling orientation teleoperation" << endl;
				haptic_controller->enableOrientationTeleop();
			}
		}

		key_was_pressed = key_pressed;
	}

	redis_client.setEigen(createRedisKey(COMMANDED_FORCE_KEY_SUFFIX, 0),
						  Vector3d::Zero());
	redis_client.setEigen(createRedisKey(COMMANDED_TORQUE_KEY_SUFFIX, 0),
						  Vector3d::Zero());

	cout << "control timer stats:" << endl;
	controlTimer.printInfoPostRun();
}