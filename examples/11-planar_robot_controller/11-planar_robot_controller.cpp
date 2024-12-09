// some standard library includes
#include <math.h>

#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// sai main libraries includes
#include "SaiGraphics.h"
#include "SaiModel.h"
#include "SaiSimulation.h"

// sai utilities from sai-common
#include "timer/LoopTimer.h"

// control tasks from sai-primitives
#include "RobotController.h"
#include "tasks/MotionForceTask.h"

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

// namespaces for compactness of code
using namespace std;
using namespace Eigen;

// config file names and object names
const string world_file = "${EXAMPLE_11_FOLDER}/world.urdf";
const string robot_file = "${EXAMPLE_11_FOLDER}/rrrrbot.urdf";
const string robot_name = "RRRRBOT";

// simulation and control loop
void control(shared_ptr<SaiModel::SaiModel> robot,
			 shared_ptr<SaiSimulation::SaiSimulation> sim);
void simulation(shared_ptr<SaiModel::SaiModel> robot,
				shared_ptr<SaiSimulation::SaiSimulation> sim);

VectorXd control_torques, ui_torques;

// mutex to read and write the control torques
mutex mutex_torques;

int main(int argc, char** argv) {
	SaiModel::URDF_FOLDERS["EXAMPLE_11_FOLDER"] =
		string(EXAMPLES_FOLDER) + "/11-planar_robot_controller";
	cout << "Loading URDF world model file: "
		 << SaiModel::ReplaceUrdfPathPrefix(world_file) << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = make_shared<SaiGraphics::SaiGraphics>(world_file);

	// load simulation world
	auto sim = make_shared<SaiSimulation::SaiSimulation>(world_file);

	// load robots
	auto robot = make_shared<SaiModel::SaiModel>(robot_file);
	// update robot model from simulation configuration
	robot->setQ(sim->getJointPositions(robot_name));
	robot->updateModel();
	control_torques.setZero(robot->dof());

	ui_torques.setZero(robot->dof());
	graphics->addUIForceInteraction(robot_name);

	// start the simulation thread first
	fSimulationRunning = true;
	thread sim_thread(simulation, robot, sim);

	// next start the control thread
	thread ctrl_thread(control, robot, sim);

	// while window is open:
	while (graphics->isWindowOpen()) {
		graphics->updateRobotGraphics(robot_name, robot->q());
		graphics->renderGraphicsWorld();
		{
			lock_guard<mutex> guard(mutex_torques);
			ui_torques = graphics->getUITorques(robot_name);
		}
	}

	// stop simulation
	fSimulationRunning = false;
	sim_thread.join();
	ctrl_thread.join();

	return 0;
}

//------------------ Controller main function
void control(shared_ptr<SaiModel::SaiModel> robot,
			 shared_ptr<SaiSimulation::SaiSimulation> sim) {
	robot->updateModel();
	int dof = robot->dof();

	// prepare the task to control y-z position and rotation around z
	string link_name = "link4";
	Affine3d compliant_frame = Affine3d::Identity();
	compliant_frame.translation() = Vector3d(0.5, 0.0, 0.0);
	vector<Vector3d> controlled_directions_translation;
	controlled_directions_translation.push_back(Vector3d::UnitX());
	controlled_directions_translation.push_back(Vector3d::UnitY());
	vector<Vector3d> controlled_directions_rotation;
	controlled_directions_rotation.push_back(Vector3d::UnitZ());
	auto motion_force_task = make_shared<SaiPrimitives::MotionForceTask>(
		robot, link_name, controlled_directions_translation,
		controlled_directions_rotation, compliant_frame);

	// initial position and orientation
	const Matrix3d initial_orientation =
		motion_force_task->getCurrentOrientation();
	const Vector3d initial_position = motion_force_task->getCurrentPosition();
	Vector3d goal_position = initial_position;
	Matrix3d goal_orientation = initial_orientation;

	// robot controller with the motion force task
	auto joint_task = make_shared<SaiPrimitives::JointTask>(robot);
	vector<shared_ptr<SaiPrimitives::TemplateTask>> task_list = {
		motion_force_task, joint_task};
	auto robot_controller =
		make_unique<SaiPrimitives::RobotController>(robot, task_list);

	// create a loop timer
	double control_freq = 1000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update tasks model
		robot_controller->updateControllerTaskModels();

		// -------- set task goals and compute control torques
		if (timer.elapsedCycles() % 4000 == 0) {
			goal_position = initial_position;
			goal_orientation = initial_orientation;
		} else if (timer.elapsedCycles() % 4000 == 2000) {
			goal_position = initial_position - Vector3d(0.25, 0.25, 0.0);
			goal_orientation =
				AngleAxisd(-M_PI / 4, Vector3d::UnitZ()) * initial_orientation;
		}

		motion_force_task->setGoalPosition(goal_position);
		motion_force_task->setGoalOrientation(goal_orientation);

		//------ Control torques
		{
			lock_guard<mutex> guard(mutex_torques);
			control_torques = robot_controller->computeControlTorques();
		}
	}
	timer.stop();
	cout << "\nControl loop timer stats:\n";
	timer.printInfoPostRun();
}

//------------------------------------------------------------------------------
void simulation(shared_ptr<SaiModel::SaiModel> robot,
				shared_ptr<SaiSimulation::SaiSimulation> sim) {
	fSimulationRunning = true;

	// create a timer
	double sim_freq = 2000;
	SaiCommon::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		{
			lock_guard<mutex> guard(mutex_torques);
			sim->setJointTorques(robot_name, control_torques + ui_torques);
		}
		sim->integrate();
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
}