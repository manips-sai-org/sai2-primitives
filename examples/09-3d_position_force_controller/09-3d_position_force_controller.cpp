// some standard library includes
#include <math.h>

#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// sai2 main libraries includes
#include "Sai2Graphics.h"
#include "Sai2Model.h"
#include "Sai2Simulation.h"

// sai2 utilities from sai2-common
#include "timer/LoopTimer.h"

// control tasks from sai2-primitives
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
const string world_file = "${EXAMPLE_09_FOLDER}/world.urdf";
const string robot_file =
	"${SAI2_MODEL_URDF_FOLDER}/panda/panda_arm_sphere.urdf";
const string robot_name = "PANDA";

// simulation and control loop
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
				shared_ptr<Sai2Simulation::Sai2Simulation> sim);

VectorXd control_torques, ui_torques;
Vector3d sensed_force;
string link_name = "end-effector";

// mutex to read and write the control torques
mutex mutex_torques;

int main(int argc, char** argv) {
	Sai2Model::URDF_FOLDERS["EXAMPLE_09_FOLDER"] =
		string(EXAMPLES_FOLDER) + "/09-3d_position_force_controller";
	cout << "Loading URDF world model file: "
		 << Sai2Model::ReplaceUrdfPathPrefix(world_file) << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = make_shared<Sai2Graphics::Sai2Graphics>(world_file);

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);
	sim->setCoeffFrictionStatic(0.0);
	sim->addSimulatedForceSensor(robot_name, link_name, Affine3d::Identity(),
								 15.0);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file);
	// update robot model from simulation configuration
	robot->setQ(sim->getJointPositions(robot_name));
	robot->updateModel();
	control_torques.setZero(robot->dof());

	ui_torques.setZero(robot->dof());
	graphics->addUIForceInteraction(robot_name);
	graphics->addForceSensorDisplay(sim->getAllForceSensorData()[0]);

	// start the simulation thread first
	fSimulationRunning = true;
	thread sim_thread(simulation, robot, sim);

	// next start the control thread
	thread ctrl_thread(control, robot, sim);

	// while window is open:
	while (graphics->isWindowOpen()) {
		graphics->updateRobotGraphics(robot_name, robot->q());
		graphics->updateDisplayedForceSensor(sim->getAllForceSensorData()[0]);
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
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();

	// prepare the task to control y-z position and rotation around z
	vector<Vector3d> controlled_directions_translation;
	controlled_directions_translation.push_back(Vector3d::UnitX());
	controlled_directions_translation.push_back(Vector3d::UnitY());
	controlled_directions_translation.push_back(Vector3d::UnitZ());
	vector<Vector3d> controlled_directions_rotation;
	auto motion_force_task = make_shared<Sai2Primitives::MotionForceTask>(
		robot, link_name, controlled_directions_translation,
		controlled_directions_rotation);
	bool force_control = false;

	// initial position and orientation
	const Vector3d initial_position = robot->position(link_name);
	Vector3d goal_position = initial_position;

	// joint task to control the nullspace
	auto joint_task = make_shared<Sai2Primitives::JointTask>(robot);

	// controller
	vector<shared_ptr<Sai2Primitives::TemplateTask>> task_list = {
		motion_force_task, joint_task};
	auto robot_controller =
		make_unique<Sai2Primitives::RobotController>(robot, task_list);

	// create a loop timer
	double control_freq = 1000;
	Sai2Common::LoopTimer timer(control_freq, 1e6);

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update tasks model
		robot_controller->updateControllerTaskModels();

		// update sensed force
		{
			lock_guard<mutex> guard(mutex_torques);
			motion_force_task->updateSensedForceAndMoment(sensed_force,
														  Vector3d::Zero());
		}

		// -------- set task goals and compute control torques
		double time = timer.elapsedSimTime();

		// move in x and y plane back and forth
		if (timer.elapsedCycles() % 2000 == 0) {
			goal_position(0) -= 0.07;
			goal_position(1) -= 0.07;
		} else if (timer.elapsedCycles() % 2000 == 1000) {
			goal_position(0) += 0.07;
			goal_position(1) += 0.07;
		}
		if (timer.elapsedCycles() > 2000 && timer.elapsedCycles() < 3000) {
			goal_position(2) -= 0.00015;
		}
		motion_force_task->setGoalPosition(goal_position);

		if (!force_control &&
			motion_force_task->getSensedForceControlWorldFrame()(2) <= -1.0) {
			force_control = true;
			motion_force_task->parametrizeForceMotionSpaces(1,
															Vector3d::UnitZ());
			motion_force_task->setGoalForce(Vector3d(0, 0, -5.0));
			motion_force_task->setClosedLoopForceControl();
			motion_force_task->enablePassivity();
		}

		if (timer.elapsedCycles() % 1000 == 999) {
			cout << "goal position: "
				 << motion_force_task->getGoalPosition().transpose() << endl;
			cout << "current position: "
				 << motion_force_task->getCurrentPosition().transpose() << endl;
			cout << "goal force: "
				 << motion_force_task->getGoalForce().transpose() << endl;
			cout << "sensed force: "
				 << motion_force_task->getSensedForceControlWorldFrame()
						.transpose()
				 << endl;
			cout << endl;
		}

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
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
				shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	fSimulationRunning = true;

	// create a timer
	double sim_freq = 2000;
	Sai2Common::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		{
			lock_guard<mutex> guard(mutex_torques);
			sim->setJointTorques(robot_name, control_torques + ui_torques);
			sensed_force = sim->getSensedForce(robot_name, link_name);
		}
		sim->integrate();
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
}