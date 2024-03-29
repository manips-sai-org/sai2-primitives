/*
 * Example of joint space control on a simulated PUMA robot
 * Will move the elbow joint back and forth, and enable the joint interpolation
 * after 5 seconds. Assumes the robot is performing its own gravity compensation
 * so the controller ignores gravity
 *
 */

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
#include "tasks/JointTask.h"

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

// namespaces for compactness of code
using namespace std;
using namespace Eigen;

// mutex to read and write the torques shared between the control thread and the
// simulation thread
mutex mutex_torques;

// config file names and object names
const string world_file = "${EXAMPLE_02_FOLDER}/world.urdf";
const string robot_file = "${SAI2_MODEL_URDF_FOLDER}/puma/puma.urdf";
const string robot_name = "PUMA";  // name in the world file

// control and ui interaction torques
VectorXd UI_interaction_torques;
VectorXd control_torques;

// simulation and control loop thread functions
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
				shared_ptr<Sai2Simulation::Sai2Simulation> sim);

/*
 * Main function
 * initializes everything,
 * handles the visualization thread
 * and starts the control and simulation threads
 */
int main(int argc, char** argv) {
	Sai2Model::URDF_FOLDERS["EXAMPLE_02_FOLDER"] =
		string(EXAMPLES_FOLDER) + "/02-joint_control_internal_otg";
	cout << "Loading URDF world model file: "
		 << Sai2Model::ReplaceUrdfPathPrefix(world_file) << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = make_shared<Sai2Graphics::Sai2Graphics>(world_file);
	graphics->addUIForceInteraction(robot_name);

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file, false);
	// update robot model from simulation configuration
	robot->setQ(sim->getJointPositions(robot_name));
	robot->updateModel();

	// initial values for torques
	UI_interaction_torques.setZero(robot->dof());
	control_torques.setZero(robot->dof());

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
			UI_interaction_torques = graphics->getUITorques(robot_name);
		}
	}

	// stop simulation and control
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
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// prepare joint task
	auto joint_task = make_shared<Sai2Primitives::JointTask>(robot);
	// set the gains to get a PD controller with critical damping
	joint_task->setGains(100, 20);
	Eigen::VectorXd goal_position = joint_task->getGoalPosition();
	// internal otg is on with acceleration limited trajectory
	// generation with max velocity PI/3 and max acceleration PI
	joint_task->enableInternalOtgAccelerationLimited(M_PI / 3, M_PI);

	// create a loop timer
	double control_freq = 1000;	 // 1 KHz
	Sai2Common::LoopTimer timer(control_freq, 1e6);

	while (fSimulationRunning) {  // automatically set to false when simulation
								  // is quit
		timer.waitForNextLoop();

		// read joint positions, velocities from simulation and update robot
		// model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update task model
		N_prec.setIdentity(dof, dof);
		joint_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		// set the goal position (step every second)
		if (timer.elapsedCycles() % 4000 == 1000) {
			goal_position(1) -= 0.2;
			goal_position(2) += 0.4;
			goal_position(3) -= 0.6;
		}
		if (timer.elapsedCycles() % 4000 == 3000) {
			goal_position(1) += 0.2;
			goal_position(2) -= 0.4;
			goal_position(3) += 0.6;
		}
		joint_task->setGoalPosition(goal_position);

		// increase velocity and acceleration limits after 5 seconds
		if (timer.elapsedCycles() == 5000) {
			cout << "------------------------------------" << endl;
			cout << "increasing velocity and acceleration limits" << endl;
			cout << "------------------------------------" << endl;
			joint_task->enableInternalOtgAccelerationLimited(M_PI, 3 * M_PI);
		}
		// adding jerk limits after 10 seconds
		if (timer.elapsedCycles() == 10000) {
			cout << "------------------------------------" << endl;
			cout << "adding jerk limits" << endl;
			cout << "------------------------------------" << endl;
			joint_task->enableInternalOtgJerkLimited(M_PI, 3 * M_PI, 3 * M_PI);
		}

		VectorXd joint_task_torques = joint_task->computeTorques();

		// compute task torques
		{
			lock_guard<mutex> guard(mutex_torques);
			control_torques = joint_task_torques;
		}

		// -------------------------------------------
		// display robot state every half second
		if (timer.elapsedCycles() % 500 == 0) {
			cout << timer.elapsedSimTime() << endl;
			cout << "goal position : "
				 << joint_task->getGoalPosition().transpose() << endl;
			cout << "current position : "
				 << joint_task->getCurrentPosition().transpose() << endl;
			cout << "position error : "
				 << (joint_task->getGoalPosition() -
					 joint_task->getCurrentPosition())
						.norm()
				 << endl;
			cout << endl;
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
	double sim_freq = 2000;	 // 2 kHz
	Sai2Common::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();
		{
			lock_guard<mutex> guard(mutex_torques);
			sim->setJointTorques(robot_name,
								 control_torques + UI_interaction_torques);
		}
		sim->integrate();
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
}