/*
 * Example of a controller for a Puma arm made with a 6DoF position and
 * orientation task at the end effector Here, the position and orientation tasks
 * are dynamically decoupled with the bounded inertia estimates method.
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
#include "tasks/MotionForceTask.h"

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

// namespaces for compactness of code
using namespace std;
using namespace Eigen;

// config file names and object names
const string world_file = "resources/world.urdf";
const string robot_file = "resources/puma.urdf";
const string robot_name = "PUMA";  // name in the workd file

// simulation and control loop
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
				shared_ptr<Sai2Simulation::Sai2Simulation> sim);

VectorXd control_torques, ui_torques;

// mutex to read and write the control torques
mutex mmutex_torques;

/*
 * Main function
 * initializes everything,
 * handles the visualization thread
 * and starts the control and simulation threads
 */
int main(int argc, char** argv) {
	cout << "Loading URDF world model file: " << world_file << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = make_shared<Sai2Graphics::Sai2Graphics>(world_file);

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file);
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
			lock_guard<mutex> guard(mmutex_torques);
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
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// prepare the task
	string link_name =
		"end-effector";	 // link where we attach the control frame
	Vector3d pos_in_link =
		Vector3d(0.07, 0.0, 0.0);  // position of control frame in link
	Affine3d compliant_frame_in_link =
		Affine3d(Translation3d(pos_in_link));  // control frame in link
	Sai2Primitives::MotionForceTask* motion_force_task =
		new Sai2Primitives::MotionForceTask(robot, link_name,
											compliant_frame_in_link);

	// gains for the position and orientation parts of the controller
	motion_force_task->setPosControlGains(100.0, 20.0);
	motion_force_task->setOriControlGains(100.0, 20.0);

	// initial position and orientation
	Matrix3d initial_orientation = robot->rotation(link_name);
	Vector3d initial_position = robot->position(link_name, pos_in_link);
	Vector3d goal_position = initial_position;
	Matrix3d goal_orientation = initial_orientation;

	// create a loop timer
	double control_freq = 1000;
	Sai2Common::LoopTimer timer(control_freq, 1e6);

	while (fSimulationRunning) {  // automatically set to false when simulation
								  // is quit
		timer.waitForNextLoop();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update tasks model
		N_prec = MatrixXd::Identity(dof, dof);
		motion_force_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		double time = timer.elapsedSimTime();

		Matrix3d R;
		double theta = M_PI / 4.0;
		R << cos(theta), sin(theta), 0, -sin(theta), cos(theta), 0, 0, 0, 1;
		if (timer.elapsedCycles() % 3000 == 2000) {
			goal_position(2) += 0.1;
			goal_orientation = R * goal_orientation;

		} else if (timer.elapsedCycles() % 3000 == 500) {
			goal_position(2) -= 0.1;
			goal_orientation = R.transpose() * goal_orientation;
		}
		motion_force_task->setGoalPosition(goal_position);
		motion_force_task->setGoalOrientation(goal_orientation);

		// disable otg after 6.5 seconds
		if (timer.elapsedCycles() == 6500) {
			motion_force_task->disableInternalOtg();
		}

		// enable interpolation with jerk limits after 12.5 seconds
		if (timer.elapsedCycles() == 12500) {
			motion_force_task->enableInternalOtgJerkLimited(
				0.3, 1.0, 3.0, M_PI / 3, M_PI, 3 * M_PI);
		}

		VectorXd motion_force_task_torques =
			motion_force_task->computeTorques();

		//------ Control torques
		{
			lock_guard<mutex> guard(mmutex_torques);
			control_torques = motion_force_task_torques;
		}

		// -------------------------------------------
		if (timer.elapsedCycles() % 500 == 0) {
			cout << time << endl;
			cout << "goal position : "
				 << motion_force_task->getGoalPosition().transpose() << endl;
			cout << "current position : "
				 << motion_force_task->getCurrentPosition().transpose() << endl;
			cout << "position error : "
				 << (motion_force_task->getGoalPosition() -
					 motion_force_task->getCurrentPosition())
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
			lock_guard<mutex> guard(mmutex_torques);
			sim->setJointTorques(robot_name, control_torques + ui_torques);
		}
		sim->integrate();
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
}