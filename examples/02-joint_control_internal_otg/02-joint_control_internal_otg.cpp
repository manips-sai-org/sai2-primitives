/*
 * Example of joint space control on a simulated PUMA robot
 * Will move the elbow joint back and forth, and enable the joint interpolation after 5 seconds.
 * Assumes the robot is performing its own gravity compensation so the controller ignores gravity
 *
 */

// some standard library includes
#include <iostream>
#include <string>
#include <thread>
#include <math.h>

// sai2 main libraries includes
#include "Sai2Model.h"
#include "Sai2Graphics.h"
#include "Sai2Simulation.h"

// sai2 utilities from sai2-common
#include "timer/LoopTimer.h"

// control tasks from sai2-primitives
#include "tasks/JointTask.h"

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int){fSimulationRunning = false;}

// namespaces for compactness of code
using namespace std;
using namespace Eigen;

// config file names and object names
const string world_file = "resources/world.urdf";
const string robot_file = "resources/puma.urdf";
const string robot_name = "PUMA";   // name in the world file

// control and ui interaction torques
VectorXd UI_interaction_torques;
VectorXd control_torques;

// simulation and control loop thread functions
void control(std::shared_ptr<Sai2Model::Sai2Model> robot, std::shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, std::shared_ptr<Sai2Simulation::Sai2Simulation> sim);

/* 
 * Main function 
 * initializes everything, 
 * handles the visualization thread 
 * and starts the control and simulation threads 
 */
int main (int argc, char** argv) {
	cout << "Loading URDF world model file: " << world_file << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = std::make_shared<Sai2Graphics::Sai2Graphics>(world_file);
	graphics->addUIForceInteraction(robot_name);

	// load simulation world
	auto sim = std::make_shared<Sai2Simulation::Sai2Simulation>(world_file);

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
		UI_interaction_torques = graphics->getUITorques(robot_name);

	}

	// stop simulation and control
	fSimulationRunning = false;
	sim_thread.join();
	ctrl_thread.join();

	return 0;
}

//------------------ Controller main function
void control(std::shared_ptr<Sai2Model::Sai2Model> robot, std::shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	
	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof,dof);

	// prepare joint task
	auto joint_task = std::make_shared<Sai2Primitives::JointTask>(robot);
	// set the gains to get a PD controller with critical damping
	joint_task->setGains(100, 20);
	Eigen::VectorXd desired_position = joint_task->getDesiredPosition();
	// internal otg is on by default with acceleration limited trajectory
	// generation with max velocity PI/3 and max acceleration PI
	// joint_task->enableInternalOtgAccelerationLimited(PI/3, PI);

	// create a loop timer
	double control_freq = 1000;   // 1 KHz
	Sai2Common::LoopTimer timer(control_freq);
	timer.initializeTimer(1000000); // 1 ms pause before starting loop

	while (fSimulationRunning) { //automatically set to false when simulation is quit
		timer.waitForNextLoop();

		// read joint positions, velocities from simulation and update robot model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update task model
		N_prec.setIdentity(dof,dof);
		joint_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		// set the desired position (step every second)
		if(timer.elapsedCycles() % 4000 == 1000) {
			desired_position(1) -= 0.2;
			desired_position(2) += 0.4;
			desired_position(3) -= 0.6;
		}
		if(timer.elapsedCycles() % 4000 == 3000) {
			desired_position(1) += 0.2;
			desired_position(2) -= 0.4;
			desired_position(3) += 0.6;
		}
		joint_task->setDesiredPosition(desired_position);

		// increase velocity and acceleration limits after 5 seconds
		if(timer.elapsedCycles() == 5000) {
			std::cout << "------------------------------------" << std::endl;
			std::cout << "increasing velocity and acceleration limits" << std::endl;
			std::cout << "------------------------------------" << std::endl;
			joint_task->enableInternalOtgAccelerationLimited(M_PI, 3*M_PI);
		}
		// adding jerk limits after 10 seconds
		if(timer.elapsedCycles() == 10000) {
			std::cout << "------------------------------------" << std::endl;
			std::cout << "adding jerk limits" << std::endl;
			std::cout << "------------------------------------" << std::endl;
			joint_task->enableInternalOtgJerkLimited(M_PI, 3*M_PI, 3*M_PI);
		}

		VectorXd joint_task_torques = joint_task->computeTorques();

		// compute task torques
		control_torques = joint_task_torques;

		// -------------------------------------------
		// display robot state every half second
		if(timer.elapsedCycles() % 500 == 0)
		{
			cout << time << endl;
			cout << "desired position : " << joint_task->getDesiredPosition().transpose() << endl;
			cout << "current position : " << joint_task->getCurrentPosition().transpose() << endl;
			cout << "position error : " << (joint_task->getDesiredPosition() - joint_task->getCurrentPosition()).norm() << endl;
			cout << endl;
		}
	}

	// display controller run frequency at the end
	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Control Loop run time  : " << end_time << " seconds\n";
    std::cout << "Control Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Control Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";

}

//------------------------------------------------------------------------------
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, std::shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	fSimulationRunning = true;

	// create a timer
	double sim_freq = 2000;   // 2 kHz
	Sai2Common::LoopTimer timer(sim_freq);
	timer.initializeTimer();

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();
		sim->setJointTorques(robot_name, control_torques + UI_interaction_torques);
		sim->integrate();
	}

	// display simulation run frequency at the end
	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Simulation Loop run time  : " << end_time << " seconds\n";
    std::cout << "Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}