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
#include <mutex>

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

// simulation and control loop thread functions
void control(Sai2Model::Sai2Model* robot, Sai2Simulation::Sai2Simulation* sim);
void simulation(Sai2Model::Sai2Model* robot, Sai2Simulation::Sai2Simulation* sim);

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
	auto graphics = new Sai2Graphics::Sai2Graphics(world_file);

	// load simulation world
	auto sim = new Sai2Simulation::Sai2Simulation(world_file);

	// load robots
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	// update robot model from simulation configuration
	robot->setQ(sim->getJointPositions(robot_name));
	robot->updateModel();

	// start the simulation thread first
	fSimulationRunning = true;
	thread sim_thread(simulation, robot, sim);
	// next start the control thread
	thread ctrl_thread(control, robot, sim);
	
    // while window is open:
    while (graphics->isWindowOpen()) {

		graphics->updateRobotGraphics(robot_name, robot->q());
		graphics->renderGraphicsWorld();

	}

	// stop simulation and control
	fSimulationRunning = false;
	sim_thread.join();
	ctrl_thread.join();

	return 0;
}

//------------------ Controller main function
void control(Sai2Model::Sai2Model* robot, Sai2Simulation::Sai2Simulation* sim) {
	
	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof,dof);
	VectorXd command_torques = VectorXd::Zero(dof);

	// prepare joint task
	Sai2Primitives::JointTask* joint_task = new Sai2Primitives::JointTask(robot);
	VectorXd joint_task_torques = VectorXd::Zero(dof);
	// set the gains to get a PD controller with critical damping
	joint_task->setGains(100, 20);
	Eigen::VectorXd desired_position = joint_task->getDesiredPosition();

	#ifdef USING_OTG
		// disable the interpolation for the first phase
		joint_task->_use_interpolation_flag = false;
	#endif

	// create a loop timer
	double control_freq = 1000;   // 1 KHz
	LoopTimer timer;
	timer.setLoopFrequency(control_freq);
	double last_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;
	timer.initializeTimer(1000000); // 1 ms pause before starting loop

	unsigned long long controller_counter = 0;

	while (fSimulationRunning) { //automatically set to false when simulation is quit
		fTimerDidSleep = timer.waitForNextLoop();

		// update time
		double curr_time = timer.elapsedTime();
		double loop_dt = curr_time - last_time;

		// read joint positions, velocities from simulation and update robot model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update task model
		N_prec.setIdentity(dof,dof);
		joint_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		// set the desired position (step every second)
		if(controller_counter % 3000 == 500) {
			desired_position(2) += 0.4;
		}
		if(controller_counter % 3000 == 2000) {
			desired_position(2) -= 0.4;
		}
		joint_task->setDesiredPosition(desired_position);
		// enable interpolation after 6 seconds and set interpolation limits
		#ifdef USING_OTG
			if(controller_counter == 6500) {
				joint_task->_use_interpolation_flag = true;
				joint_task->_otg->setMaxVelocity(M_PI/6);
				joint_task->_otg->setMaxAcceleration(M_PI);
				joint_task->_otg->setMaxJerk(4*M_PI);
			}
		#endif
		// compute task torques
		joint_task_torques = joint_task->computeTorques();

		//------ Final torques
		command_torques = joint_task_torques;
		// command_torques.setZero();

		// -------------------------------------------
		sim->setJointTorques(robot_name, command_torques);

		// cout << robot->_M << endl << endl << endl;

		// -------------------------------------------
		// display robot state every half second
		if(controller_counter % 500 == 0)
		{
			cout << time << endl;
			cout << "desired position : " << joint_task->getDesiredPosition().transpose() << endl;
			cout << "current position : " << joint_task->getCurrentPosition().transpose() << endl;
			cout << "position error : " << (joint_task->getDesiredPosition() - joint_task->getCurrentPosition()).norm() << endl;
			cout << endl;
		}

		controller_counter++;

		// -------------------------------------------
		// update last time
		last_time = curr_time;
	}

	// display controller run frequency at the end
	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Control Loop run time  : " << end_time << " seconds\n";
    std::cout << "Control Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Control Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";

}

//------------------------------------------------------------------------------
void simulation(Sai2Model::Sai2Model* robot, Sai2Simulation::Sai2Simulation* sim) {
	fSimulationRunning = true;

	// create a timer
	double sim_freq = 2000;   // 2 kHz
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(sim_freq); 
	double last_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		fTimerDidSleep = timer.waitForNextLoop();

		sim->integrate();

	}

	// display simulation run frequency at the end
	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Simulation Loop run time  : " << end_time << " seconds\n";
    std::cout << "Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}