/*
 * Example of a controller for a Puma arm made with a 6DoF position and
 * orientation task at the end effector Here, the position and orientation tasks
 * are dynamically decoupled with the bounded inertia estimates method.
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
#include "tasks/MotionForceTask.h"

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
const string robot_name = "PUMA"; // name in the workd file

// simulation and control loop
void control(Sai2Model::Sai2Model* robot, Sai2Simulation::Sai2Simulation* sim);
void simulation(Sai2Model::Sai2Model* robot, Sai2Simulation::Sai2Simulation* sim);

Eigen::VectorXd control_torques, ui_torques;

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
	auto robot = new Sai2Model::Sai2Model(robot_file);
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
		ui_torques = graphics->getUITorques(robot_name);
	}

	// stop simulation
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

	// prepare the task
	string link_name = "end-effector";             // link where we attach the control frame
	Vector3d pos_in_link = Vector3d(0.07,0.0,0.0); // location of the control frame in the link
	Matrix3d rot_in_link = Matrix3d::Identity();   // orientation of the control frame with respect to the link frame
	Sai2Primitives::MotionForceTask* motion_force_task = new Sai2Primitives::MotionForceTask(robot, link_name, pos_in_link, rot_in_link);
	VectorXd motion_force_task_torques = VectorXd::Zero(dof);

#ifdef USING_OTG
	// disable the interpolation for the first phase
	motion_force_task->_use_interpolation_flag = false;
#endif

	// gains for the position controller
	motion_force_task->setPosControlGains(100.0, 20.0);
	// gains for the orientation cotroller
	motion_force_task->setOriControlGains(100.0, 20.0);

	// initial position and orientation
	Matrix3d initial_orientation = robot->rotation(link_name);
	Vector3d initial_position = robot->position(link_name, pos_in_link);
	Vector3d desired_position = initial_position;
	Matrix3d desired_orientation = initial_orientation;

	// create a loop timer
	double control_freq = 1000;
	LoopTimer timer;
	timer.setLoopFrequency(control_freq);   // 1 KHz
	double last_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;
	timer.initializeTimer(1000000); // 1 ms pause before starting loop

	unsigned long long controller_counter = 0;

	while (fSimulationRunning) { //automatically set to false when simulation is quit
		fTimerDidSleep = timer.waitForNextLoop();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update tasks model
		N_prec = MatrixXd::Identity(dof,dof);
		motion_force_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		double time = controller_counter/control_freq;

		Matrix3d R;
		double theta = M_PI/4.0;
		R << cos(theta) , sin(theta) , 0,
        	-sin(theta) , cos(theta) , 0,
	    	     0      ,      0     , 1;
		if(controller_counter % 3000 == 2000)
		{
			desired_position(2) += 0.1;
			desired_orientation = R*desired_orientation;

		}
		else if(controller_counter % 3000 == 500)
		{
			desired_position(2) -= 0.1;
			desired_orientation = R.transpose()*desired_orientation;
		}
		motion_force_task->setDesiredPosition(desired_position);
		motion_force_task->setDesiredOrientation(desired_orientation);
		// enable interpolation after 6 seconds and set interpolation limits
		#ifdef USING_OTG
			if(controller_counter == 6500) {
				motion_force_task->_use_interpolation_flag = true;

				motion_force_task->_otg->setMaxLinearVelocity(0.3);
				motion_force_task->_otg->setMaxLinearAcceleration(1.0);
				motion_force_task->_otg->setMaxLinearJerk(3.0);

				motion_force_task->_otg->setMaxAngularVelocity(M_PI/3);
				motion_force_task->_otg->setMaxAngularAcceleration(M_PI);
				motion_force_task->_otg->setMaxAngularJerk(3*M_PI);
			}
		#endif

		// compute task torques
		motion_force_task_torques = motion_force_task->computeTorques();
		
		//------ Final torques
		control_torques = motion_force_task_torques;

		// -------------------------------------------
		if (controller_counter % 500 == 0) {
				cout << time << endl;
				cout << "desired position : "
					 << motion_force_task->getDesiredPosition().transpose() << endl;
				cout << "current position : "
					 << motion_force_task->getCurrentPosition().transpose() << endl;
				cout << "position error : "
					 << (motion_force_task->getDesiredPosition() -
						 motion_force_task->getCurrentPosition())
							.norm()
					 << endl;
				cout << endl;
		}

		controller_counter++;
	}

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

	sim->setTimestep(1.0/sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		sim->setJointTorques(robot_name, control_torques + ui_torques);
		sim->integrate();

	}

	// display simulation run frequency at the end
	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Sai2Simulation Loop run time  : " << end_time << " seconds\n";
    std::cout << "Sai2Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Sai2Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}