/*
 * Example of a controller for a Panda arm (7DoF robot) 
 * performing a surface-surface alignment (zero moment control) 
 * after reaching contact
 */

// Initialization is the same as previous examples
#include <iostream>
#include <string>
#include <thread>
#include <math.h>
#include <mutex>

#include "Sai2Model.h"
#include "Sai2Graphics.h"
#include "Sai2Simulation.h"
#include "tasks/MotionForceTask.h"
#include "tasks/JointTask.h"
#include "timer/LoopTimer.h"

#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int){fSimulationRunning = false;}

using namespace std;
using namespace Eigen;

const string world_file = "resources/world.urdf";
const string robot_file = "resources/panda_arm.urdf";
const string robot_name = "PANDA";
// need a second robot model for the plate
const string plate_file = "resources/plate.urdf";
const string plate_name = "Plate";

// global variables for sensed force and moment
Vector3d sensed_force;
Vector3d sensed_moment;

// global variables for controller parametrization
const string link_name = "end-effector";
const Vector3d pos_in_link = Vector3d(0.0,0.0,0.04);
const Vector3d sensor_pos_in_link = Vector3d(0.0,0.0,0.0);

// state machine for control
#define GO_TO_CONTACT    0
#define CONTACT_CONTROL  1

// simulation and control loop
void control(shared_ptr<Sai2Model::Sai2Model> robot, Sai2Simulation::Sai2Simulation* sim);
void simulation(shared_ptr<Sai2Model::Sai2Model> robot, shared_ptr<Sai2Model::Sai2Model> plate, Sai2Simulation::Sai2Simulation* sim);

//------------ main function
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
	sim->setCoeffFrictionStatic(0.3);
	sim->setCollisionRestitution(0);

	// load robot and plate
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file);
	robot->setQ(sim->getJointPositions(robot_name));
	robot->updateModel();

	// load plate
	auto plate = make_shared<Sai2Model::Sai2Model>(plate_file);

	// create simulated force sensor
	Affine3d T_sensor = Affine3d::Identity();
	T_sensor.translation() = sensor_pos_in_link;
	sim->addSimulatedForceSensor(robot_name, link_name, T_sensor);
	graphics->addForceSensorDisplay(sim->getAllForceSensorData()[0]);
	// fsensor->enableFilter(0.005);

	// start the simulation thread first
	fSimulationRunning = true;
	thread sim_thread(simulation, robot, plate, sim);

	// next start the control thread
	thread ctrl_thread(control, robot, sim);
	
    // while window is open:
    while (graphics->isWindowOpen()) {
		graphics->updateRobotGraphics(robot_name, robot->q());
		graphics->updateRobotGraphics(plate_name, plate->q());
		graphics->updateDisplayedForceSensor(sim->getAllForceSensorData()[0]);
		graphics->renderGraphicsWorld();
	}

	// stop simulation
	fSimulationRunning = false;
	sim_thread.join();
	ctrl_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void control(shared_ptr<Sai2Model::Sai2Model> robot, Sai2Simulation::Sai2Simulation* sim) {

	// prepare state machine
	int state = GO_TO_CONTACT;
	
	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof,dof);

	// Position plus orientation task

	Sai2Primitives::MotionForceTask* motion_force_task =
		new Sai2Primitives::MotionForceTask(
			robot, link_name, Affine3d(Translation3d(pos_in_link)), true);
	motion_force_task->enablePassivity();
	VectorXd motion_force_task_torques = VectorXd::Zero(dof);
	// set the force sensor location for the contact part of the task
	motion_force_task->setForceSensorFrame(link_name, Affine3d::Identity());

#ifdef USING_OTG
	// disable the interpolation
	motion_force_task->_use_interpolation_flag = false;
#endif
	// no gains setting here, using the default task values
	const Matrix3d initial_orientation = robot->rotation(link_name);
	const Vector3d initial_position = robot->position(link_name, pos_in_link);
	Vector3d desired_position = initial_position;

	// joint task to control the redundancy
	Sai2Primitives::JointTask* joint_task = new Sai2Primitives::JointTask(robot);
	VectorXd joint_task_torques = VectorXd::Zero(dof);

	VectorXd initial_q = robot->q();

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

		// update time
		double curr_time = timer.elapsedTime();
		double loop_dt = curr_time - last_time;

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update force sensor values (needs to be the force applied by the robot to the environment, in sensor frame)
		motion_force_task->updateSensedForceAndMoment(-sensed_force, -sensed_moment);

		// update tasks model. Order is important to define the hierarchy
		N_prec = MatrixXd::Identity(dof,dof);

		motion_force_task->updateTaskModel(N_prec);
		N_prec = motion_force_task->getN();    
		// after each task, need to update the nullspace 
		// of the previous tasks in order to garantee 
		// the dyamic consistency

		joint_task->updateTaskModel(N_prec);

		// -------- set task goals in the state machine and compute control torques
		if(state == GO_TO_CONTACT) {
			desired_position(2) -= 0.00003; // go down at 30 cm/s until contact is detected
			motion_force_task->setDesiredPosition(desired_position);  
		
			if(motion_force_task->getSensedForce()(2) <= -1.0) {
				// switch the local z axis to be force controlled and the local x and y axis to be moment controlled
				motion_force_task->parametrizeForceMotionSpaces(1, Vector3d::UnitZ());
				motion_force_task->parametrizeMomentRotMotionSpaces(2, Vector3d::UnitZ());

				motion_force_task->setClosedLoopForceControl();
				motion_force_task->setClosedLoopMomentControl();

				// set the force and moment control set points and gains gains
				Vector3d local_z = motion_force_task->getCurrentOrientation().col(2);
				motion_force_task->setDesiredForce(5.0*local_z);
				motion_force_task->setDesiredMoment(Vector3d::Zero());

				motion_force_task->setForceControlGains(0.7, 3.0, 1.3);
				motion_force_task->setMomentControlGains(0.5, 1.0, 1.5);

				// change the state of the state machine
				state = CONTACT_CONTROL;
			}	
		}
		else if(state == CONTACT_CONTROL) {
				Vector3d local_z = motion_force_task->getCurrentOrientation().col(2);
				motion_force_task->setDesiredForce(10.0*local_z);
		}

		// compute torques for the different tasks
		motion_force_task_torques = motion_force_task->computeTorques();
		joint_task_torques = joint_task->computeTorques();

		//------ compute the final torques
		command_torques = motion_force_task_torques + joint_task_torques;

		// send to simulation
		sim->setJointTorques(robot_name, command_torques);

		controller_counter++;

		// -------------------------------------------
		// update last time
		last_time = curr_time;
	}

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Control Loop run time  : " << end_time << " seconds\n";
    std::cout << "Control Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Control Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";

}

//------------------------------------------------------------------------------
void simulation(shared_ptr<Sai2Model::Sai2Model> robot, shared_ptr<Sai2Model::Sai2Model> plate, Sai2Simulation::Sai2Simulation* sim)
{
	fSimulationRunning = true;

	// plate controller
	Vector2d plate_qd = Vector2d::Zero();
	Vector2d plate_torques = Vector2d::Zero();

	// create a timer
	double sim_freq = 2000;
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(sim_freq); 
	double last_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		fTimerDidSleep = timer.waitForNextLoop();
		double time = timer.elapsedTime();

		// force sensor update
		sensed_force = sim->getSensedForce(robot_name, link_name);
		sensed_moment = sim->getSensedMoment(robot_name, link_name);

		// plate controller
		plate->setQ(sim->getJointPositions(plate_name));
		plate->setDq(sim->getJointVelocities(plate_name));
		plate->updateKinematics();

		plate_qd(0) = 5.0/180.0*M_PI*sin(2*M_PI*0.12*time);
		plate_qd(1) = 7.0/180.0*M_PI*sin(2*M_PI*0.08*time);

		plate_torques = -1000.0*(plate->q() - plate_qd) - 75.0*plate->dq();

		sim->setJointTorques(plate_name, plate_torques);

		// integrate forward
		sim->integrate();

		// if(timer.elapsedCycles() % 8000 == 0) {
		// 	sim->showContactInfo();
		// 	cout << endl << endl;
		// }
		// auto contacts = sim->getContactList(robot_name, "end-effector");
		// cout << contacts.size() << endl;

	}

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Simulation Loop run time  : " << end_time << " seconds\n";
    std::cout << "Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}