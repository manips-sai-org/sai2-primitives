/*
 * Example of a controller for a Panda arm (7DoF robot)
 * performing a surface-surface alignment (zero moment control)
 * after reaching contact
 */

// Initialization is the same as previous examples
#include <math.h>
#include <signal.h>

#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "Sai2Graphics.h"
#include "Sai2Model.h"
#include "Sai2Simulation.h"
#include "tasks/JointTask.h"
#include "tasks/MotionForceTask.h"
#include "timer/LoopTimer.h"
bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

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
const Vector3d pos_in_link = Vector3d(0.0, 0.0, 0.04);
const Vector3d sensor_pos_in_link = Vector3d(0.0, 0.0, 0.0);

// state machine for control
#define GO_TO_CONTACT 0
#define CONTACT_CONTROL 1

// simulation and control loop
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
				shared_ptr<Sai2Model::Sai2Model> plate,
				shared_ptr<Sai2Simulation::Sai2Simulation> sim);

//------------ main function
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
	sim->addSimulatedForceSensor(robot_name, link_name, T_sensor, 5.0);
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
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	// prepare state machine
	int state = GO_TO_CONTACT;

	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// Position plus orientation task
	auto motion_force_task = make_unique<Sai2Primitives::MotionForceTask>(
		robot, link_name, Affine3d(Translation3d(pos_in_link)), true);
	motion_force_task->enablePassivity();
	VectorXd motion_force_task_torques = VectorXd::Zero(dof);
	// set the force sensor location for the contact part of the task
	Affine3d T_control_sensor = Affine3d::Identity();
	T_control_sensor.translation() = sensor_pos_in_link - pos_in_link;
	motion_force_task->setForceSensorFrame(link_name, T_control_sensor);
	motion_force_task->disableInternalOtg();

	// no gains setting here, using the default task values
	const Matrix3d initial_orientation = robot->rotation(link_name);
	const Vector3d initial_position = robot->position(link_name, pos_in_link);
	Vector3d desired_position = initial_position;

	// joint task to control the redundancy
	auto joint_task = make_unique<Sai2Primitives::JointTask>(robot);
	VectorXd joint_task_torques = VectorXd::Zero(dof);

	VectorXd initial_q = robot->q();

	// create a loop timer
	double control_freq = 1000;
	Sai2Common::LoopTimer timer(control_freq);
	timer.initializeTimer(1e6);

	while (fSimulationRunning) {  // automatically set to false when simulation
								  // is quit
		timer.waitForNextLoop();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();
		motion_force_task->updateSensedForceAndMoment(sensed_force,
													  sensed_moment);

		// update tasks model. Order is important to define the hierarchy
		N_prec = MatrixXd::Identity(dof, dof);

		motion_force_task->updateTaskModel(N_prec);
		N_prec = motion_force_task->getN();
		// after each task, need to update the nullspace
		// of the previous tasks in order to garantee
		// the dyamic consistency

		joint_task->updateTaskModel(N_prec);

		// -------- set task goals in the state machine and compute control
		// torques
		if (state == GO_TO_CONTACT) {
			desired_position(2) -=
				0.00003;  // go down at 30 cm/s until contact is detected
			motion_force_task->setDesiredPosition(desired_position);

			if (motion_force_task->getSensedForce()(2) <= -5.0) {
				// switch the local z axis to be force controlled and the
				// local x and y axis to be moment controlled
				motion_force_task->parametrizeForceMotionSpaces(
					1, Vector3d::UnitZ());
				motion_force_task->parametrizeMomentRotMotionSpaces(
					2, Vector3d::UnitZ());

				motion_force_task->setClosedLoopForceControl();
				motion_force_task->setClosedLoopMomentControl();

				motion_force_task->setDesiredForce(10.0 * Vector3d::UnitZ());
				motion_force_task->setDesiredMoment(Vector3d::Zero());
				motion_force_task->setForceControlGains(0.3, 5.0, 1.0);
				motion_force_task->setMomentControlGains(0.3, 5.0, 1.0);

				// change the state of the state machine
				state = CONTACT_CONTROL;
			}
		} else if (state == CONTACT_CONTROL) {
			// nothing, the controller is already setup
		}

		// compute torques for the different tasks
		motion_force_task_torques = motion_force_task->computeTorques();
		joint_task_torques = joint_task->computeTorques();

		//------ compute the final torques
		command_torques = motion_force_task_torques + joint_task_torques;

		// send to simulation
		sim->setJointTorques(robot_name, command_torques);
	}
	timer.stop();
	cout << "Control loop timer stats:\n";
	timer.printInfoPostRun();
}

//------------------------------------------------------------------------------
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
				shared_ptr<Sai2Model::Sai2Model> plate,
				shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	fSimulationRunning = true;

	// plate controller
	Vector2d plate_qd = Vector2d::Zero();
	Vector2d plate_torques = Vector2d::Zero();

	// create a timer
	double sim_freq = 2000;
	Sai2Common::LoopTimer timer(sim_freq);
	timer.initializeTimer();

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();
		double time = timer.elapsedTime();

		// force sensor update
		sensed_force = sim->getSensedForce(robot_name, link_name);
		sensed_moment = sim->getSensedMoment(robot_name, link_name);

		// plate controller
		plate->setQ(sim->getJointPositions(plate_name));
		plate->setDq(sim->getJointVelocities(plate_name));
		plate->updateKinematics();

		plate_qd(0) = 5.0 / 180.0 * M_PI * sin(2 * M_PI * 0.12 * time);
		plate_qd(1) = 7.0 / 180.0 * M_PI * sin(2 * M_PI * 0.08 * time);

		plate_torques = -1000.0 * (plate->q() - plate_qd) - 75.0 * plate->dq();

		sim->setJointTorques(plate_name, plate_torques);

		// integrate forward
		sim->integrate();
	}
	timer.stop();
	cout << "Simulation loop timer stats:\n";
	timer.printInfoPostRun();
}