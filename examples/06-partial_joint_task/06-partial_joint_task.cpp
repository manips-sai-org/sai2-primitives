/*
 * Example of a controller for a Panda arm (7DoF robot) on a 1dof sliding base,
 * where the sliding and elbow angle are controlled by a partial joint task, and
 * the position and orientation of the end effector are controlled by a
 * MotionForceTask
 */

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
const string robot_file = "resources/panda_arm_sliding_base.urdf";
const string robot_name = "PANDA";

// ui and control torques
VectorXd UI_torques;
VectorXd control_torques;

// mutex to read and write the control torques
mutex mutex_torques;

// simulation and control loop
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
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
	graphics->addUIForceInteraction(robot_name);

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file, false);
	robot->setQ(sim->getJointPositions(robot_name));
	robot->updateModel();

	UI_torques = VectorXd::Zero(robot->dof());
	control_torques = VectorXd::Zero(robot->dof());

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
			lock_guard<mutex> lock_guard_torques(mutex_torques);
			UI_torques = graphics->getUITorques(robot_name);
		}
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
	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// partial joint task to control slider and elbow
	MatrixXd joint_selection = MatrixXd::Zero(2, dof);
	joint_selection(0, 0) = 1;
	joint_selection(1, 7) = 1;
	auto partial_joint_task =
		make_unique<Sai2Primitives::JointTask>(robot, joint_selection);
	VectorXd joint_task_torques = VectorXd::Zero(dof);
	VectorXd joint_desired_pos = partial_joint_task->getCurrentPosition();

	// Motion task for end effector
	string link_name = "end-effector";
	Vector3d pos_in_link = Vector3d(0.0, 0.0, 0.07);
	Affine3d compliant_frame = Affine3d(Translation3d(pos_in_link));
	auto motion_force_task = make_unique<Sai2Primitives::MotionForceTask>(
		robot, link_name, compliant_frame);
	VectorXd motion_force_task_torques = VectorXd::Zero(dof);
	motion_force_task->disableInternalOtg();
	const Vector3d initial_position = motion_force_task->getCurrentPosition();

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

		// update tasks model. Order is important to define the hierarchy
		N_prec = MatrixXd::Identity(dof, dof);

		partial_joint_task->updateTaskModel(N_prec);
		N_prec = partial_joint_task->getN();

		motion_force_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		// partial joint task
		if (timer.elapsedCycles() % 4000 == 1000) {
			joint_desired_pos(0) -= 1.0;
		} else if (timer.elapsedCycles() % 4000 == 3000) {
			joint_desired_pos(0) += 1.0;
		}
		partial_joint_task->setDesiredPosition(joint_desired_pos);

		joint_task_torques = partial_joint_task->computeTorques();

		// motion force task
		double time = timer.elapsedSimTime();
		Vector3d desired_position =
			initial_position + 0.1 * Vector3d(sin(2.0 * M_PI * 0.3 * time), 0.0,
											  1 - cos(2.0 * M_PI * 0.3 * time));
		desired_position(1) = partial_joint_task->getCurrentPosition()(0);
		motion_force_task->setDesiredPosition(desired_position);

		motion_force_task_torques = motion_force_task->computeTorques();

		//------ compute the final torques
		{
			lock_guard<mutex> lock_guard_torques(mutex_torques);
			control_torques = joint_task_torques + motion_force_task_torques;
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
	Sai2Common::LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(sim_freq);
	double last_time = timer.elapsedTime();	 // secs
	bool fTimerDidSleep = true;

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		fTimerDidSleep = timer.waitForNextLoop();

		{
			lock_guard<mutex> lock_guard_torques(mutex_torques);
			sim->setJointTorques(robot_name, control_torques + UI_torques);
		}
		sim->integrate();
	}

	timer.stop();
	cout << "Simulation loop timer stats:\n";
	timer.printInfoPostRun();
}