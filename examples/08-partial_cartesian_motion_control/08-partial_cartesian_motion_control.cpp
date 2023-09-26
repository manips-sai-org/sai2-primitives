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
#include "tasks/JointTask.h"

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
const string robot_name = "PUMA";  // name in the world file

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

	// prepare the task to control y-z position and rotation around z
	string link_name = "end-effector";
	Vector3d pos_in_link = Vector3d(0.07, 0.0, 0.0);
	Affine3d compliant_frame_in_link = Affine3d(Translation3d(pos_in_link));
	vector<Vector3d> controlled_directions_translation = {Vector3d::UnitY(),
														  Vector3d::UnitZ()};
	vector<Vector3d> controlled_directions_rotation = {Vector3d::UnitX()};
	auto motion_force_task = make_shared<Sai2Primitives::MotionForceTask>(
		robot, link_name, controlled_directions_translation,
		controlled_directions_rotation, compliant_frame_in_link);

	motion_force_task->disableInternalOtg();
	motion_force_task->disableVelocitySaturation();

	// gains for the position and orientation parts of the controller
	motion_force_task->setPosControlGains(100.0, 20.0);
	motion_force_task->setOriControlGains(100.0, 20.0);

	// initial position and orientation
	Matrix3d initial_orientation = robot->rotation(link_name);
	Vector3d initial_position = robot->position(link_name, pos_in_link);
	Vector3d desired_position = initial_position;
	Matrix3d desired_orientation = initial_orientation;

	// joint task to control the nullspace
	auto joint_task = make_shared<Sai2Primitives::JointTask>(robot);
	joint_task->setGains(100.0, 20.0);
	joint_task->disableInternalOtg();
	joint_task->disableVelocitySaturation();

	// create a loop timer
	double control_freq = 1000;
	Sai2Common::LoopTimer timer(control_freq);
	timer.initializeTimer(1000000);	 // 1 ms pause before starting loop

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update tasks model
		N_prec = MatrixXd::Identity(dof, dof);
		motion_force_task->updateTaskModel(N_prec);
		N_prec = motion_force_task->getTaskAndPreviousNullspace();
		joint_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		double time = timer.elapsedSimTime();

		// try to move in X, cannot do it because the partial task does not control X direction
		if(timer.elapsedCycles() == 1000) {
			desired_position += Vector3d(0.1, 0.0, 0.0);
		}
		// then move in Y and Z, this should be doable
		if(timer.elapsedCycles() == 2000) {
			desired_position += Vector3d(0.0, 0.1, 0.1);
		}

		// try to rotate around Z, should not be able to do it
		if(timer.elapsedCycles() == 3000) {
			desired_orientation = AngleAxisd(M_PI/6, Vector3d::UnitZ()) * initial_orientation;
		}
		// then rotate around X, this should be doable
		if(timer.elapsedCycles() == 4000) {
			desired_orientation = AngleAxisd(M_PI/6, Vector3d::UnitX()) * initial_orientation;
		}

		// move the first joint in the nullspace of the partial task
		if (timer.elapsedCycles() == 8000) {
			VectorXd q_des = robot->q();
			q_des(0) += 0.5;
			joint_task->setDesiredPosition(q_des);
		}

		motion_force_task->setDesiredPosition(desired_position);
		motion_force_task->setDesiredOrientation(desired_orientation);

		VectorXd motion_force_task_torques =
			motion_force_task->computeTorques();
		VectorXd joint_task_torques = joint_task->computeTorques();

		//------ Control torques
		{
			lock_guard<mutex> guard(mmutex_torques);
			control_torques = motion_force_task_torques + joint_task_torques;
		}

		// -------------------------------------------
		if (timer.elapsedCycles() % 500 == 0) {
			cout << time << endl;
			cout << "desired position : "
				 << motion_force_task->getDesiredPosition().transpose() << endl;
			cout << "desired position projected in controlled space: "
				 << (motion_force_task->posSelectionProjector() * motion_force_task->getDesiredPosition()).transpose() << endl;				 
			cout << "current position : "
				 << motion_force_task->getCurrentPosition().transpose() << endl;
			cout << "current position projected in controlled space: "
				 << (motion_force_task->posSelectionProjector()*motion_force_task->getCurrentPosition()).transpose() << endl;
			cout << "position error : "
				 << (motion_force_task->posSelectionProjector()*(motion_force_task->getDesiredPosition() -
					 motion_force_task->getCurrentPosition()))
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
	timer.initializeTimer();

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