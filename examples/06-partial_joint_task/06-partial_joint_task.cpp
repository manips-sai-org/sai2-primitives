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

#include "RobotController.h"
#include "SaiGraphics.h"
#include "SaiModel.h"
#include "SaiSimulation.h"
#include "tasks/JointTask.h"
#include "tasks/MotionForceTask.h"
#include "timer/LoopTimer.h"
bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

using namespace std;
using namespace Eigen;

const string world_file = "${EXAMPLE_06_FOLDER}/world.urdf";
const string robot_file = "${EXAMPLE_06_FOLDER}/panda_arm_sliding_base.urdf";
const string robot_name = "PANDA";

// ui and control torques
VectorXd UI_torques;
VectorXd control_torques;

// mutex to read and write the control torques
mutex mutex_torques;

// simulation and control loop
void control(shared_ptr<SaiModel::SaiModel> robot,
			 shared_ptr<SaiSimulation::SaiSimulation> sim);
void simulation(shared_ptr<SaiModel::SaiModel> robot,
				shared_ptr<SaiSimulation::SaiSimulation> sim);

//------------ main function
int main(int argc, char** argv) {
	SaiModel::URDF_FOLDERS["EXAMPLE_06_FOLDER"] =
		string(EXAMPLES_FOLDER) + "/06-partial_joint_task";
	cout << "Loading URDF world model file: "
		 << SaiModel::ReplaceUrdfPathPrefix(world_file) << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = make_shared<SaiGraphics::SaiGraphics>(world_file);
	graphics->addUIForceInteraction(robot_name);

	// load simulation world
	auto sim = make_shared<SaiSimulation::SaiSimulation>(world_file);

	// load robots
	auto robot = make_shared<SaiModel::SaiModel>(robot_file, false);
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
void control(shared_ptr<SaiModel::SaiModel> robot,
			 shared_ptr<SaiSimulation::SaiSimulation> sim) {
	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// partial joint task to control slider and elbow
	MatrixXd joint_selection = MatrixXd::Zero(2, dof);
	joint_selection(0, 0) = 1;
	joint_selection(1, 7) = 1;
	auto partial_joint_task =
		make_shared<SaiPrimitives::JointTask>(robot, joint_selection);
	VectorXd joint_goal_pos = partial_joint_task->getCurrentPosition();

	// Motion task for end effector
	string link_name = "end-effector";
	Vector3d pos_in_link = Vector3d(0.0, 0.0, 0.07);
	Affine3d compliant_frame = Affine3d(Translation3d(pos_in_link));
	auto motion_force_task = make_shared<SaiPrimitives::MotionForceTask>(
		robot, link_name, compliant_frame);
	motion_force_task->disableInternalOtg();
	const Vector3d initial_position = motion_force_task->getCurrentPosition();

	// make controller
	vector<shared_ptr<SaiPrimitives::TemplateTask>> task_list = {
		partial_joint_task, motion_force_task};
	auto robot_controller =
		make_unique<SaiPrimitives::RobotController>(robot, task_list);

	// create a loop timer
	double control_freq = 1000;
	SaiCommon::LoopTimer timer(control_freq, 1e6);

	while (fSimulationRunning) {  // automatically set to false when simulation
								  // is quit
		timer.waitForNextLoop();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update tasks model
		robot_controller->updateControllerTaskModels();

		// -------- set task goals and compute control torques
		// partial joint task
		if (timer.elapsedCycles() % 4000 == 1000) {
			joint_goal_pos(0) -= 1.0;
		} else if (timer.elapsedCycles() % 4000 == 3000) {
			joint_goal_pos(0) += 1.0;
		}
		partial_joint_task->setGoalPosition(joint_goal_pos);

		// motion force task
		double time = timer.elapsedSimTime();
		double w = 2.0 * M_PI * 0.3;
		Vector3d goal_position =
			initial_position + 0.1 * Vector3d(sin(w * time), 0.0,
											  1 - cos(w * time));
		goal_position(1) = partial_joint_task->getDesiredPosition()(0);
		Vector3d goal_velocity = 0.1 * Vector3d(w * cos(w * time), 0.0,
											  w * sin(w * time));
		goal_velocity(1) = partial_joint_task->getDesiredVelocity()(0);
		Vector3d goal_acceleration = 0.1 * Vector3d(- w * w * sin(w * time), 0.0,
											  w * w * cos(w * time));	
		goal_acceleration(1) = partial_joint_task->getDesiredAcceleration()(0);	
		motion_force_task->setGoalPosition(goal_position);
		motion_force_task->setGoalLinearVelocity(goal_velocity);
		motion_force_task->setGoalLinearAcceleration(goal_acceleration);

		//------ compute the final torques
		{
			lock_guard<mutex> lock_guard_torques(mutex_torques);
			control_torques = robot_controller->computeControlTorques();
		}
	}
	timer.stop();
	cout << "\nControl loop timer stats:\n";
	timer.printInfoPostRun();
}

//------------------------------------------------------------------------------
void simulation(shared_ptr<SaiModel::SaiModel> robot,
				shared_ptr<SaiSimulation::SaiSimulation> sim) {
	fSimulationRunning = true;

	// create a timer
	double sim_freq = 2000;
	SaiCommon::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();

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