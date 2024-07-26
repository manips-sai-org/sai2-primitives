/*
 * Example of singularity handling by smoothing Lambda in/out of singularities.
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
#include "logger/Logger.h"
bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

using namespace std;
using namespace Eigen;

const string world_file = "${EXAMPLE_19_FOLDER}/world.urdf";
const string robot_file = "${SAI2_MODEL_URDF_FOLDER}/puma/puma.urdf";
const string robot_name = "PUMA";  // name in the world file

// ui torques and control torques
VectorXd ui_torques;
VectorXd control_torques;

// mutex for global variables between different threads
mutex mutex_torques;
mutex mutex_robot;

// switch for which type 2 singularity
bool flag_overhead_singularity = false;

enum State {
	GO_TO_SINGULARITY,
	EXIT_SINGULARITY
};

// simulation and control loop
void control(shared_ptr<Sai2Model::Sai2Model> robot,
			 shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void simulation(shared_ptr<Sai2Model::Sai2Model> robot,
				shared_ptr<Sai2Simulation::Sai2Simulation> sim);

//------------ main function
int main(int argc, char** argv) {
	Sai2Model::URDF_FOLDERS["EXAMPLE_19_FOLDER"] =
		string(EXAMPLES_FOLDER) + "/19-puma_singularity";
	cout << "Loading URDF world model file: " << world_file << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = make_shared<Sai2Graphics::Sai2Graphics>(world_file);
	graphics->addUIForceInteraction(robot_name);
	// graphics->showTransparency(true, robot_name, 0.5);
	graphics->showLinkFrame(true, robot_name, "end-effector", 0.25);

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);
	VectorXd init_q = sim->getJointPositions(robot_name);

	/*
		Type 2 described in Marcelo's paper 
	*/
	init_q(1) = - M_PI / 4;
	init_q(2) = M_PI / 4 + M_PI / 2;
	init_q(3) = M_PI / 2;
	init_q(4) = 0;
	init_q(5) = M_PI;

	// /*
	// 	Extended elbow condition 
	// */
	// init_q(1) = 0;
	// init_q(2) = M_PI / 2;
	// init_q(3) = 0;
	// init_q(4) = 0;

	sim->setJointPositions(robot_name, init_q);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file, false);
	robot->setQ(sim->getJointPositions(robot_name));
	robot->updateModel();

	// intitialize global torques variables
	ui_torques = VectorXd::Zero(robot->dof());
	control_torques = VectorXd::Zero(robot->dof());

	// start the simulation thread first
	fSimulationRunning = true;
	thread sim_thread(simulation, robot, sim);

	// next start the control thread
	thread ctrl_thread(control, robot, sim);

	// while window is open:
	while (graphics->isWindowOpen()) {
		{
			lock_guard<mutex> lock(mutex_robot);
			graphics->updateRobotGraphics(robot_name, robot->q());
		}
		graphics->renderGraphicsWorld();
		{
			lock_guard<mutex> lock(mutex_torques);
			ui_torques = graphics->getUITorques(robot_name);
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

	// Position plus orientation task
	string link_name = "end-effector";
	// Vector3d pos_in_link = Vector3d(0.0, 0.0, 0.07);
	Vector3d pos_in_link = Vector3d(0.0, 0.0, 0.0);
	Affine3d compliant_frame = Affine3d(Translation3d(pos_in_link));

	// Full motion force task
	auto motion_force_task = make_unique<Sai2Primitives::MotionForceTask>(
		robot, link_name, compliant_frame);
	motion_force_task->setSingularityHandlingGains(50, 20, 20);

	// // Partial motion force task
	// vector<Vector3d> controlled_directions_translation = {
	// 	Vector3d::UnitX(), Vector3d::UnitY(), Vector3d::UnitZ()};
	// vector<Vector3d> controlled_directions_rotation = {};
	// auto motion_force_task = make_shared<Sai2Primitives::MotionForceTask>(
	// 	robot, link_name, controlled_directions_translation,
	// 	controlled_directions_rotation);

	VectorXd motion_force_task_torques = VectorXd::Zero(dof);

	// no gains setting here, using the default task values
	const Matrix3d initial_orientation = robot->rotation(link_name);
	const Vector3d initial_position = robot->position(link_name, pos_in_link);

	// singularity cases
	// if overhead, then move robot to overhead, then move out
	// if wrist lock, then start robot in wrist lock, and translate in z-axis
	// and rotate about z-axis by 15 degrees 

	// desired position based on singularity testing
	Matrix3d desired_orientation = Matrix3d::Identity();
	if (flag_overhead_singularity) {	
		desired_orientation = AngleAxisd(M_PI / 2, Vector3d::UnitY()).toRotationMatrix();
		// motion_force_task->setGoalPosition(Vector3d(-0.12, 0.12, 1.3));
		motion_force_task->setGoalPosition(Vector3d(0, 0, 1.3));
	} else {
	}
	// motion_force_task->setGoalOrientation(desired_orientation);

	// joint task to control the redundancy
	// using default gains and interpolation settings
	auto joint_task = make_unique<Sai2Primitives::JointTask>(robot);
    joint_task->setGains(100, 20);
	VectorXd joint_task_torques = VectorXd::Zero(dof);

	VectorXd initial_q = robot->q();
    joint_task->setGoalPosition(initial_q);

	int state = GO_TO_SINGULARITY;
	double start_time = 0;
	Vector3d starting_ee_pos;
	Matrix3d starting_ee_ori;
	int cnt = 0;

	// create a loop timer
	double control_freq = 1000;
	Sai2Common::LoopTimer timer(control_freq, 1e6);

	while (fSimulationRunning) {
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();

		// read joint positions, velocities, update model
		robot->setQ(sim->getJointPositions(robot_name));
		robot->setDq(sim->getJointVelocities(robot_name));
		robot->updateModel();

		// update kinematics
		Vector3d ee_pos = robot->position(link_name, compliant_frame.translation());
		std::cout << "ee position: " << ee_pos.transpose() << "\n";
		Matrix3d ee_ori = robot->rotation(link_name, compliant_frame.linear());

		// state transition logic 
		if (time - start_time > 5 && state == GO_TO_SINGULARITY) {
			// std::cout << "Updated direction\n";
			starting_ee_pos = ee_pos;
			starting_ee_ori = ee_ori;
			// motion_force_task->setGoalPosition(starting_ee_pos + Vector3d(-0.25, 0, 0));
			motion_force_task->setGoalPosition(starting_ee_pos + Vector3d(0, 0, -0.2));
			// motion_force_task->setGoalOrientation(starting_ee_ori * AngleAxisd(15 * M_PI / 180, Vector3d::UnitZ()));
			motion_force_task->setGoalOrientation(starting_ee_ori * AngleAxisd(90 * M_PI / 180, Vector3d::UnitZ()).toRotationMatrix());
			// motion_force_task->setGoalOrientation(AngleAxisd(15 * M_PI / 180, Vector3d::UnitZ()).toRotationMatrix() * starting_ee_ori);
			state = EXIT_SINGULARITY;
			start_time = time;
		}

		if (time - start_time > 5 && state == EXIT_SINGULARITY) {
			if (cnt == 0) {
				motion_force_task->setGoalPosition(starting_ee_pos);
				motion_force_task->setGoalOrientation(starting_ee_ori);
			} else if (cnt == 1) {
				// motion_force_task->setGoalPosition(starting_ee_pos + Vector3d(0, 0.25, 0));
				motion_force_task->setGoalPosition(starting_ee_pos + Vector3d(0, 0, -0.2));
				motion_force_task->setGoalOrientation(starting_ee_ori * AngleAxisd(90 * M_PI / 180, Vector3d::UnitZ()).toRotationMatrix());
			} else if (cnt == 2) {
				motion_force_task->setGoalPosition(starting_ee_pos);
				motion_force_task->setGoalOrientation(starting_ee_ori);
			} else if (cnt == 3) {
				// motion_force_task->setGoalPosition(starting_ee_pos + Vector3d(-0.25, 0, 0));
				motion_force_task->setGoalPosition(starting_ee_pos + Vector3d(0, 0, -0.2));
				motion_force_task->setGoalOrientation(starting_ee_ori * AngleAxisd(90 * M_PI / 180, Vector3d::UnitZ()).toRotationMatrix());
			}
			start_time = time;
			cnt++;
			if (cnt == 4) {
				cnt = 0;
			}
		}

		// std::cout << "pos error: " << motion_force_task->getPositionError().norm() << "\n";
		// std::cout << "ori error: " << motion_force_task->getOrientationError().norm() << "\n";

		// update tasks model. Order is important to define the hierarchy
		N_prec = MatrixXd::Identity(dof, dof);
		{
			lock_guard<mutex> lock(mutex_robot);
			motion_force_task->updateTaskModel(N_prec);
		}
		N_prec = motion_force_task->getTaskAndPreviousNullspace();
		// after each task, need to update the nullspace
		// of the previous tasks in order to garantee
		// the dyamic consistency

		joint_task->updateTaskModel(N_prec);

		// compute torques for the different tasks
		motion_force_task_torques = motion_force_task->computeTorques();
		joint_task_torques = joint_task->computeTorques();

		//------ compute the final torques
		{
			lock_guard<mutex> lock(mutex_torques);
			control_torques = motion_force_task_torques + joint_task_torques;
		}

		// -------------------------------------------
		if (timer.elapsedCycles() % 500 == 0) {
			cout << "time: " << time << endl;
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

	sim->disableJointLimits(robot_name);

	// create a timer
	double sim_freq = 2000;
	Sai2Common::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);

	while (fSimulationRunning) {
		timer.waitForNextLoop();
		{
			lock_guard<mutex> lock(mutex_torques);
			sim->setJointTorques(robot_name, control_torques + ui_torques);
		}
		sim->integrate();
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
}