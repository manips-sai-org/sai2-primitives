/*
 * Example of a controller for a Panda arm (7DoF robot) using a 6DoF
 * position plus orientation task and a joint task in its nullspace 
 * to control the redundancy. The joint task activates after a few seconds.
 * Interpolation is not used. Instead, the trajectory is directly sent 
 * to the robot.
 */

// Initialization is the same as examples 1 and 2
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

// simulation and control loop
void control(std::shared_ptr<Sai2Model::Sai2Model> robot, Sai2Simulation::Sai2Simulation* sim);
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, Sai2Simulation::Sai2Simulation* sim);

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

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file, false);
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

	// stop simulation
	fSimulationRunning = false;
	sim_thread.join();
	ctrl_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void control(std::shared_ptr<Sai2Model::Sai2Model> robot, Sai2Simulation::Sai2Simulation* sim) {
	
	// update robot model and initialize control vectors
	robot->updateModel();
	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof,dof);

	// Position plus orientation task
	string link_name = "end-effector";
	Vector3d pos_in_link = Vector3d(0.0,0.0,0.0);
	Affine3d compliant_frame = Affine3d(Translation3d(pos_in_link));
	Sai2Primitives::MotionForceTask* motion_force_task = new Sai2Primitives::MotionForceTask(robot, link_name, compliant_frame); // no orientation parameter, default is identity
	VectorXd motion_force_task_torques = VectorXd::Zero(dof);

#ifdef USING_OTG
	// disable the interpolation because trajectory is sent directly
	motion_force_task->_use_interpolation_flag = false;
#endif
	// no gains setting here, using the default task values
	const Matrix3d initial_orientation = robot->rotation(link_name);
	const Vector3d initial_position = robot->position(link_name, pos_in_link);

	// joint task to control the redundancy
	// using default gains and interpolation settings
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

		// update tasks model. Order is important to define the hierarchy
		N_prec = MatrixXd::Identity(dof,dof);

		motion_force_task->updateTaskModel(N_prec);
		N_prec = motion_force_task->getN();    
		// after each task, need to update the nullspace 
		// of the previous tasks in order to garantee 
		// the dyamic consistency

		joint_task->updateTaskModel(N_prec);

		// -------- set task goals and compute control torques
		// first the posori task.
		// orientation:
		double w_ori_traj = 2*M_PI*0.2;
		double amp_ori_traj = M_PI/8;
		double angle_ori_traj = amp_ori_traj * sin(w_ori_traj * curr_time);
		double ang_vel_traj = amp_ori_traj * w_ori_traj * cos(w_ori_traj * curr_time);
		double ang_accel_traj = amp_ori_traj * w_ori_traj * w_ori_traj * -sin(w_ori_traj * curr_time);

		Matrix3d R = AngleAxisd(angle_ori_traj, Vector3d::UnitY()).toRotationMatrix();

		motion_force_task->setDesiredOrientation(R.transpose() * initial_orientation);
		motion_force_task->setDesiredAngularVelocity(ang_vel_traj *
											   Vector3d::UnitY());
		motion_force_task->setDesiredAngularAcceleration(ang_accel_traj *
												   Vector3d::UnitY());

		// position:
		double radius_circle_pos = 0.05;
		double w_circle_pos = 2 * M_PI * 0.33;
		motion_force_task->setDesiredPosition(
			initial_position +
			radius_circle_pos * Vector3d(0.0, sin(w_circle_pos * curr_time),
										 1 - cos(w_circle_pos * curr_time)));
		motion_force_task->setDesiredVelocity(
			radius_circle_pos * w_circle_pos *
			Vector3d(0.0, cos(w_circle_pos * curr_time),
					 sin(w_circle_pos * curr_time)));
		motion_force_task->setDesiredAcceleration(
			radius_circle_pos * w_circle_pos * w_circle_pos *
			Vector3d(0.0, -sin(w_circle_pos * curr_time),
					 cos(w_circle_pos * curr_time)));

		// compute torques for the different tasks
		motion_force_task_torques = motion_force_task->computeTorques();
		joint_task_torques = joint_task->computeTorques();

		// activate joint task only after 5 seconds and try to rotate the first joint
		if(controller_counter < 5000) {
			joint_task_torques.setZero();
		}
		if(controller_counter == 5000) {
			joint_task->reInitializeTask();
			Eigen::VectorXd desired_joint_pos = initial_q;
			desired_joint_pos(0) += 1;
			joint_task->setDesiredPosition(desired_joint_pos);
		}

		//------ compute the final torques
		command_torques = motion_force_task_torques + joint_task_torques;

		// send to simulation
		sim->setJointTorques(robot_name, command_torques);
		

		// -------------------------------------------
		if(controller_counter % 500 == 0)
		{
			cout << curr_time << endl;
			cout << "position error : " << (motion_force_task->getDesiredPosition() - motion_force_task->getCurrentPosition()).norm() << endl;
			cout << endl;
		}

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
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, Sai2Simulation::Sai2Simulation* sim) {
	fSimulationRunning = true;

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

		// integrate forward
		sim->integrate();

	}

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Sai2Simulation Loop run time  : " << end_time << " seconds\n";
    std::cout << "Sai2Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Sai2Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}