/*
 * Example of a controller for two Panda arms manipulating a tray together 
 */

// includes
#include "Sai2Model.h"
#include "Sai2Graphics.h"
#include "Sai2Simulation.h"
#include "Sai2Primitives.h"
#include "timer/LoopTimer.h"

#include <GLFW/glfw3.h> //must be loaded after loading opengl/glew

#include <iostream>
#include <string>

#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int){fSimulationRunning = false;}

using namespace std;
using namespace Eigen;

const string world_file = "./resources/world.urdf";
const vector<string> robot_sim_files = {
	"./resources/panda_arm_hand.urdf",
	"./resources/panda_arm_hand.urdf",
};
const vector<string> robot_control_files = {
	"./resources/panda_arm.urdf",
	"./resources/panda_arm.urdf",
};
const vector<string> robot_names = {
	"PANDA1",
	"PANDA2",
};
const vector<string> object_names = {
	"tray",
};
const string camera_name = "camera";

const int n_robots = robot_names.size();
const int n_objects = object_names.size();

vector<VectorXd> robot_joint_positions = {
	VectorXd::Zero(7),
	VectorXd::Zero(7),
};

vector<VectorXd> robot_joint_velocities = {
	VectorXd::Zero(7),
	VectorXd::Zero(7),
};

vector<VectorXd> robot_control_torques = {
	VectorXd::Zero(7),
	VectorXd::Zero(7),
};

// - grippers
vector<string> gripper_modes = {   // m for move and g for graps
	"m",
	"m",
};
vector<double> gripper_current_widths = {
	0,
	0,
};
vector<double> gripper_desired_widths = {
	0.04,
	0.04
};
vector<double> gripper_desired_speeds = {
	0,
	0,
};
vector<double> gripper_desired_forces = {
	0,
	0,
};

const vector<double> gripper_max_widths = {
	0.08,
	0.08,
};
vector<double> gripper_widths = {
	0,
	0,
};

vector<Vector3d> object_positions;
vector<Quaterniond> object_orientations;

// simulation function prototype
void simulation(vector<Sai2Model::Sai2Model*> robots, Simulation::Sai2Simulation* sim);
void control(Simulation::Sai2Simulation* sim);

// function to perform gripper control for a given robot
Vector2d gripperControl(const int robot_index, vector<Sai2Model::Sai2Model*> robots);

// callback to print glfw errors
void glfwError(int error, const char* description);

// callback when a key is pressed
void keySelect(GLFWwindow* window, int key, int scancode, int action, int mods);

// callback when a mouse button is pressed
void mouseClick(GLFWwindow* window, int button, int action, int mods);

// flags for scene camera movement
bool fTransXp = false;
bool fTransXn = false;
bool fTransYp = false;
bool fTransYn = false;
bool fTransZp = false;
bool fTransZn = false;
bool fRotPanTilt = false;

// state machine
#define PRE_PICK_TRAY               0
#define PICK_TRAY                   1
#define LIFT_TRAY                   2

int state = PRE_PICK_TRAY;
unsigned long long controller_counter = 0;

int main() {
	cout << "Loading URDF world model file: " << world_file << endl;

	// load graphics scene
	auto graphics = new Sai2Graphics::Sai2Graphics(world_file, true);
	Vector3d camera_pos, camera_lookat, camera_vertical;
	graphics->getCameraPose(camera_name, camera_pos, camera_vertical, camera_lookat);

	// load simulation world
	auto sim = new Simulation::Sai2Simulation(world_file, false);
	sim->setCollisionRestitution(0);
	sim->setCoeffFrictionStatic(0.8);

	// load robots
	vector<Sai2Model::Sai2Model*> robots_sim;
	for(int i=0 ; i<n_robots ; i++)
	{
		robots_sim.push_back(new Sai2Model::Sai2Model(robot_sim_files[i], false, sim->getRobotBaseTransform(robot_names[i])));
	}

	// read joint positions, velocities, update model
	for(int i=0 ; i<n_robots ; i++)
	{
		sim->getJointPositions(robot_names[i], robots_sim[i]->_q);
		sim->getJointVelocities(robot_names[i], robots_sim[i]->_dq);
		robots_sim[i]->updateKinematics();
	}

	// read objects initial positions
	for(int i=0 ; i< n_objects ; i++)
	{
		Vector3d obj_pos = Vector3d::Zero();
		Quaterniond obj_ori = Quaterniond::Identity();
		sim->getObjectPosition(object_names[i], obj_pos, obj_ori);
		object_positions.push_back(obj_pos);
		object_orientations.push_back(obj_ori);
	}

	/*------- Set up visualization -------*/
	// set up error callback
	glfwSetErrorCallback(glfwError);

	// initialize GLFW
	glfwInit();

	// retrieve resolution of computer display and position window accordingly
	GLFWmonitor* primary = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(primary);

	// information about computer screen and GLUT display window
	int screenW = mode->width;
	int screenH = mode->height;
	int windowW = 0.8 * screenH;
	int windowH = 0.5 * screenH;
	int windowPosY = (screenH - windowH) / 2;
	int windowPosX = windowPosY;

	// create window and make it current
	glfwWindowHint(GLFW_VISIBLE, 0);
	GLFWwindow* window = glfwCreateWindow(windowW, windowH, "SAI2.0 - PandaApplications", NULL, NULL);
	glfwSetWindowPos(window, windowPosX, windowPosY);
	glfwShowWindow(window);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// set callbacks
	glfwSetKeyCallback(window, keySelect);
	glfwSetMouseButtonCallback(window, mouseClick);

	// cache variables
	double last_cursorx, last_cursory;

	fSimulationRunning = true;
	thread sim_thread(simulation, robots_sim, sim);
	thread control_thread(control, sim);

	// while window is open:
	while (!glfwWindowShouldClose(window))
	{
		// update graphics. this automatically waits for the correct amount of time
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		for(int i=0 ; i<n_robots ; i++)
		{
			graphics->updateGraphics(robot_names[i], robots_sim[i]);
		}
		for(int i=0 ; i< n_objects ; i++)
		{
			graphics->updateObjectGraphics(object_names[i], object_positions[i], object_orientations[i]);
		}
		graphics->render(camera_name, width, height);

		// swap buffers
		glfwSwapBuffers(window);

		// wait until all GL commands are completed
		glFinish();

		// check for any OpenGL errors
		GLenum err;
		err = glGetError();
		assert(err == GL_NO_ERROR);

		// poll for events
		glfwPollEvents();

		// move scene camera as required
		// graphics->getCameraPose(camera_name, camera_pos, camera_vertical, camera_lookat);
		Vector3d cam_depth_axis;
		cam_depth_axis = camera_lookat - camera_pos;
		cam_depth_axis.normalize();
		Vector3d cam_up_axis;
		// cam_up_axis = camera_vertical;
		// cam_up_axis.normalize();
		cam_up_axis << 0.0, 0.0, 1.0; //TODO: there might be a better way to do this
		Vector3d cam_roll_axis = (camera_lookat - camera_pos).cross(cam_up_axis);
		cam_roll_axis.normalize();
		Vector3d cam_lookat_axis = camera_lookat;
		cam_lookat_axis.normalize();
		if (fTransXp) {
			camera_pos = camera_pos + 0.05*cam_roll_axis;
			camera_lookat = camera_lookat + 0.05*cam_roll_axis;
		}
		if (fTransXn) {
			camera_pos = camera_pos - 0.05*cam_roll_axis;
			camera_lookat = camera_lookat - 0.05*cam_roll_axis;
		}
		if (fTransYp) {
			// camera_pos = camera_pos + 0.05*cam_lookat_axis;
			camera_pos = camera_pos + 0.05*cam_up_axis;
			camera_lookat = camera_lookat + 0.05*cam_up_axis;
		}
		if (fTransYn) {
			// camera_pos = camera_pos - 0.05*cam_lookat_axis;
			camera_pos = camera_pos - 0.05*cam_up_axis;
			camera_lookat = camera_lookat - 0.05*cam_up_axis;
		}
		if (fTransZp) {
			camera_pos = camera_pos + 0.1*cam_depth_axis;
			camera_lookat = camera_lookat + 0.1*cam_depth_axis;
		}	    
		if (fTransZn) {
			camera_pos = camera_pos - 0.1*cam_depth_axis;
			camera_lookat = camera_lookat - 0.1*cam_depth_axis;
		}
		if (fRotPanTilt) {
			// get current cursor position
			double cursorx, cursory;
			glfwGetCursorPos(window, &cursorx, &cursory);
			//TODO: might need to re-scale from screen units to physical units
			double compass = 0.006*(cursorx - last_cursorx);
			double azimuth = 0.006*(cursory - last_cursory);
			double radius = (camera_pos - camera_lookat).norm();
			Matrix3d m_tilt; m_tilt = AngleAxisd(azimuth, -cam_roll_axis);
			camera_pos = camera_lookat + m_tilt*(camera_pos - camera_lookat);
			Matrix3d m_pan; m_pan = AngleAxisd(compass, -cam_up_axis);
			camera_pos = camera_lookat + m_pan*(camera_pos - camera_lookat);
		}
		graphics->setCameraPose(camera_name, camera_pos, cam_up_axis, camera_lookat);
		glfwGetCursorPos(window, &last_cursorx, &last_cursory);
	}

	// stop simulation
	fSimulationRunning = false;
	sim_thread.join();
	control_thread.join();

	// destroy context
	glfwDestroyWindow(window);

	// terminate
	glfwTerminate();

	return 0;
}

//------------------------------------------------------------------------------
void simulation(vector<Sai2Model::Sai2Model*> robots_sim, Simulation::Sai2Simulation* sim) {

	// create a timer
	double sim_freq = 2000.0;
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(sim_freq); 
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs
	double last_time = start_time;

	unsigned long long simulation_counter = 0;

	while (fSimulationRunning) {
		fTimerDidSleep = timer.waitForNextLoop();

		for(int i=0 ; i<n_robots ; i++)
		{
			int dof = robots_sim[i]->dof();
			VectorXd command_torques = VectorXd::Zero(dof);
			// read arm torques from controller
			command_torques.head(dof-2) = robot_control_torques[i];

			// compute gripper torques
			Vector2d gripper_torques = gripperControl(i, robots_sim);

			command_torques(dof-2) = gripper_torques(0);
			command_torques(dof-1) = gripper_torques(1);

			// set robot torques to simulation
			sim->setJointTorques(robot_names[i], command_torques);
		}

		// integrate forward
		sim->integrate(1.0/sim_freq);

		// read joint positions, velocities, update model in the simulation because robot model is different in simulation and control
		for(int i=0 ; i<n_robots ; i++)
		{
			sim->getJointPositions(robot_names[i], robots_sim[i]->_q);
			sim->getJointVelocities(robot_names[i], robots_sim[i]->_dq);
			robots_sim[i]->updateKinematics();
		}

		// get object positions from simulation
		for(int i=0 ; i< n_objects ; i++)
		{
			sim->getObjectPosition(object_names[i], object_positions[i], object_orientations[i]);
		}

		// send the first 7 joints to the robot models in the controller
		for(int i=0 ; i<n_robots ; i++)
		{
			robot_joint_positions[i] = robots_sim[i]->_q.head<7>();
			robot_joint_velocities[i] = robots_sim[i]->_dq.head<7>();
		}

		simulation_counter++;
	}

	double end_time = timer.elapsedTime();
	cout << "\n";
	cout << "Simulation Loop run time  : " << end_time << " seconds\n";
	cout << "Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
	cout << "Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}

//------------------------------------------------------------------------------
Vector2d gripperControl(const int robot_index, vector<Sai2Model::Sai2Model*> robots_sim)
{
	int i = robot_index;
	Vector2d gripper_torques = Vector2d::Zero();

	// controller gains and force
	double kp_gripper = 400.0;
	double kv_gripper = 40.0;
	double gripper_behavior_force = 0;

	// gripper state
	double gripper_center_point = (robots_sim[i]->_q(7) + robots_sim[i]->_q(8))/2.0;
	double gripper_center_point_velocity = (robots_sim[i]->_dq(7) + robots_sim[i]->_dq(8))/2.0;

	gripper_widths[i] = (robots_sim[i]->_q(7) - robots_sim[i]->_q(8))/2;
	double gripper_opening_speed = (robots_sim[i]->_dq(7) - robots_sim[i]->_dq(8))/2;

	// compute gripper torques
	double gripper_desired_width = gripper_desired_widths[i];
	double gripper_desired_speed = gripper_desired_speeds[i];
	double gripper_desired_force = gripper_desired_forces[i];
	string gripper_mode = gripper_modes[i];
	if(gripper_desired_width > gripper_max_widths[i])
	{
		gripper_desired_width = gripper_max_widths[i];
		cout << "WARNING : Desired gripper " << i << " width higher than max width. saturating to max width\n" << endl;
	}
	if(gripper_desired_width < 0)
	{
		gripper_desired_width = 0;
		cout << "WARNING : Desired gripper " << i << " width lower than 0. saturating to 0\n" << endl;
	}
	if(gripper_desired_speed < 0)
	{
		gripper_desired_speed = 0;
		cout << "WARNING : Desired gripper " << i << " speed lower than 0. saturating to 0\n" << endl;
	} 
	if(gripper_desired_force < 0)
	{
		gripper_desired_force = 0;
		cout << "WARNING : Desired gripper " << i << " force lower than 0. saturating to 0\n" << endl;
	}

	double gripper_constraint_force = -400.0*gripper_center_point - 40.0*gripper_center_point_velocity;

	if(gripper_mode == "m")
	{
		gripper_behavior_force = -kp_gripper*(gripper_widths[i] - gripper_desired_width) - kv_gripper*(gripper_opening_speed - gripper_desired_speed);
	}
	else if(gripper_mode == "g")
	{
		gripper_behavior_force = -gripper_desired_force;
	}
	else
	{
		cout << "gripper mode not recognized\n" << endl;
	}

	gripper_torques(0) = gripper_constraint_force + gripper_behavior_force;
	gripper_torques(1) = gripper_constraint_force - gripper_behavior_force;

	return gripper_torques;

}


//------------------------------------------------------------------------------
void control(Simulation::Sai2Simulation* sim)
{
	// object gravity
	VectorXd object_gravity = VectorXd::Zero(6);
	object_gravity << 0, 0, -9.81, 0, 0, 0;

	// position of robots in world
	vector<Affine3d> robot_pose_in_world;
	Affine3d pose = Affine3d::Identity();
	pose.translation() = Vector3d(0, -0.5, 0.0);
	pose.linear() = AngleAxisd(0.3010693, Vector3d::UnitZ()).toRotationMatrix();
	robot_pose_in_world.push_back(pose);

	pose.translation() = Vector3d(-0.06, 0.57, 0.0);
	pose.linear() = AngleAxisd(-1.0864675, Vector3d::UnitZ()).toRotationMatrix();
	robot_pose_in_world.push_back(pose);

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load robots for controller
	vector<Sai2Model::Sai2Model*> robots_control;
	for(int i=0 ; i<n_robots ; i++)
	{
		robots_control.push_back(new Sai2Model::Sai2Model(robot_control_files[i], false, sim->getRobotBaseTransform(robot_names[i])));
		robots_control[i]->_q = robot_joint_positions[i];
		robots_control[i]->_dq = robot_joint_velocities[i];
		robots_control[i]->updateModel();
	}

	// prepare task controllers
	vector<int> dof;
	vector<VectorXd> command_torques;
	vector<VectorXd> coriolis;
	vector<MatrixXd> N_prec;

	vector<Sai2Primitives::JointTask*> joint_tasks;
	vector<VectorXd> joint_task_torques;
	vector<Sai2Primitives::PosOriTask*> posori_tasks;
	vector<VectorXd> posori_task_torques;

	for(int i=0 ; i<n_robots ; i++)
	{
		dof.push_back(robots_control[i]->dof());
		command_torques.push_back(VectorXd::Zero(dof[i]));
		coriolis.push_back(VectorXd::Zero(dof[i]));
		N_prec.push_back(MatrixXd::Identity(dof[i],dof[i]));

		// joint tasks
		joint_tasks.push_back(new Sai2Primitives::JointTask(robots_control[i]));
		joint_task_torques.push_back(VectorXd::Zero(dof[i]));

		joint_tasks[i]->_kp = 50.0;
		joint_tasks[i]->_kv = 14.0;

		// end effector tasks
		string link_name = "link7";
		Vector3d pos_in_link = Vector3d(0.0,0.0,0.2);
		posori_tasks.push_back(new Sai2Primitives::PosOriTask(robots_control[i], link_name, pos_in_link));
		posori_task_torques.push_back(VectorXd::Zero(dof[i]));

		posori_tasks[i]->_kp_pos = 200.0;
		posori_tasks[i]->_kv_pos = 25.0;
		posori_tasks[i]->_kp_ori = 400.0;
		posori_tasks[i]->_kv_ori = 40.0;		

	#ifndef USING_OTG  // use velocity saturation is OTG is not compiled
		posori_tasks[i]->_use_velocity_saturation_flag = true;
		posori_tasks[i]->_linear_saturation_velocity = 0.15;
		posori_tasks[i]->_angular_saturation_velocity = 45.0/180.0*M_PI;
	#endif

	}

	// collaborative task
	Sai2Primitives::TwoHandTwoRobotsTask* two_hand_task = new Sai2Primitives::TwoHandTwoRobotsTask(robots_control[0], robots_control[1], "link7", "link7", Vector3d(0.0,0.0,0.2), Vector3d(0.0,0.0,0.2));

#ifndef USING_OTG // use velocity saturation is OTG is not compiled
	two_hand_task->_use_velocity_saturation_flag = true;
	two_hand_task->_linear_saturation_velocity = 0.15;
	two_hand_task->_angular_saturation_velocity = 20.0/180.0*M_PI;
#endif

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(1000); 
	double current_time = 0;
	double prev_time = 0;
	double dt = 0;
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs

	while (fSimulationRunning) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		current_time = timer.elapsedTime() - start_time;
		dt = current_time - prev_time;

		// read robot state and update robot model
		for(int i=0 ; i<n_robots ; i++)
		{
			robots_control[i]->_q = robot_joint_positions[i];
			robots_control[i]->_dq = robot_joint_velocities[i];

			robots_control[i]->updateModel();
			robots_control[i]->coriolisForce(coriolis[i]);
		}

		if(state == PRE_PICK_TRAY) // control the robots independently
		{
			if(controller_counter % 50 == 0)
			{
				for(int i=0 ; i<n_robots ; i++)
				{
					N_prec[i].setIdentity();
					posori_tasks[i]->updateTaskModel(N_prec[i]);

					N_prec[i] = posori_tasks[i]->_N;
					joint_tasks[i]->updateTaskModel(N_prec[i]);
				}
			}

			// set goal positions
			Vector3d robot1_desired_position_in_world = Vector3d(0.4, -0.15, 0.1);
			Matrix3d robot1_desired_orientation_in_world = AngleAxisd(180.0/180.0*M_PI, Vector3d::UnitX()).toRotationMatrix()*AngleAxisd(45.0/180.0*M_PI, Vector3d::UnitZ()).toRotationMatrix();
			Vector3d robot2_desired_position_in_world = Vector3d(0.4,  0.15, 0.1);
			Matrix3d robot2_desired_orientation_in_world = AngleAxisd(180.0/180.0*M_PI, Vector3d::UnitX()).toRotationMatrix()*AngleAxisd(45.0/180.0*M_PI, Vector3d::UnitZ()).toRotationMatrix();

			posori_tasks[0]->_desired_position = robot_pose_in_world[0].linear().transpose()*(robot1_desired_position_in_world - robot_pose_in_world[0].translation());
			posori_tasks[0]->_desired_orientation = robot_pose_in_world[0].linear().transpose()*robot1_desired_orientation_in_world;
			posori_tasks[1]->_desired_position = robot_pose_in_world[1].linear().transpose()*(robot2_desired_position_in_world - robot_pose_in_world[1].translation());
			posori_tasks[1]->_desired_orientation = robot_pose_in_world[1].linear().transpose()*robot2_desired_orientation_in_world;

			// compute torques
			for(int i=0 ; i<n_robots ; i++)
			{
				posori_tasks[i]->computeTorques(posori_task_torques[i]);
				joint_tasks[i]->computeTorques(joint_task_torques[i]);

				command_torques[i] = posori_task_torques[i] + joint_task_torques[i] + coriolis[i];
			}

			if((posori_tasks[0]->_desired_position - posori_tasks[0]->_current_position).norm() 
				+ (posori_tasks[1]->_desired_position - posori_tasks[1]->_current_position).norm() < 0.01)
			{
				state = PICK_TRAY;
			}
		}

		else if(state == PICK_TRAY)  // control the robots independently
		{
			if(controller_counter % 50 == 0)
			{
				for(int i=0 ; i<n_robots ; i++)
				{
					N_prec[i].setIdentity();
					posori_tasks[i]->updateTaskModel(N_prec[i]);

					N_prec[i] = posori_tasks[i]->_N;
					joint_tasks[i]->updateTaskModel(N_prec[i]);
				}
			}

			// set goal positions
			Vector3d robot1_desired_position_in_world = Vector3d(0.4, -0.15, 0.03);
			Matrix3d robot1_desired_orientation_in_world = AngleAxisd(180.0/180.0*M_PI, Vector3d::UnitX()).toRotationMatrix()*AngleAxisd(45.0/180.0*M_PI, Vector3d::UnitZ()).toRotationMatrix();
			Vector3d robot2_desired_position_in_world = Vector3d(0.4,  0.15, 0.03);
			Matrix3d robot2_desired_orientation_in_world = AngleAxisd(180.0/180.0*M_PI, Vector3d::UnitX()).toRotationMatrix()*AngleAxisd(45.0/180.0*M_PI, Vector3d::UnitZ()).toRotationMatrix();

			posori_tasks[0]->_desired_position = robot_pose_in_world[0].linear().transpose()*(robot1_desired_position_in_world - robot_pose_in_world[0].translation());
			posori_tasks[0]->_desired_orientation = robot_pose_in_world[0].linear().transpose()*robot1_desired_orientation_in_world;
			posori_tasks[1]->_desired_position = robot_pose_in_world[1].linear().transpose()*(robot2_desired_position_in_world - robot_pose_in_world[1].translation());
			posori_tasks[1]->_desired_orientation = robot_pose_in_world[1].linear().transpose()*robot2_desired_orientation_in_world;

			// compute torques
			for(int i=0 ; i<n_robots ; i++)
			{
				posori_tasks[i]->computeTorques(posori_task_torques[i]);
				joint_tasks[i]->computeTorques(joint_task_torques[i]);

				command_torques[i] = posori_task_torques[i] + joint_task_torques[i] + coriolis[i];
			}

			if((posori_tasks[0]->_desired_position - posori_tasks[0]->_current_position).norm() 
				+ (posori_tasks[1]->_desired_position - posori_tasks[1]->_current_position).norm() < 1e-2)
			{
				// grasp the sides of the tray
				gripper_modes[0] = "g";
				gripper_modes[1] = "g";
				gripper_desired_forces[0] = 5.0;
				gripper_desired_forces[1] = 5.0;
				state = LIFT_TRAY;

				joint_tasks[0]->reInitializeTask();
				joint_tasks[1]->reInitializeTask();

				// prepare the two hands task with the object inertial properties
				two_hand_task->reInitializeTask();
				two_hand_task->setObjectMassPropertiesAndInitialInertialFrameLocation(0.5, Affine3d::Identity(), Matrix3d::Identity());

				two_hand_task->_desired_object_position(2) += 0.15;
				two_hand_task->_desired_internal_tension = -1;
			}
		}

		else if(state == LIFT_TRAY)   // coordinated motion
		{

			// update tasks model
			if(controller_counter % 50 == 0)
			{
				for(int i=0 ; i<n_robots ; i++)
				{
					N_prec[i].setIdentity();
				}

				two_hand_task->updateTaskModel(N_prec[0], N_prec[1]);
				N_prec[0] = two_hand_task->_N_1;
				N_prec[1] = two_hand_task->_N_2;


				for(int i=0 ; i<n_robots ; i++)
				{
					N_prec[i] = posori_tasks[i]->_N;
					joint_tasks[i]->updateTaskModel(N_prec[i]);
				}
			}

			// rotate the object after a few seconds
			if(controller_counter == 5000)
			{
				two_hand_task->_desired_object_orientation = AngleAxisd(30.0/180.0*M_PI, Vector3d::UnitZ()).toRotationMatrix()*two_hand_task->_desired_object_orientation;
			}

			// compute torques
			two_hand_task->computeTorques(posori_task_torques[0], posori_task_torques[1]);
			for(int i=0 ; i<n_robots ; i++)
			{
				joint_tasks[i]->computeTorques(joint_task_torques[i]);

				command_torques[i] = posori_task_torques[i] + joint_task_torques[i] + coriolis[i];
			}

		}

		else
		{
			for(int i=0 ; i<n_robots ; i++)
			{
				command_torques[i].setZero(dof[i]);
			}
		}

		// send control commands
		for(int i=0 ; i<n_robots ; i++)
		{
			robot_control_torques[i] = command_torques[i];
		}

		prev_time = current_time;
		controller_counter++;
	}

	for(int i=0 ; i<n_robots ; i++)
	{
		command_torques[i].setZero();
		robot_control_torques[i] = command_torques[i];
	}

	double end_time = timer.elapsedTime();
	cout << "\n";
	cout << "Controller Loop run time  : " << end_time << " seconds\n";
	cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}


//------------------------------------------------------------------------------

void glfwError(int error, const char* description) {
	cerr << "GLFW Error: " << description << endl;
	exit(1);
}

//------------------------------------------------------------------------------

void keySelect(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	bool set = (action != GLFW_RELEASE);
	switch(key) {
		case GLFW_KEY_ESCAPE:
			// exit application
			glfwSetWindowShouldClose(window,GL_TRUE);
			break;
		case GLFW_KEY_RIGHT:
			fTransXp = set;
			break;
		case GLFW_KEY_LEFT:
			fTransXn = set;
			break;
		case GLFW_KEY_UP:
			fTransYp = set;
			break;
		case GLFW_KEY_DOWN:
			fTransYn = set;
			break;
		case GLFW_KEY_A:
			fTransZp = set;
			break;
		case GLFW_KEY_Z:
			fTransZn = set;
			break;
		default:
			break;
	}
}

//------------------------------------------------------------------------------

void mouseClick(GLFWwindow* window, int button, int action, int mods) {
	bool set = (action != GLFW_RELEASE);
	//TODO: mouse interaction with robot
	switch (button) {
		// left click pans and tilts
		case GLFW_MOUSE_BUTTON_LEFT:
			fRotPanTilt = set;
			// NOTE: the code below is recommended but doesn't work well
			// if (fRotPanTilt) {
			// 	// lock cursor
			// 	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			// } else {
			// 	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			// }
			break;
		// if right click: don't handle. this is for menu selection
		case GLFW_MOUSE_BUTTON_RIGHT:
			//TODO: menu
			break;
		// if middle click: don't handle. doesn't work well on laptops
		case GLFW_MOUSE_BUTTON_MIDDLE:
			break;
		default:
			break;
	}
}
