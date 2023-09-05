/*
* This example implementes the classic haptic controllers available in "sai2-primitives/haptic_tasks" :
* An operational space position control is used to maintain the haptic device at its home position.
* An impedance-type haptic controller is used to remotely control the panda robot with haptic feedback.
* An example of plane haptic guidance is given. Any line or plane guidance can be set through the controller.
*
* The application loads a URDF world file and simulates the haptic teleoperation of the robot
* with physics and contact in a Dynamics3D virtual world. A graphics model of it is also shown using Chai3D.
*
* This example is designed to control an OMEGA.7 or a SIGMA.7 Force Dimension haptic device.
* Its extension to other haptic device is straightforward after updating the 'Haptic driver' which handles
* device communication and exhanges data structures between the device and the controller.
*/

#include "Sai2Model.h"
#include "Sai2Graphics.h"
#include "Sai2Simulation.h"
#include "Sai2Primitives.h"
#include <dynamics3d.h>
#include "timer/LoopTimer.h"
#include "force_sensor/ForceSensorSim.h"

#include <GLFW/glfw3.h> //must be loaded after loading opengl/glew

#include <iostream>
#include <string>

#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int){fSimulationRunning = false;}

using namespace std;
using namespace Eigen;
using namespace chai3d;

const string world_file = "./resources/world.urdf";
const string robot_file = "./resources/panda_arm_sim.urdf";
const string robot_name = "panda";
const string camera_name = "camera";
const string link_name = "link7"; //robot end-effector
// Set sensor frame transform in end-effector frame
Affine3d sensor_transform_in_link = Affine3d::Identity();
Vector3d sensor_pos_in_link;
Vector3d pos_in_link;

// Create simulation and control function
void simulation(Sai2Model::Sai2Model* robot, Simulation::Sai2Simulation* sim);
void control(Simulation::Sai2Simulation* sim);

// create graphics object from world
auto graphics = new Sai2Graphics::Sai2Graphics(world_file, true);
// add plane object for guidance representation
auto plane_object = new chai3d::cMesh();

// Open world simulation
auto sim = new Simulation::Sai2Simulation(world_file, false);

// callback to print glfw errors
void glfwError(int error, const char* description);
// callback when a key is pressed
void keySelect(GLFWwindow* window, int key, int scancode, int action, int mods);
// callback when a mouse button is pressed
void mouseClick(GLFWwindow* window, int button, int action, int mods);
// flags for scene camera movement
bool fTransXp = false;
bool fTransYp = false;
bool fTransXn = false;
bool fTransYn = false;
bool fTransZp = false;
bool fTransZn = false;
bool fshowCameraPose = false;
bool fRotPanTilt = false;
// flag for enabling/disabling remote task
bool fOnOffRemote = false;

//////////////////////////////////////////////////////////////
// Chai - Eigen transformation functions for haptic driver
cVector3d convertEigenToChaiVector( Eigen::Vector3d a_vec )
{
    double x = a_vec(0);
    double y = a_vec(1);
    double z = a_vec(2);
    return cVector3d(x,y,z);
}
Eigen::Vector3d convertChaiToEigenVector( cVector3d a_vec )
{
    double x = a_vec.x();
    double y = a_vec.y();
    double z = a_vec.z();
    return Eigen::Vector3d(x,y,z);
}
cMatrix3d convertEigenToChaiRotation( Eigen::Matrix3d a_mat )
{
    return cMatrix3d( a_mat(0,0), a_mat(0,1), a_mat(0,2), a_mat(1,0), a_mat(1,1), a_mat(1,2), a_mat(2,0), a_mat(2,1), a_mat(2,2) );
}
Eigen::Matrix3d convertChaiToEigenMatrix( cMatrix3d a_mat )
{
    Eigen::Matrix3d asdf;
    asdf << a_mat(0,0), a_mat(0,1), a_mat(0,2),
            a_mat(1,0), a_mat(1,1), a_mat(1,2),
            a_mat(2,0), a_mat(2,1), a_mat(2,2);
    return asdf;
}
///////////////////////////////////////////////////////////////
//// create a handler for the haptic device ////
auto handler = new cHapticDeviceHandler();
// a pointer to the current haptic device
 cGenericHapticDevicePtr hapticDevice;
// Info structure of the haptic device
cHapticDeviceInfo device_info;
Vector3d max_device_force;
Vector2d max_device_stiffness;
Vector2d max_device_damping;
// Haptic device commands
cVector3d _force_chai;
cVector3d _torque_chai;
double commanded_gripper_force_device;

//// Robot global variables /////
// sensed task force from robot interaction
Vector3d sensed_force = Vector3d::Zero();
Vector3d sensed_moment = Vector3d::Zero();
// robot joint data
VectorXd robot_joint_positions = Eigen::VectorXd::Zero(7);
VectorXd robot_joint_velocities = Eigen::VectorXd::Zero(7);
VectorXd robot_control_torques = Eigen::VectorXd::Zero(7);

//// Haptic guidance data ////
cVector3d plane_origin_point_robot_frame;
cMatrix3d plane_rotation_robot_frame;
bool plane_guidance = false;
int plane_counter = 0;

//////////////////////////////////////////////////////////////////////
// Definition of the state machine
#define HOME_POSITION			          0
#define IMPEDANCE_CONTROL           1
#define PLANE_GUIDANCE				      2

int state = HOME_POSITION;
unsigned long long controller_counter = 0;

//const bool inertia_regularization = true;
//int switch_state_counter = 500;

int main() {
  //Initialize haptic commands to zero
  _force_chai.set(0.0,0.0,0.0);
  _torque_chai.set(0.0,0.0,0.0);
  commanded_gripper_force_device = 0.0;

  // Set sensor frame and control point for the controllers
  sensor_pos_in_link = Eigen::Vector3d(0.0,0.0,0.117);
  sensor_transform_in_link.translation() = sensor_pos_in_link;
  sensor_transform_in_link.linear() = Matrix3d::Identity();
  pos_in_link = Vector3d(0.0,0.0,0.127);

	cout << "Loading URDF world model file: " << world_file << endl;

	// Find, open and calibrate haptic device
	handler->getDevice(hapticDevice, 0);
		if (NULL == hapticDevice)
		{
			cout << "No haptic device found. " << endl;
		}
		else
		{
			if(!hapticDevice->open())
			{
				cout << "could not open the haptic device" << endl;
			}
			if(!hapticDevice->calibrate())
			{
				cout << "could not calibrate the haptic device" << endl;
			}
			else
			{
				// read info from haptic device
				handler->getDeviceSpecifications(device_info, 0);
				max_device_force << device_info.m_maxLinearForce, device_info.m_maxAngularTorque, device_info.m_maxGripperForce;
				max_device_stiffness << device_info.m_maxLinearStiffness, device_info.m_maxAngularStiffness;
				max_device_damping << device_info.m_maxLinearDamping, device_info.m_maxAngularDamping;

				// Set zero force to the haptic device
				hapticDevice->setForceAndTorqueAndGripperForce(_force_chai, _torque_chai, commanded_gripper_force_device);
			}
		}

	// load graphics scene
	Eigen::Vector3d camera_pos, camera_lookat, camera_vertical;
	graphics->getCameraPose(camera_name, camera_pos, camera_vertical, camera_lookat);

  Eigen::Vector3d cam_depth_axis;
  cam_depth_axis = camera_lookat - camera_pos;
  cam_depth_axis.normalize();
  Eigen::Vector3d cam_up_axis;
  // cam_up_axis = camera_vertical;
  // cam_up_axis.normalize();
  cam_up_axis << 0.0, 0.0, 1.0; //TODO: there might be a better way to do this
  Eigen::Vector3d cam_roll_axis = (camera_lookat - camera_pos).cross(cam_up_axis);
  cam_roll_axis.normalize();
  // Adjust camera position in scene
  camera_pos = camera_pos + 0.8*cam_depth_axis;
  camera_lookat = camera_lookat + 0.8*cam_depth_axis;
  camera_pos = camera_pos - 0.2*cam_roll_axis;
  camera_lookat = camera_lookat - 0.2*cam_roll_axis;

	// load simulation world
	sim->setCollisionRestitution(0);
	sim->setCoeffFrictionStatic(0.1);

	// load robots
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	// read joint positions, velocities, update model
	sim->getJointPositions(robot_name, robot->_q);
	sim->getJointVelocities(robot_name, robot->_dq);
	robot->updateKinematics();

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

	// Run simulation and control threads
	fSimulationRunning = true;
	thread sim_thread(simulation, robot, sim);
	thread control_thread(control, sim);

	// while window is open:
	while (!glfwWindowShouldClose(window))
	{
		// update graphics. this automatically waits for the correct amount of time
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		graphics->updateGraphics(robot_name, robot);
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


		Eigen::Vector3d cam_lookat_axis = camera_lookat;
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
			Eigen::Matrix3d m_tilt; m_tilt = Eigen::AngleAxisd(azimuth, -cam_roll_axis);
			camera_pos = camera_lookat + m_tilt*(camera_pos - camera_lookat);
			Eigen::Matrix3d m_pan; m_pan = Eigen::AngleAxisd(compass, -cam_up_axis);
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
////// Simulation thread //////
//------------------------------------------------------------------------------
void simulation(Sai2Model::Sai2Model* robot, Simulation::Sai2Simulation* sim) {

	// create a timer
	Sai2Common::LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(1000);
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs
	double last_time = start_time;

	unsigned long long simulation_counter = 0;

	// Add force sensor to the end-effector
	auto force_sensor = new ForceSensorSim(robot_name, link_name, sensor_transform_in_link, robot);

	while (fSimulationRunning) {
		fTimerDidSleep = timer.waitForNextLoop();

    // Add plane representation while haptic guidance is activated
		if (plane_guidance && plane_counter==0) {
			// create the plane, set the color and transparency
			chai3d::cCreatePlane(plane_object, 3.0, 3.0, plane_origin_point_robot_frame, plane_rotation_robot_frame);
			plane_object->m_material->setColorf(0.69, 0.93, 0.93);
			plane_object->setTransparencyLevel(0.3, false, false, false);

			// add the plane to the world
			graphics->_world->addChild(plane_object);

			// increment the plane counter so we don't break into this statement continuously
			plane_counter++;
		}
    else if (!plane_guidance && plane_counter>=1)
    {
      // remove the plane from visualization and make the pointer null
      plane_object->~cMesh();
      plane_object = NULL;

      // add new plane object for guidance representation
      plane_object = new cMesh();

      // reset plane counter
      plane_counter = 0;
    }

		// set command robot torques to simulation
		sim->setJointTorques(robot_name, robot_control_torques);

		// integrate forward
		double curr_time = timer.elapsedTime();
		double loop_dt = curr_time - last_time;
		sim->integrate(loop_dt);

		// read joint positions, velocities, update model
		sim->getJointPositions(robot_name, robot->_q);
		sim->getJointVelocities(robot_name, robot->_dq);
		robot->updateKinematics();

		// copy joint position/velocity values to global variables
		robot_joint_positions = robot->_q;
		robot_joint_velocities = robot->_dq;

		// read end-effector task forces from the force sensor simulation
		force_sensor->update(sim);
		force_sensor->getForce(sensed_force);
		force_sensor->getMoment(sensed_moment);

		//update last time
		last_time = curr_time;

		simulation_counter++;
	}

	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Simulation Loop run time  : " << end_time << " seconds\n";
	std::cout << "Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
	std::cout << "Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}



//------------------------------------------------------------------------------
////// Control thread //////
//------------------------------------------------------------------------------
void control(Simulation::Sai2Simulation* sim)
{
	// position of robots in world
	Eigen::Affine3d robot_pose_in_world = Affine3d::Identity();
	robot_pose_in_world.linear() = Matrix3d::Identity();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load robot
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	robot->_q = robot_joint_positions;
	robot->_dq = robot_joint_velocities;
	robot->updateModel();

	// Create robot joint and operational space controllers
	//----------------------------------------------------------------------------------------
  // Commanded, coriolis and gravity torque vector for the robot
  VectorXd command_torques = VectorXd::Zero(robot->dof());
  VectorXd coriolis_torques = VectorXd::Zero(robot->dof());
  VectorXd gravity_torques = VectorXd::Zero(robot->dof());
	// Joint Task controller (manage panda robot nullspace - posture)
	auto joint_task = new Sai2Primitives::JointTask(robot);
	MatrixXd N_prec = MatrixXd::Identity(robot->dof(), robot->dof());
	VectorXd joint_task_torques = VectorXd::Zero(robot->dof());
	// Controller gains
  joint_task->_kp = 250.0;
	joint_task->_kv = 18.0;
	joint_task->_ki = 35.0;
	// Define goal posture as current position
	VectorXd goal_posture(robot->dof());
	goal_posture = joint_task->_current_position;

	// MotionForceTask controller (manage the control in position of panda robot)
	auto posori_task = new Sai2Primitives::MotionForceTask(robot, link_name, pos_in_link);
	VectorXd posori_task_torques = VectorXd::Zero(robot->dof());
	// Controller gains
	posori_task->_kp_pos = 150.0;
	posori_task->_kv_pos = 12.0;
	posori_task->_kp_ori = 200.0;
	posori_task->_kv_ori = 15.0;
	// Velocity saturation
	#ifndef USING_OTG
		posori_task->_use_velocity_saturation_flag = true;
		posori_task->_linear_saturation_velocity = 0.8;
		posori_task->_angular_saturation_velocity = 90.0/180.0*M_PI;
	#endif

  // Define sensor frame and position on end-effector
  posori_task->setForceSensorFrame(link_name, sensor_transform_in_link);

	// Copy current robot position to define task center
	Vector3d centerPos_rob = posori_task->_current_position;
	Matrix3d centerRot_rob = posori_task->_current_orientation;
	// Rotation Matrix from world to robot frames
	Matrix3d transformDev_Rob = robot_pose_in_world.linear();

  // Sensor frame rotation in world
  Matrix3d R_sensor = Matrix3d::Identity();
  // Sensed task force at robot control point
  Vector3d sensed_force_control_point;
  Vector3d sensed_torque_control_point;

	// Create haptic device position and impedance-type controllers
	//----------------------------------------------------------------------------------------
	// Haptic device data
	cVector3d _current_position_device_chai;
	cMatrix3d _current_rotation_device_chai;
	cVector3d _current_trans_velocity_device_chai;
	cVector3d _current_rot_velocity_device_chai;
	double _current_gripper_position;
	double _current_gripper_velocity;
	// Haptic device commands
	Vector3d commanded_force_device;
	Vector3d commanded_torque_device;
  commanded_force_device.setZero();
  commanded_torque_device.setZero();

	// State of gripper switch
	bool gripper_state = false;
  bool gripper_state_prev = false;
  bool isPressed = false;
  // Intialize guidance plane parameters
  Vector3d plane_origin_point;
  Vector3d plane_normal_vec = Vector3d(1.0, 1.0, 1.0);
  // compute guidance plane rotation matrix from floor for vizualization
  Vector3d FloorAxis;			// world plane normal vector
  FloorAxis << 0.0, 0.0, 1.0;
  FloorAxis = transformDev_Rob * FloorAxis;		// floor axis in robot space
  Vector3d normal_vec; //guidance plane normale vector in robot space
  normal_vec = transformDev_Rob * plane_normal_vec;
  // plane rotation matrix and position in chai3d vector/matrix form
  plane_rotation_robot_frame = convertEigenToChaiRotation((Quaterniond().setFromTwoVectors(FloorAxis, normal_vec)).toRotationMatrix());

	// Define center position of the haptic device workspace
	Vector3d HomePos_op;
	HomePos_op << 0.0, 0.01, 0.0;
	Matrix3d HomeRot_op;
	HomeRot_op.setIdentity();
  double home_gripper_pos = 10*M_PI/180.0;

	// Position control (to maintain haptic device at home position)
	auto haptic_pos_ctrl = new Sai2Primitives::PositionControl(max_device_force, max_device_stiffness, max_device_damping);

	// Set desired position to device center
	haptic_pos_ctrl->setDesiredPosition(HomePos_op, HomeRot_op, home_gripper_pos);
	// Set device center to check workspace limits
	haptic_pos_ctrl->setDeviceCenter(HomePos_op, HomeRot_op);
	// Activate command saturation to workspace limits
  haptic_pos_ctrl->setWorkspaceLimits(true, 0.05, 90*M_PI/180.0);

	// Impedance-type control for haptic teleoperation
	auto haptic_teleop = new Sai2Primitives::ImpedanceControl(centerPos_rob, centerRot_rob,
									max_device_force, max_device_stiffness, max_device_damping, transformDev_Rob);
  // Set device center to compute robot relative motion command
	haptic_teleop->setDeviceCenter(HomePos_op, HomeRot_op);
  // Set a scaling factor in translation and rotation
  double trans_scaling_factor = 2.0;
  haptic_teleop->setScalingFactors(trans_scaling_factor, 1.0);
  // Add a reduction factor from sensed force to haptic feedback
  haptic_teleop->setReductionFactorForceFeedback(1/1.8, 1/8.0, 1.0);
  // Activate filtering of force sensor data with default cut-off frequency
  haptic_teleop->setFilterOn(true);
  // Activate virtual feedback of the device workspace limits
  haptic_teleop->setWorkspaceLimits(true, 0.05, 90*M_PI/180.0);

  std::cout << "Device and robot position-controlled at workspace center" << std::endl;
  std::cout << "Press gripper to switch to haptic teleoperation" << std::endl;

	// create a timer
	Sai2Common::LoopTimer timer;
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


		// read robot data from simulation thread
    robot->_q = robot_joint_positions;
  	robot->_dq = robot_joint_velocities;
  	// Update robot model
		robot->updateModel();
    // Get coriolis and gravity torques from model
		robot->coriolisForce(coriolis_torques);
		robot->gravityVector(gravity_torques);

    // update sensor frame rotation in world frame
    robot->rotation(R_sensor, link_name);
    R_sensor = R_sensor*sensor_transform_in_link.rotation();
    // Send sensed force to robot controller in sensor frame
		posori_task->updateSensedForceAndMoment(-R_sensor.transpose()*sensed_force, -R_sensor.transpose()*sensed_moment);
		// Send sensed force at control point in world frame to haptic controller
		sensed_force_control_point = -posori_task->_sensed_force;
		sensed_torque_control_point = -posori_task->_sensed_moment;

    haptic_teleop->updateRobotSensedForce(sensed_force_control_point, sensed_torque_control_point);

  	// read haptic device position and velocity
  	hapticDevice->getPosition(_current_position_device_chai);
  	hapticDevice->getLinearVelocity(_current_trans_velocity_device_chai);
  	hapticDevice->getRotation(_current_rotation_device_chai);
  	hapticDevice->getAngularVelocity(_current_rot_velocity_device_chai);
  	hapticDevice->getGripperAngleRad(_current_gripper_position);
  	hapticDevice->getGripperAngularVelocity(_current_gripper_velocity);

  	// Update device position and velocity to controllers
  	haptic_pos_ctrl->updateDevicePosition(convertChaiToEigenVector(_current_position_device_chai), convertChaiToEigenMatrix(_current_rotation_device_chai), _current_gripper_position);
  	haptic_pos_ctrl->updateDeviceVelocity(convertChaiToEigenVector(_current_trans_velocity_device_chai), convertChaiToEigenVector(_current_rot_velocity_device_chai), _current_gripper_velocity);
    haptic_teleop->updateDevicePosition(convertChaiToEigenVector(_current_position_device_chai), convertChaiToEigenMatrix(_current_rotation_device_chai), _current_gripper_position);
    haptic_teleop->updateDeviceVelocity(convertChaiToEigenVector(_current_trans_velocity_device_chai), convertChaiToEigenVector(_current_rot_velocity_device_chai), _current_gripper_velocity);

    // Use the haptic device gripper as a switch and update the gripper state
    haptic_teleop->useGripperAsSwitch(gripper_state, commanded_gripper_force_device);
    // This function is also available in the haptic position controller if needed
    //haptic_pos_ctrl->useGripperAsSwitch(gripper_state, commanded_gripper_force_device);

    switch (state)
    {
       case HOME_POSITION:
       // Maintain robot in goal posture
       joint_task->_desired_position = goal_posture;
       // update tasks model and priority
 			 N_prec.setIdentity();
 			 joint_task->updateTaskModel(N_prec);

 			 // Adjust mass matrix to increase wrist desired stiffness
 		 	 for(int i=4 ; i<7 ; i++)
 		 	 {
 				    robot->_M(i,i) += 0.07;
 			 }
 			 // Compute robot torques
 			 joint_task->computeTorques(joint_task_torques);
 			 command_torques = joint_task_torques + coriolis_torques;

       // Compute haptic commands to maintain device at the desired position (device center)
       haptic_pos_ctrl->computeCommands(commanded_force_device, commanded_torque_device);

       // The desired gripper force to maintain gripper at home_gripper_pos can be computed with :
       //haptic_pos_ctrl->computeGripperCommands(commanded_gripper_force_device);
       // Note : in this example the gripper force is rather computed to use it as a switch

       // Switch to impedance-type haptic teleoperation at gripper switch
       if(haptic_pos_ctrl->isInPos && gripper_state && (joint_task->_desired_position - joint_task->_current_position).norm() < 0.2)
       {
         // Reinitialize controllers
         joint_task->reInitializeTask();
         posori_task->reInitializeTask();

         // reduce joint controller gains
         joint_task->_kp = 50.0;
         joint_task->_kv = 14.0;
         joint_task->_ki = 0.0;

         // copy current device position as device center
         HomePos_op = convertChaiToEigenVector(_current_position_device_chai);
         HomeRot_op = convertChaiToEigenMatrix(_current_rotation_device_chai);
         haptic_teleop->setDeviceCenter(HomePos_op, HomeRot_op);

         // copy current robot position as task center
         centerPos_rob = posori_task->_current_position;
         centerRot_rob = posori_task->_current_orientation;
         haptic_teleop->setRobotCenter(centerPos_rob, centerRot_rob);

         commanded_force_device.setZero();
         commanded_torque_device.setZero();

         gripper_state_prev = gripper_state;

         std::cout << "Press gripper to activate/desactivate haptic guidance plane" << std::endl;

         state = IMPEDANCE_CONTROL;
       }
       break;

       case IMPEDANCE_CONTROL:

       // Compute haptic commands and send desired position to robot controller
       haptic_teleop->computeCommands(posori_task->_desired_position, posori_task->_desired_orientation, commanded_force_device, commanded_torque_device);

       // update robot tasks' model and priority
 			 N_prec.setIdentity();
 			 posori_task->updateTaskModel(N_prec);
 			 N_prec = posori_task->_N;
 			 joint_task->updateTaskModel(N_prec);

       // Compute commanded robot torques
 			 posori_task->computeTorques(posori_task_torques);
 			 joint_task->computeTorques(joint_task_torques);
 			 command_torques = posori_task_torques + joint_task_torques + coriolis_torques;

       // button code to activate/desactivate guidance plane
 			 if(gripper_state && gripper_state != gripper_state_prev) //if button pushed and no change recorded yet
 			 {
 				    // if button pushed
 				       isPressed = true;
 			 }
       else
       {
 				    isPressed = false;
 			 }
 			 gripper_state_prev = gripper_state;

       if (isPressed)
       {
          // copy current device position as origin point of plane guidance
          plane_origin_point = convertChaiToEigenVector(_current_position_device_chai);
          // transfer plane point position from device workspace to the robot space for vizualization
     			plane_origin_point_robot_frame = convertEigenToChaiVector(trans_scaling_factor * transformDev_Rob * plane_origin_point + (centerPos_rob - HomePos_op));
          // activate plane guidance
          plane_guidance = true;
          // Constraint motion in a plane through haptic guidance
          haptic_teleop->setGuidancePlane(plane_guidance, plane_origin_point, plane_normal_vec);
          // Add plane guidance to haptic controller
          state = PLANE_GUIDANCE;
       }
       break;

        case PLANE_GUIDANCE:
        // Compute haptic commands and send desired position to robot controller with haptic guidance
        haptic_teleop->computeCommands(posori_task->_desired_position, posori_task->_desired_orientation, commanded_force_device, commanded_torque_device);

        // update robot tasks' model and priority
        N_prec.setIdentity();
        posori_task->updateTaskModel(N_prec);
        N_prec = posori_task->_N;
        joint_task->updateTaskModel(N_prec);

        // Compute commanded robot torques
        posori_task->computeTorques(posori_task_torques);
        joint_task->computeTorques(joint_task_torques);
        command_torques = posori_task_torques + joint_task_torques + coriolis_torques;

        // button code to activate/desactivate guidance plane
        if(gripper_state && gripper_state != gripper_state_prev) //if button pushed and no change recorded yet
        {
         // if button pushed
         isPressed = true;
        }
        else
        {
         isPressed = false;
        }
        gripper_state_prev = gripper_state;

        if (isPressed)
        {
          // activate plane guidance
          plane_guidance = false;
          // Constraint motion in a plane through haptic guidance
          haptic_teleop->setGuidancePlane(plane_guidance);
          // Switch back to classic impedance-type bilateral control
          state = IMPEDANCE_CONTROL;
        }
        break;

        default:
        command_torques.setZero(robot->dof());
        commanded_force_device.setZero();
        commanded_torque_device.setZero();
    }

		// send control commands to robot simulation
		robot_control_torques = command_torques;

    // Transform haptic command to chai formalism
    _force_chai = convertEigenToChaiVector(commanded_force_device);
    _torque_chai = convertEigenToChaiVector(commanded_torque_device);
    // _force_chai.set(0.0,0.0,0.0);
    // _torque_chai.set(0.0,0.0,0.0);

    // send haptic command to device
    hapticDevice->setForceAndTorqueAndGripperForce(_force_chai, _torque_chai, commanded_gripper_force_device);

		prev_time = current_time;
		controller_counter++;
	}

  command_torques.setZero(robot->dof());
  commanded_force_device.setZero();
  commanded_torque_device.setZero();
  commanded_gripper_force_device = 0.0;
  _force_chai.set(0.0,0.0,0.0);
  _torque_chai.set(0.0,0.0,0.0);

  // send zero torque to the robot
  robot_control_torques = command_torques;

  // Set zero force to the haptic device
  hapticDevice->setForceAndTorqueAndGripperForce(_force_chai, _torque_chai, commanded_gripper_force_device);

  // close haptic device
  hapticDevice->close();

	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
	std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
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
