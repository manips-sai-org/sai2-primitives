// This example application loads a URDF world file and simulates two robots
// with physics and contact in a Dynamics3D virtual world. A graphics model of it is also shown using 
// Chai3D.

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include "Sai2Model.h"
#include "Sai2Simulation.h"
#include "Sai2Graphics.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include <dynamics3d.h>

#include <thread>
#include <mutex>
#include <iostream>
#include <string>
#include <GLFW/glfw3.h> //must be loaded after loading opengl/glew as part of Sai2Graphics
#include <signal.h>

using namespace std;
using namespace cv;

std::mutex mtx; // synchronize redis read & write

const string world_file = "resources/world.urdf";
const string robot_file = "resources/kuka_iiwa.urdf";
const string robot_name = "Kuka-IIWA";
const string object_file = "resources/cube.urdf";
const string object_name = "Object";
const string camera_name = "camera";
const string cube_ee_link = "object";
const string ee_link = "link6";
const Eigen::Vector3d cube_ee_pos = Eigen::Vector3d(0.0, 0.0, 0.0);;
const Eigen::Vector3d ee_pos = Eigen::Vector3d(0.0,0.0,0.08);

// redis keys:
// - read:
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::iiwaForceControl::iiwaBot::actuators::fgc";
// - write:
const std::string JOINT_ANGLES_KEY  = "sai2::iiwaForceControl::iiwaBot::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai2::iiwaForceControl::iiwaBot::sensors::dq";
const std::string SIM_TIMESTAMP_KEY = "sai2::iiwaForceControl::iiwaBot::simulation::timestamp";
const std::string OBJ_ENDEFF_POS = "sai2::pnp::obj_endeff_pos";
const std::string OBJ_ENDEFF_RMAT = "sai2::pnp::obj_endeff_rmat";

bool fSimulationRunning = false;
void sighandler(int){fSimulationRunning = false;}

// simulation loop
void simulation(Sai2Model::Sai2Model* robot, Sai2Model::Sai2Model* cube, Simulation::Sai2Simulation* sim);
unsigned long long sim_counter = 0;
// initialize window manager
GLFWwindow* glfwInitialize();

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

RedisClient redis_client;

int main() {
	cout << "Loading URDF world model file: " << world_file << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// start redis client
	HiredisServerInfo info;
	info.hostname_ = "127.0.0.1";
	info.port_ = 6379;
	info.timeout_ = { 1, 500000 }; // 1.5 seconds
	redis_client = RedisClient();
	redis_client.serverIs(info);

	// load graphics scene
	auto graphics = new Sai2Graphics::Sai2Graphics(world_file, false);
	Eigen::Vector3d camera_pos, camera_lookat, camera_vertical;
	graphics->getCameraPose(camera_name, camera_pos, camera_vertical, camera_lookat);

	// initialize opencv aruco detection
	auto camera = graphics->getCamera(camera_name);
	float marker_length = 0.1; // 10cm
	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    int dictionary_id = 10;
    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionary_id));

	// load simulation world
	auto sim = new Simulation::Sai2Simulation(world_file, false);
	sim->setCollisionRestitution(0);
	sim->setCoeffFrictionStatic(0.4);

	// load robots
	Eigen::Vector3d world_gravity = sim->_world->getGravity().eigen();
	auto robot = new Sai2Model::Sai2Model(robot_file, false, world_gravity, sim->getRobotBaseTransform(robot_name));

	sim->getJointPositions(robot_name, robot->_q);
	robot->updateModel();

	// load cube
	auto cube = new Sai2Model::Sai2Model(object_file, false, world_gravity, sim->getRobotBaseTransform(object_name));

	// load simulated force sensor
	// Eigen::Affine3d T_sensor = Eigen::Affine3d::Identity();
	// T_sensor.translation() = sensor_pos_in_link;
	// auto fsensor = new ForceSensorSim(robot_name, link_name, T_sensor, robot);
	// auto fsensor_display = new ForceSensorDisplay(fsensor, graphics);

	// initialize GLFW window
	GLFWwindow* window = glfwInitialize();

	double last_cursorx, last_cursory;

    // set callbacks
	glfwSetKeyCallback(window, keySelect);
	glfwSetMouseButtonCallback(window, mouseClick);

	// start the simulation thread first
	fSimulationRunning = true;
	thread sim_thread(simulation, robot, cube, sim);

    // while window is open:
    while (!glfwWindowShouldClose(window)) {
		// update kinematic models
		// robot->updateModel();

    	// fsensor_display->update();

		// update graphics. this automatically waits for the correct amount of time
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		graphics->updateGraphics(robot_name, robot);
		graphics->updateGraphics(object_name, cube);
		graphics->render(camera_name, width, height);

		// copy image from frame buffer
		// cout << "width: " << width << " height: " << height << endl;
		chai3d::cImagePtr image = chai3d::cImage::create();
		image->allocate(width, height);
		camera->copyImageBuffer(image);
		unsigned int image_size = image->getSizeInBytes();
		Mat cv_image_raw(height, width, CV_8UC3, image->getData());
		Mat cv_image_flip;
		Mat cv_image;
		flip(cv_image_raw, cv_image_flip, 0); // need to flip vertically to match opengl image
		cvtColor(cv_image_flip, cv_image, COLOR_BGR2RGB); // need to transform color spaces from RGB to BGR

		// convert openGL projection matrix to camera intrinsic matrix
		// https://www.opengl.org/discussion_boards/showthread.php/159764-intrinsic-camera-parameters
		auto proj_mat = camera->m_projectionMatrix;
		auto _m = proj_mat.m;
		double proj_mat_data[9] = {
			_m[0][0] * width / 2., 0., (_m[2][0] + 1.) * width / 2. - 0.5,
			0., _m[1][1] * height / 2., (_m[2][1] + 1.) * height / 2. - 0.5,
			0., 0., 1.
		};
		Mat camera_matrix = cv::Mat(3, 3, CV_64F, proj_mat_data);
		double dist_coef_data[4] = {0., 0., 0., 0.};
		Mat dist_coef = cv::Mat(4, 1, CV_32F, dist_coef_data);
		// cout << "projection matrix: " << proj_mat.str(3) << endl;
		// cout << "camera intrinsic matrix: " << endl;
		// cout << camera_matrix << endl;
		// cout << "camera distortion coefficient: " << endl;
		// cout << dist_coef << endl;

		// detect aruco marker
		vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        vector< Vec3d > rvecs, tvecs;

        // detect markers and estimate pose
        aruco::detectMarkers(cv_image, dictionary, corners, ids, detectorParams, rejected);
        if(ids.size() == 2) {
            aruco::estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coef, rvecs, tvecs);
	        // draw results
        	// find endeffector frame, object frame
        	int endeff_i = 0;
        	int obj_i = 1;
			if (ids[0] < ids[1]) {
				endeff_i = 1;
				obj_i = 0;	
			}
			Vec3d endeff_rvec = rvecs[endeff_i];
			Vec3d endeff_tvec = tvecs[endeff_i];
			Vec3d obj_rvec = rvecs[obj_i];
			Vec3d obj_tvec = tvecs[obj_i];
			aruco::drawAxis(cv_image, camera_matrix, dist_coef, endeff_rvec, endeff_tvec, marker_length * 2.0f);
			aruco::drawAxis(cv_image, camera_matrix, dist_coef, obj_rvec, obj_tvec, marker_length * 2.0f);
			Mat endeff_rmat;
			Mat obj_rmat;
        	Rodrigues(endeff_rvec, endeff_rmat);
        	Rodrigues(obj_rvec, obj_rmat);
			
			// calculate delta rotation in endeffector frame
        	Mat delta_rmat_ee = Mat(endeff_rmat.inv()) * obj_rmat;
        	Mat delta_rvec;
        	Rodrigues(delta_rmat_ee, delta_rvec);
        	// cout << "delta_rmat_ee: " << delta_rmat_ee << endl;
        	// cout << "delta_rvec: " << delta_rvec << endl;
        	// visualize delta rotation
        	Vec3d origin;
        	origin << -0.110225, 0.755015, 2.17658;
        	aruco::drawAxis(cv_image, camera_matrix, dist_coef, delta_rvec, origin, marker_length * 2.0f);
        	// convert to eigen matrix
        	Eigen::Matrix3d delta_rmat_eigen; 
        	cv2eigen(delta_rmat_ee, delta_rmat_eigen);
        	// mtx.lock();
        	// cout << "delta_rmat_eigen: " << delta_rmat_eigen << endl;
        	// redis_client.setEigenMatrixDerived(OBJ_ENDEFF_RMAT, delta_rmat_eigen);
        	// mtx.unlock();

        	// calculate delta position in endeffector frame
        	// cout << "obj_tvec: " << obj_tvec << endl;
        	// cout << "endeff_tvec: " << endeff_tvec << endl;
        	Vec3d delta_pos = obj_tvec - endeff_tvec;
        	// cout << "delta_pos: " << delta_pos << endl;
			auto delta_pos_ee = Mat(endeff_rmat.inv()) * Mat(delta_pos);
			// cout << "delta_pos_ee: " << delta_pos_ee << endl;
			// convert to eigen matrix
        	Eigen::Vector3d delta_pos_eff_eigen; 
        	cv2eigen(delta_pos_ee, delta_pos_eff_eigen);
        	// mtx.lock();
			// cout << "delta_pos_ee: " << delta_pos_ee << endl;
        	// redis_client.setEigenMatrixDerived(OBJ_ENDEFF_POS, delta_pos_eff_eigen);
        	// mtx.unlock();
        }

		Mat image_resized;
        resize(cv_image, image_resized, Size(), 0.5, 0.5);
		namedWindow("cv_image", CV_WINDOW_AUTOSIZE);
		imshow("cv_image", image_resized);
		// imwrite("./image.jpg", cv_image);
		// char key = (char)waitKey(0);
		// if(key == 27) break;
		image->erase();		

		glfwSwapBuffers(window);
		glFinish();

	    // poll for events
	    glfwPollEvents();

		// move scene camera as required
    	// graphics->getCameraPose(camera_name, camera_pos, camera_vertical, camera_lookat);
    	Eigen::Vector3d cam_depth_axis;
    	cam_depth_axis = camera_lookat - camera_pos;
    	cam_depth_axis.normalize();
    	Eigen::Vector3d cam_up_axis;
    	// cam_up_axis = camera_vertical;
    	// cam_up_axis.normalize();
    	cam_up_axis << 0.0, 0.0, 1.0; //TODO: there might be a better way to do this
	    Eigen::Vector3d cam_roll_axis = (camera_lookat - camera_pos).cross(cam_up_axis);
    	cam_roll_axis.normalize();
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

    // destroy context
    glfwDestroyWindow(window);

    // terminate
    glfwTerminate();

	return 0;
}

// void simulation(Sai2Model::Sai2Model* robot, Sai2Model::Sai2Model* cube, ForceSensorSim* fsensor, Simulation::Sai2Simulation* sim) {
void simulation(Sai2Model::Sai2Model* robot, Sai2Model::Sai2Model* cube, Simulation::Sai2Simulation* sim) {
	fSimulationRunning = true;

	// cube controller
	// Eigen::Vector2d plate_qd = Eigen::Vector2d::Zero();
	// Eigen::Vector2d plate_torques = Eigen::Vector2d::Zero();

	int dof = robot->dof();
	Eigen::VectorXd robot_torques = Eigen::VectorXd::Zero(dof);
	redis_client.setEigenMatrixDerived(JOINT_TORQUES_COMMANDED_KEY, robot_torques);

	// create a timer
	unsigned long long sim_counter = 0;
	double sim_freq = 2000.0;
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(sim_freq); 
	double last_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	// read initial object position w.r.t. end-effector
	robot->updateModel();
	Eigen::Affine3d base_ee_transform;
	robot->transformInWorld(base_ee_transform, ee_link, ee_pos);

	Eigen::Vector3d cube_pos;
	Eigen::Matrix3d cube_rmat;
	cube->updateModel();
	cube->positionInWorld(cube_pos, cube_ee_link, cube_ee_pos);
	cube->rotationInWorld(cube_rmat, cube_ee_link);
	Eigen::Vector3d cube_ee_pos = base_ee_transform.inverse() * cube_pos;
	// set cube position, rotation from model
	mtx.lock();
	// cout << "cube_pos: " << cube_pos << endl;
	cout << "initial cube_ee_pos: " << cube_ee_pos << endl;
	redis_client.setEigenMatrixDerived(OBJ_ENDEFF_POS, cube_ee_pos);
	redis_client.setEigenMatrixDerived(OBJ_ENDEFF_RMAT, cube_rmat);
	mtx.unlock();

	while (fSimulationRunning) {
		fTimerDidSleep = timer.waitForNextLoop();

		double time = sim_counter/sim_freq;

		// read torques from Redis
		mtx.lock();
		redis_client.getEigenMatrixDerived(JOINT_TORQUES_COMMANDED_KEY, robot_torques);
		mtx.unlock();
		sim->setJointTorques(robot_name, robot_torques);

		// update simulation by 1ms
		sim->integrate(1/sim_freq);

		// update kinematic models
		sim->getJointPositions(robot_name, robot->_q);
		sim->getJointVelocities(robot_name, robot->_dq);
		robot->updateModel();

		// write joint kinematics to redis
		mtx.lock();
		redis_client.setEigenMatrixDerived(JOINT_ANGLES_KEY, robot->_q);
		redis_client.setEigenMatrixDerived(JOINT_VELOCITIES_KEY, robot->_dq);
		redis_client.setCommandIs(SIM_TIMESTAMP_KEY, std::to_string(timer.elapsedTime()));
		mtx.unlock();

		sim_counter++;

		// // force sensor update
		// fsensor->update(sim);
		// fsensor->getForce(sensed_force);
		// fsensor->getMoment(sensed_moment);

		// // cube controller
		// sim->getJointPositions(object_name, cube->_q);
		// sim->getJointVelocities(object_name, cube->_dq);
		// cube->updateKinematics();

		// plate_qd(0) = 5.0/180.0*M_PI*sin(2*M_PI*0.12*time);
		// plate_qd(1) = 7.0/180.0*M_PI*sin(2*M_PI*0.08*time);

		// plate_torques = -1000.0*(cube->_q - plate_qd) - 75.0*cube->_dq;

		// sim->setJointTorques(object_name, plate_torques);

		// integrate forward
		// sim->integrate(0.0005);

		// sim_counter++;
	}

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Simulation Loop run time  : " << end_time << " seconds\n";
    std::cout << "Simulation Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Simulation Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}


//------------------------------------------------------------------------------
GLFWwindow* glfwInitialize() {
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
    GLFWwindow* window = glfwCreateWindow(windowW, windowH, "SAI2.0 - CS327a HW2", NULL, NULL);
	glfwSetWindowPos(window, windowPosX, windowPosY);
	glfwShowWindow(window);
    glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	return window;
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
