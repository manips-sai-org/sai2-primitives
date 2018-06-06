/*
 * Example of a controller for a Kuka arm made with the surface surface alignment primitive
 *
 */

#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <math.h>
#include <Eigen/Dense>

#include "redis/RedisClient.h"
#include "Sai2Model.h"
#include "Sai2Graphics.h"
#include "Sai2Simulation.h"
#include <dynamics3d.h>

// #include "primitives/RedundantArmMotion.h"
// #include "primitives/SurfaceSurfaceAlignment.h"
#include "primitives/AllegroGrasp.h"
#include "primitives/PickAndPlace.h"
#include "timer/LoopTimer.h"
#include "force_sensor/ForceSensorSim.h"
#include "force_sensor/ForceSensorDisplay.h"

#include <signal.h>

bool runloop = true;
void sighandler(int sig)
{ runloop = false; }

using namespace std;

// std::mutex mtx; // synchronize redis read & write

unsigned long long controller_counter = 0;

const bool simulation = false;
// const bool simulation = false;

std::string JOINT_TORQUES_COMMANDED_KEY;
std::string JOINT_ANGLES_KEY;
std::string JOINT_VELOCITIES_KEY;
std::string EE_FORCE_SENSOR_KEY;
std::string OBJ_ENDEFF_POS;
std::string OBJ_ENDEFF_RMAT;

const string robot_file = "resources/kuka_iiwa.urdf";

// control link and position in link
const string ee_link = "link6";
const Eigen::Vector3d ee_pos = Eigen::Vector3d(0.088, 0., 0.02);
const Eigen::Vector3d sensor_ee_pos = Eigen::Vector3d(0.0,0.0,0.05);
Eigen::Vector3d sensed_force;
Eigen::Vector3d sensed_moment;

int main (int argc, char** argv) {
	if(simulation)
	{
		JOINT_TORQUES_COMMANDED_KEY = "sai2::iiwaForceControl::iiwaBot::actuators::fgc";
		JOINT_ANGLES_KEY  = "sai2::iiwaForceControl::iiwaBot::sensors::q";
		JOINT_VELOCITIES_KEY = "sai2::iiwaForceControl::iiwaBot::sensors::dq";
		EE_FORCE_SENSOR_KEY = "sai2::optoforceSensor::6Dsensor::force_moment";
		OBJ_ENDEFF_POS = "sai2::pnp::obj_endeff_pos";
		OBJ_ENDEFF_RMAT = "sai2::pnp::obj_endeff_rmat";
	}
	else
	{
		JOINT_TORQUES_COMMANDED_KEY = "sai2::KUKA_IIWA::actuators::fgc";
		JOINT_ANGLES_KEY  = "sai2::KUKA_IIWA::sensors::q";
		JOINT_VELOCITIES_KEY = "sai2::KUKA_IIWA::sensors::dq";
		EE_FORCE_SENSOR_KEY = "sai2::optoforceSensor::6Dsensor::force";
		OBJ_ENDEFF_POS = "sai2::pnp::obj_endeff_pos";
		OBJ_ENDEFF_RMAT = "sai2::pnp::obj_endeff_rmat";
	}

	// start redis client
	HiredisServerInfo info;
	info.hostname_ = "127.0.0.1";
	info.port_ = 6379;
	info.timeout_ = { 1, 500000 }; // 1.5 seconds
	auto redis_client = RedisClient();
	redis_client.serverIs(info);

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);
	
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	
	// read from Redis
	redis_client.getEigenMatrixDerived(JOINT_ANGLES_KEY, robot->_q);
	redis_client.getEigenMatrixDerived(JOINT_VELOCITIES_KEY, robot->_dq);

	////////////////////////////////////////////////
	///    Prepare the joint controller        /////
	////////////////////////////////////////////////

	robot->updateModel();
	int dof = robot->dof();
	Eigen::VectorXd command_torques = Eigen::VectorXd::Zero(dof);

	Eigen::Affine3d control_frame_in_link = Eigen::Affine3d::Identity();
	control_frame_in_link.translation() = ee_pos;
	// Eigen::Affine3d sensor_frame_in_link = Eigen::Affine3d::Identity();
	// sensor_frame_in_link.translation() = ee_pos;

	// Motion arm primitive
	Sai2Primitives::AllegroGrasp* grasp_primitive = new Sai2Primitives::AllegroGrasp(&redis_client, robot, ee_link, control_frame_in_link);
	Sai2Primitives::PickAndPlace* pnp_primitive = new Sai2Primitives::PickAndPlace(&redis_client, robot, ee_link, control_frame_in_link);
	Eigen::VectorXd torques;
	bool pnp_started = false;
	// pnp_primitive->enableGravComp();

	// create a loop timer
	double control_freq = 1000;
	LoopTimer timer;
	timer.setLoopFrequency(control_freq);   // 1 KHz
	double last_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;
	timer.initializeTimer(1000000); // 1 ms pause before starting loop
	unsigned long long controller_counter = 0;

	// read object position, set by simviz / perception
	// read from Redis
	redis_client.getEigenMatrixDerived(JOINT_ANGLES_KEY, robot->_q);
	redis_client.getEigenMatrixDerived(JOINT_VELOCITIES_KEY, robot->_dq);
	robot->updateModel();

	// compute target position, based on vision pose estimation
	Eigen::Vector3d obj_endeff_pos = Eigen::Vector3d::Zero();
	Eigen::Matrix3d obj_endeff_rmat = Eigen::Matrix3d::Zero();
	Eigen::Vector3d endeff_base_pos = Eigen::Vector3d::Zero();
	Eigen::Matrix3d endeff_base_rmat = Eigen::Matrix3d::Zero();
	// mtx.lock();
	redis_client.getEigenMatrixDerived(OBJ_ENDEFF_POS, obj_endeff_pos);
	redis_client.getEigenMatrixDerived(OBJ_ENDEFF_RMAT, obj_endeff_rmat);
	// mtx.unlock();
	robot->position(endeff_base_pos, ee_link, ee_pos);
	robot->rotation(endeff_base_rmat, ee_link);
	Eigen::Matrix3d obj_base_rmat = obj_endeff_rmat * endeff_base_rmat;
	// cout << "obj_base_rmat: " << obj_base_rmat << endl;
	// cout << "obj_endeff_pos: " << obj_endeff_pos << endl;

	// transform from endeffector to hand marker
	Eigen::Affine3d marker_endeff_transform = Eigen::Affine3d::Identity();
	Eigen::Matrix3d marker_endeff_rmat = Eigen::Matrix3d();
	marker_endeff_rmat << 0., 1., 0., 1., 0., 0., 0., 0., -1.;
	marker_endeff_transform.linear() = marker_endeff_rmat;
	marker_endeff_transform.translation() = Eigen::Vector3d::Zero();

	Eigen::Vector3d obj_base_pos = endeff_base_pos + endeff_base_rmat * marker_endeff_transform * obj_endeff_pos;
	cout << "endeff_base_pos: " << endeff_base_pos << endl;
	cout << "obj_endeff_pos: " << obj_endeff_pos << endl;
	cout << "marker_endeff_transform * obj_endeff_pos: " << marker_endeff_transform * obj_endeff_pos << endl;
	cout << "endeff_base_rmat * marker_endeff_transform * obj_endeff_pos: " << endeff_base_rmat * marker_endeff_transform * obj_endeff_pos << endl;
	cout << "obj_base_pos: " << obj_base_pos << endl;
	grasp_primitive->start(obj_base_pos, obj_base_rmat);

	while (runloop) { //automatically set to false when simulation is quit
		fTimerDidSleep = timer.waitForNextLoop();

		// update time
		double curr_time = timer.elapsedTime();
		double loop_dt = curr_time - last_time;

		// read from Redis
		redis_client.getEigenMatrixDerived(JOINT_ANGLES_KEY, robot->_q);
		redis_client.getEigenMatrixDerived(JOINT_VELOCITIES_KEY, robot->_dq);
		robot->updateModel();

		// -------------------------------------------
		////////////////////////////// Compute joint torques
		double time = controller_counter / control_freq;

		// update sensed values (need to put them back in sensor frame)
		// Eigen::Matrix3d R_link;
		// robot->rotation(R_link, ee_link);
		// Eigen::Matrix3d R_sensor = R_link*sensor_frame_in_link.rotation();
		// pnp_primitive->updateSensedForceAndMoment(- R_sensor.transpose() * sensed_force, - R_sensor.transpose() * sensed_moment);

		// torques
		if (!grasp_primitive->done()) {
			grasp_primitive->updatePrimitiveModel();
			grasp_primitive->step();
			grasp_primitive->computeTorques(torques);
		} else if (!pnp_started) {
			// start pick and place
			pnp_started = true;
			const double delta_z = .1;
			Eigen::Matrix3d place_rmat = Eigen::Matrix3d();
			robot->rotation(place_rmat, ee_link);
			Eigen::Vector3d current_pos;
			robot->position(current_pos, ee_link, ee_pos);
			Eigen::Vector3d place_pos = Eigen::Vector3d(-0.2, 
						grasp_primitive->_pos_grasp(1), 
						grasp_primitive->_pos_grasp(2) - 0.005);
			pnp_primitive->start(delta_z, place_pos, place_rmat);
		} else {
			pnp_primitive->updatePrimitiveModel();
			pnp_primitive->step();
			pnp_primitive->computeTorques(torques);
		}

		//------ Final torques
		command_torques = torques;
		// command_torques.setZero();

		// -------------------------------------------
		redis_client.setEigenMatrixDerived(JOINT_TORQUES_COMMANDED_KEY, command_torques);
		
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
