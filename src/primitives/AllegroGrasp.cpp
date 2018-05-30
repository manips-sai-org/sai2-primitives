/*
 * AllegroGrasp.cpp
 *
 *      This class creates a motion primitive for a redundant arm using a posori task and a joint task in its nullspace
 *
 *      Author: Mikael Jorda
 */

#include "AllegroGrasp.h"

using namespace std;

const string ALLEGRO_COMMAND = "allegro::command";
const string ALLEGRO_PRE_GRASP = "3";
const string ALLEGRO_GRASP = "4";
const string ALLEGRO_FORCE_GRASP = "5";

namespace Sai2Primitives
{


AllegroGrasp::AllegroGrasp(RedisClient* redis_client,
				   Sai2Model::Sai2Model* robot,
				   const std::string link_name,
                   const Eigen::Affine3d control_frame)
{
	_redis_client = redis_client;
	_robot = robot;
	_link_name = link_name;
	_control_frame = control_frame;
	_state = AllegroGrasp::State::APPROACH;

	_redundant_arm_motion = new RedundantArmMotion(robot, link_name, control_frame);
}

AllegroGrasp::AllegroGrasp(RedisClient* redis_client,
				   Sai2Model::Sai2Model* robot,
				   const std::string link_name,
                   const Eigen::Vector3d pos_in_link,
                   const Eigen::Matrix3d rot_in_link)
{
	_redis_client = redis_client;
	_robot = robot;
	_link_name = link_name;

	Eigen::Affine3d control_frame = Eigen::Affine3d::Identity();
	control_frame.linear() = rot_in_link;
	control_frame.translation() = pos_in_link;
	_control_frame = control_frame;
	_state = AllegroGrasp::State::APPROACH;

	_redundant_arm_motion = new RedundantArmMotion(robot, link_name, pos_in_link, rot_in_link);
}

AllegroGrasp::~AllegroGrasp()
{
	delete _redundant_arm_motion;
	_redundant_arm_motion = NULL;
}

void AllegroGrasp::updatePrimitiveModel()
{
	_redundant_arm_motion->updatePrimitiveModel();
}

void AllegroGrasp::computeTorques(Eigen::VectorXd& torques)
{
	_redundant_arm_motion->computeTorques(torques);
}

void AllegroGrasp::start(const Eigen::Vector3d obj_pos_base,
			   			 const Eigen::Matrix3d obj_rmat_base) {

	_state = AllegroGrasp::State::APPROACH;
	_iter = 0;
	_state_iter = 0;
	// TODO: move 0.2 meter in z axis
	_pos_pre_grasp = obj_pos_base + obj_rmat_base * Eigen::Vector3d(0., 0., -0.25);
	_pos_grasp = obj_pos_base + obj_rmat_base * Eigen::Vector3d(0., 0., -0.15);
	_rot_obj = obj_rmat_base;
	
	// set desired position, rotation to current position, rotation
	Eigen::Vector3d pos_current;
	Eigen::Matrix3d rmat_current;
	_robot->positionInWorld(pos_current, _link_name, _control_frame.translation());
	_robot->rotationInWorld(rmat_current, _link_name);
	_redundant_arm_motion->_desired_position = pos_current;
	_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
	// TODO: set desired rotation
}

void AllegroGrasp::step() {

	double velocity = 0.05;
	double control_freq = 1000;
	_iter++;
	Eigen::Vector3d pos_current;
	Eigen::Matrix3d rmat_current;
	Eigen::Vector3d pos_goal;
	Eigen::Matrix3d rmat_goal;
	Eigen::Vector3d direction;
	Eigen::Vector3d delta_pos;
	Eigen::Vector3d step_pos;
	Eigen::Vector3d pos_to_goal;

	switch(_state) {
		case AllegroGrasp::State::APPROACH:
			_robot->positionInWorld(pos_current, _link_name, _control_frame.translation());
			_robot->rotationInWorld(rmat_current, _link_name);
			
			pos_goal = _pos_pre_grasp;
			rmat_goal = _rot_obj;

			// cout << "pos_current: " << pos_current << endl;
			// cout << "pos_goal: " << pos_goal << endl;
			delta_pos = pos_goal - pos_current;
			direction = delta_pos / delta_pos.norm();
			step_pos = direction * velocity / control_freq;
			_redundant_arm_motion->_desired_position += step_pos;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
			
			// less than 1mm away from goal, advance to next state
			if (_state_iter <= 0) {
				cout << "state: approach" << endl;
			}
			_state_iter++;
			if (delta_pos.norm() < 0.001) {
				_state_iter = 0;
				_state = AllegroGrasp::State::PREGRASP;
			}
			break;
		case AllegroGrasp::State::PREGRASP:
			// set allegro::command 3 // cube
			// set allegro::command 6 // cylinder
			if (_state_iter <= 0) {
				// first iteration
				cout << "state: pre-grasp" << endl;
				_redis_client->set(ALLEGRO_COMMAND, ALLEGRO_PRE_GRASP);
			}
			_state_iter++;
			// wait for two seconds
			if (_state_iter >= control_freq * 2) {
				_state_iter = 0;
				_state = AllegroGrasp::State::LOWER;
			}
			break;
		case AllegroGrasp::State::LOWER:
			_robot->positionInWorld(pos_current, _link_name, _control_frame.translation());
			_robot->rotationInWorld(rmat_current, _link_name);
			
			pos_goal = _pos_grasp;
			rmat_goal = _rot_obj;

			// cout << "pos_current: " << pos_current << endl;
			// cout << "pos_goal: " << pos_goal << endl;
			delta_pos = pos_goal - pos_current;
			direction = delta_pos / delta_pos.norm();
			step_pos = direction * velocity / control_freq;
			_redundant_arm_motion->_desired_position += step_pos;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
			
			if (_state_iter <= 0) {
				// first iteration
				cout << "state: lower" << endl;
			}
			_state_iter++;
			// less than 1mm away from goal, advance to next state
			if (delta_pos.norm() < 0.001) {
				_state_iter = 0;
				_state = AllegroGrasp::State::PAUSE;
			}
			break;
		case AllegroGrasp::State::PAUSE:
			if (_state_iter <= 0) {
				cout << "state: pause" << endl;
			}
			_state_iter++;
			// wait for two seconds
			if (_state_iter >= control_freq * 2) {
				_state_iter = 0;
				_state = AllegroGrasp::State::GRASP;
			}
			break;
		case AllegroGrasp::State::GRASP:
			// set allegro::command 4 // cube
			// set allegro::command 7 // cylinder
			if (_state_iter <= 0) {
				// first iteration
				cout << "state: grasp" << endl;
				_redis_client->set(ALLEGRO_COMMAND, ALLEGRO_GRASP);
			}
			_state_iter++;
			// wait for two seconds
			if (_state_iter >= control_freq * 2) {
				_state_iter = 0;
				_state = AllegroGrasp::State::FORCE_GRASP;
			}
			break;
		case AllegroGrasp::State::FORCE_GRASP:
			// set allegro::command 5 // cube
			// set allegro::command 8 // cylinder
			if (_state_iter <= 0) {
				// first iteration
				cout << "state: force grasp" << endl;
				_redis_client->set(ALLEGRO_COMMAND, ALLEGRO_FORCE_GRASP);
			}
			_state_iter++;
			// wait for two seconds
			if (_state_iter >= control_freq * 2) {
				_state_iter = 0;
				_state = AllegroGrasp::State::DONE;
			}
			break;
		default:
			break;
	}
}

void AllegroGrasp::enableGravComp()
{
	_gravity_compensation = true;
}

void AllegroGrasp::disbleGravComp()
{
	_gravity_compensation = false;
}

} /* namespace Sai2Primitives */

