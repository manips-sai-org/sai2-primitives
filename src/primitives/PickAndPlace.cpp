/*
 * PickAndPlace.cpp
 *
 *      This class creates a motion primitive for a redundant arm using a posori task and a joint task in its nullspace
 *
 *      Author: Mikael Jorda
 */

#include "PickAndPlace.h"

using namespace std;

const string ALLEGRO_COMMAND = "allegro::command";
const string ALLEGRO_UNCONTROLLED = "o";

namespace Sai2Primitives
{


PickAndPlace::PickAndPlace(RedisClient* redis_client,
				   Sai2Model::Sai2Model* robot,
				   const std::string link_name,
                   const Eigen::Affine3d control_frame)
{
	_redis_client = redis_client;
	_robot = robot;
	_link_name = link_name;
	_control_frame = control_frame;
	_state = PickAndPlace::State::PICK;

	_redundant_arm_motion = new RedundantArmMotion(robot, link_name, control_frame);
}

PickAndPlace::PickAndPlace(RedisClient* redis_client,
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
	_state = PickAndPlace::State::PICK;

	_redundant_arm_motion = new RedundantArmMotion(robot, link_name, pos_in_link, rot_in_link);
}

PickAndPlace::~PickAndPlace()
{
	delete _redundant_arm_motion;
	_redundant_arm_motion = NULL;
}

void PickAndPlace::updatePrimitiveModel()
{
	_redundant_arm_motion->updatePrimitiveModel();
}

void PickAndPlace::computeTorques(Eigen::VectorXd& torques)
{
	_redundant_arm_motion->computeTorques(torques);
}

void PickAndPlace::start(const double delta_z,
					   const Eigen::Vector3d pos_place,
					   const Eigen::Matrix3d rot_place) {

	_state = PickAndPlace::State::PICK;
	_robot->position(_pos_pick, _link_name, _control_frame.translation());
	_robot->rotation(_rot_pick, _link_name);
	_redundant_arm_motion->_desired_position = _pos_pick;
	_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
	_redundant_arm_motion->_posori_task->_kp_pos = 20.;
	_redundant_arm_motion->_posori_task->_kv_pos = 9.;
	_redundant_arm_motion->_posori_task->_kp_ori = 20.;
	_redundant_arm_motion->_posori_task->_kv_ori = 9.;
	_redundant_arm_motion->_joint_task->_kp = 5.;
	_redundant_arm_motion->_joint_task->_kv = 4.;
	_delta_z = delta_z;
	_pos_place = pos_place;
	_rot_place = rot_place;
	_iter = 0;
	_state_iter = 0;
}

void PickAndPlace::step() {

	double velocity = 0.05;
	double control_freq = 1000;
	_iter++;
	Eigen::Vector3d pos_current;
	Eigen::Vector3d pos_goal;
	Eigen::Vector3d step;
	Eigen::Vector3d direction;
	Eigen::Vector3d delta_pos;
	Eigen::Vector3d pos_to_goal;
	switch(_state) {
		case PickAndPlace::State::PICK:
			_robot->position(pos_current, _link_name, _control_frame.translation());
			pos_goal = PickAndPlace::get_pos_move_start();

			if (_state_iter <= 0) {
				cout << "pnp state: pick" << endl;
				delta_pos = pos_goal - pos_current;
				direction = delta_pos / delta_pos.norm();
				_step_pos = direction * velocity / control_freq;
			}
			
			// cout << "_step_pos: " << _step_pos << endl;
			_redundant_arm_motion->_desired_position += _step_pos;
			// cout << "_redundant_arm_motion->_desired_position: " << _redundant_arm_motion->_desired_position << endl;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
			// cout << "delta_pos: " << delta_pos << endl;
			pos_to_goal = pos_goal - _redundant_arm_motion->_desired_position;

			_state_iter++;
			if (pos_to_goal.norm() < 0.01) {
				_state_iter = 0;
				_state = PickAndPlace::State::MOVE;
			}
			break;
		case PickAndPlace::State::MOVE:
			_robot->position(pos_current, _link_name, _control_frame.translation());
			pos_goal = PickAndPlace::get_pos_move_end();
			
			if (_state_iter <= 0) {
				cout << "pnp state: move" << endl;
				_redundant_arm_motion->_desired_position = PickAndPlace::get_pos_move_start();
				delta_pos = pos_goal - PickAndPlace::get_pos_move_start();
				direction = delta_pos / delta_pos.norm();
				_step_pos = direction * velocity / control_freq;
			}
			
			_redundant_arm_motion->_desired_position += _step_pos;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
			pos_to_goal = pos_goal - _redundant_arm_motion->_desired_position;

			_state_iter++;
			if (pos_to_goal.norm() < 0.01) {
				_state_iter = 0;
				_state = PickAndPlace::State::PLACE;
			}
			break;
		case PickAndPlace::State::PLACE:
			_robot->position(pos_current, _link_name, _control_frame.translation());
			pos_goal = _pos_place;
			
			if (_state_iter <= 0) {
				cout << "pnp state: place" << endl;
				_redundant_arm_motion->_desired_position = PickAndPlace::get_pos_move_end();
				delta_pos = pos_goal - PickAndPlace::get_pos_move_end();
				direction = delta_pos / delta_pos.norm();
				_step_pos = direction * velocity / control_freq;
			}
			
			_redundant_arm_motion->_desired_position += _step_pos;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
			pos_to_goal = pos_goal - _redundant_arm_motion->_desired_position;

			_state_iter++;
			if (pos_to_goal.norm() < 0.01) {
				_state_iter = 0;
				_state = PickAndPlace::State::RELEASE;
			}
			break;
		case PickAndPlace::State::RELEASE:
			if (_state_iter <= 0) {
				// first iteration
				cout << "grasp state: release" << endl;
				_redis_client->set(ALLEGRO_COMMAND, ALLEGRO_UNCONTROLLED);
			}
			_state_iter++;
			// wait for two seconds
			if (_state_iter >= control_freq * 2) {
				_state_iter = 0;
				_state = PickAndPlace::State::UP;
			}
			break;
		case PickAndPlace::State::UP:
			_robot->position(pos_current, _link_name, _control_frame.translation());
			pos_goal = PickAndPlace::get_pos_move_end();
			
			if (_state_iter <= 0) {
				cout << "pnp state: up" << endl;
				_redundant_arm_motion->_desired_position = _pos_place;
				delta_pos = pos_goal - _pos_place;
				direction = delta_pos / delta_pos.norm();
				_step_pos = direction * velocity / control_freq;
			}
			
			_redundant_arm_motion->_desired_position += _step_pos;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
			pos_to_goal = pos_goal - _redundant_arm_motion->_desired_position;

			_state_iter++;
			if (pos_to_goal.norm() < 0.01) {
				_state_iter = 0;
				_state = PickAndPlace::State::DONE;
			}
			break;
		default:
			break;
	}
}

Eigen::Vector3d PickAndPlace::get_pos_move_start() {
	Eigen::Vector3d pos;
	pos << _pos_pick(0), _pos_pick(1), _pos_pick(2) + _delta_z;
	return pos;
}

Eigen::Vector3d PickAndPlace::get_pos_move_end() {
	Eigen::Vector3d pos;
	pos << _pos_place(0), _pos_place(1), _pos_place(2) + _delta_z;
	return pos;
}

void PickAndPlace::enableGravComp()
{
	_gravity_compensation = true;
}

void PickAndPlace::disbleGravComp()
{
	_gravity_compensation = false;
}

} /* namespace Sai2Primitives */

