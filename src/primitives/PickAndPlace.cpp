/*
 * PickAndPlace.cpp
 *
 *      This class creates a motion primitive for a redundant arm using a posori task and a joint task in its nullspace
 *
 *      Author: Mikael Jorda
 */

#include "PickAndPlace.h"

using namespace std;

namespace Sai2Primitives
{


PickAndPlace::PickAndPlace(Sai2Model::Sai2Model* robot,
				   const std::string link_name,
                   const Eigen::Affine3d control_frame)
{
	_robot = robot;
	_link_name = link_name;
	_control_frame = control_frame;
	_state = PickAndPlace::State::PICK;

	_redundant_arm_motion = new RedundantArmMotion(robot, link_name, control_frame);
}

PickAndPlace::PickAndPlace(Sai2Model::Sai2Model* robot,
				   const std::string link_name,
                   const Eigen::Vector3d pos_in_link,
                   const Eigen::Matrix3d rot_in_link)
{
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
	Eigen::Vector3d step_pos;

	switch(_state) {
		case PickAndPlace::State::PICK:
			_robot->position(pos_current, _link_name, _control_frame.translation());
			pos_goal = PickAndPlace::get_pos_move_start();
			delta_pos = pos_goal - pos_current;
			direction = delta_pos / delta_pos.norm();
			step_pos = direction * velocity / control_freq;
			// cout << "step_pos: " << step_pos << endl;
			_redundant_arm_motion->_desired_position += step_pos;
			// cout << "_redundant_arm_motion->_desired_position: " << _redundant_arm_motion->_desired_position << endl;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();
			// cout << "delta_pos: " << delta_pos << endl;

			if (_state_iter <= 0) {
				cout << "pnp state: pick" << endl;
			}
			_state_iter++;
			if (delta_pos.norm() < 0.001) {
				_state_iter = 0;
				_state = PickAndPlace::State::MOVE;
			}
			break;
		case PickAndPlace::State::MOVE:
			_robot->position(pos_current, _link_name, _control_frame.translation());
			pos_goal = PickAndPlace::get_pos_move_end();
			delta_pos = pos_goal - pos_current;
			direction = delta_pos / delta_pos.norm();
			step_pos = direction * velocity / control_freq;
			_redundant_arm_motion->_desired_position += step_pos;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();

			if (_state_iter <= 0) {
				cout << "pnp state: move" << endl;
			}
			_state_iter++;
			if (delta_pos.norm() < 0.001) {
				_state_iter = 0;
				_state = PickAndPlace::State::PLACE;
			}
			break;
		case PickAndPlace::State::PLACE:
			_robot->position(pos_current, _link_name, _control_frame.translation());
			pos_goal = _pos_place;
			delta_pos = pos_goal - pos_current;
			direction = delta_pos / delta_pos.norm();
			step_pos = direction * velocity / control_freq;
			_redundant_arm_motion->_desired_position += step_pos;
			_redundant_arm_motion->_desired_velocity = Eigen::Vector3d::Zero();

			if (_state_iter <= 0) {
				cout << "pnp state: place" << endl;
			}
			_state_iter++;
			if (delta_pos.norm() < 0.001) {
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

