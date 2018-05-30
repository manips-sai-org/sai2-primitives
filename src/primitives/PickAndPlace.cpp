/*
 * PickAndPlace.cpp
 *
 *      This class creates a motion primitive for a redundant arm using a posori task and a joint task in its nullspace
 *
 *      Author: Mikael Jorda
 */

#include "PickAndPlace.h"

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

void PickAndPlace::start(const Eigen::Vector3d pos_pick,
					   const Eigen::Matrix3d rot_pick,
					   const double delta_z,
					   const Eigen::Vector3d pos_place,
					   const Eigen::Matrix3d rot_place) {

	_state = PickAndPlace::State::PICK;
	_iter = 0;
	_pos_pick = pos_pick;
	_rot_pick = rot_pick;
	_delta_z = delta_z;
	_pos_place = pos_place;
	_rot_place = rot_place;
}

void PickAndPlace::step() {

	_iter++;
	Eigen::Vector3d pos_move_start;
	Eigen::Vector3d pos_move_end;
	Eigen::Vector3d step;
	Eigen::Vector3d des_pos;

	switch(_state) {
		case PickAndPlace::State::PICK:
			pos_move_start = PickAndPlace::get_pos_move_start();
			step = (pos_move_start - _pos_pick) / _pick_n_iters;
			des_pos = _pos_pick + step * _iter;
			_redundant_arm_motion->_desired_position = des_pos;

			if (_iter >= _pick_n_iters) {
				_state = PickAndPlace::State::MOVE;
				_iter = 0;
			}
			break;
		case PickAndPlace::State::MOVE:
			pos_move_start = PickAndPlace::get_pos_move_start();
			pos_move_end = PickAndPlace::get_pos_move_end();
			step = (pos_move_end - pos_move_start) / _pick_n_iters;
			des_pos = pos_move_start + step * _iter;
			_redundant_arm_motion->_desired_position = des_pos;

			if (_iter >= _move_n_iters) {
				_state = PickAndPlace::State::PLACE;
				_iter = 0;
			}
			break;
		case PickAndPlace::State::PLACE:
			pos_move_end = PickAndPlace::get_pos_move_end();
			step = (_pos_place - pos_move_end) / _pick_n_iters;
			des_pos = pos_move_end + step * _iter;
			_redundant_arm_motion->_desired_position = des_pos;

			if (_iter >= _place_n_iters) {
				_iter = _place_n_iters;
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

