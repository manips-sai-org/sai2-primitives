/*
 * ComMotionTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "ComMotionTask.h"

#include <stdexcept>

using namespace std;
using namespace Eigen;

namespace Sai2Primitives {

namespace {
const double MAX_FEEDBACK_FORCE_FORCE_CONTROLLER = 20.0;
const double MAX_FEEDBACK_MOMENT_FORCE_CONTROLLER = 10.0;
}  // namespace

ComMotionTask::ComMotionTask(
	std::shared_ptr<Sai2Model::Sai2Model>& robot, const string& link_name,
	const Affine3d& compliant_frame, const std::string& task_name,
	const bool is_force_motion_parametrization_in_compliant_frame,
	const double loop_timestep)
	: TemplateTask(robot, task_name, TaskType::MOTION_FORCE_TASK,
				   loop_timestep) {
	_link_name = link_name;
	_compliant_frame = compliant_frame;
	_is_force_motion_parametrization_in_compliant_frame =
		is_force_motion_parametrization_in_compliant_frame;

	_partial_task_projection = Matrix<double, 6, 6>::Identity(6, 6);

	initialSetup();
}

ComMotionTask::ComMotionTask(
	std::shared_ptr<Sai2Model::Sai2Model>& robot, const string& link_name,
	std::vector<Vector3d> controlled_directions_translation,
	std::vector<Vector3d> controlled_directions_rotation,
	const Affine3d& compliant_frame, const std::string& task_name,
	const bool is_force_motion_parametrization_in_compliant_frame,
	const double loop_timestep)
	: TemplateTask(robot, task_name, TaskType::MOTION_FORCE_TASK,
				   loop_timestep) {
	_link_name = link_name;
	_compliant_frame = compliant_frame;
	_is_force_motion_parametrization_in_compliant_frame =
		is_force_motion_parametrization_in_compliant_frame;

	if (controlled_directions_translation.empty() &&
		controlled_directions_rotation.empty()) {
		throw invalid_argument(
			"controlled_directions_translation and "
			"controlled_directions_rotation cannot both be empty in "
			"ComMotionTask::ComMotionTask\n");
	}

	MatrixXd controlled_translation_range_basis = MatrixXd::Zero(3, 1);
	MatrixXd controlled_rotation_range_basis = MatrixXd::Zero(3, 1);

	if (controlled_directions_translation.size() != 0) {
		MatrixXd controlled_translation_vectors =
			MatrixXd::Zero(3, controlled_directions_translation.size());
		for (int i = 0; i < controlled_directions_translation.size(); i++) {
			controlled_translation_vectors.col(i) =
				controlled_directions_translation[i];
		}
		controlled_translation_range_basis =
			Sai2Model::matrixRangeBasis(controlled_translation_vectors);
	}

	if (controlled_directions_rotation.size() != 0) {
		MatrixXd controlled_rotation_vectors =
			MatrixXd::Zero(3, controlled_directions_rotation.size());
		for (int i = 0; i < controlled_directions_rotation.size(); i++) {
			controlled_rotation_vectors.col(i) =
				controlled_directions_rotation[i];
		}

		controlled_rotation_range_basis =
			Sai2Model::matrixRangeBasis(controlled_rotation_vectors);
	}

	_partial_task_projection.setZero();
	_partial_task_projection.block<3, 3>(0, 0) =
		controlled_translation_range_basis *
		controlled_translation_range_basis.transpose();
	_partial_task_projection.block<3, 3>(3, 3) =
		controlled_rotation_range_basis *
		controlled_rotation_range_basis.transpose();

	initialSetup();
}

void ComMotionTask::initialSetup() {
	setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

	int dof = getConstRobotModel()->dof();
	_T_control_to_sensor = Affine3d::Identity();

	// motion
	_current_position = getConstRobotModel()->comPosition();
	_current_orientation = getConstRobotModel()->rotationInWorld(
		_link_name, _compliant_frame.rotation());

	// default values for gains and velocity saturation
	setPosControlGains(50.0, 14.0, 0.0);
	setOriControlGains(50.0, 14.0, 0.0);

	disableVelocitySaturation();
	_linear_saturation_velocity = 0;
	_angular_saturation_velocity = 0;

	_k_ff = 0.95;

	// initialize matrices sizes
	_jacobian.setZero(6, dof);
	_projected_jacobian.setZero(6, dof);
	_Lambda.setZero(6, 6);
	_Lambda_modified.setZero(6, 6);
	_Jbar.setZero(dof, 6);
	_N.setZero(dof, dof);
	_N_prec = MatrixXd::Identity(dof, dof);

	MatrixXd range_pos =
		Sai2Model::matrixRangeBasis(_partial_task_projection.block<3, 3>(0, 0));
	MatrixXd range_ori =
		Sai2Model::matrixRangeBasis(_partial_task_projection.block<3, 3>(3, 3));

	_pos_range = range_pos.norm() == 0 ? 0 : range_pos.cols();
	_ori_range = range_ori.norm() == 0 ? 0 : range_ori.cols();

	if (_pos_range + _ori_range == 0)  // should not happen
	{
		throw invalid_argument(
			"controlled_directions_translation and "
			"controlled_directions_rotation cannot both be empty in "
			"ComMotionTask::ComMotionTask\n");
	}

	_current_task_range.setZero(6, _pos_range + _ori_range);
	if (_pos_range > 0) {
		_current_task_range.block(0, 0, 3, _pos_range) = range_pos;
	}
	if (_ori_range > 0) {
		_current_task_range.block(3, _pos_range, 3, _ori_range) = range_ori;
	}

	// trajectory generation
	_otg = make_unique<OTG_6dof_cartesian>(
		_current_position, _current_orientation, getLoopTimestep());
	enableInternalOtgAccelerationLimited(0.3, 1.0, M_PI / 3, M_PI);

	reInitializeTask();
}

void ComMotionTask::reInitializeTask() {
	int dof = getConstRobotModel()->dof();

	// motion
	_current_position = getConstRobotModel()->comPosition();
	_desired_position = _current_position;
	_current_orientation = getConstRobotModel()->rotationInWorld(
		_link_name, _compliant_frame.rotation());
	_desired_orientation = _current_orientation;

	_current_velocity.setZero();
	_desired_velocity.setZero();
	_current_angular_velocity.setZero();
	_desired_angular_velocity.setZero();
	_desired_acceleration.setZero();
	_desired_angular_acceleration.setZero();

	_orientation_error.setZero();
	_integrated_position_error.setZero();
	_integrated_orientation_error.setZero();

	resetIntegrators();

	_unit_mass_force.setZero(6);

	_otg->reInitialize(_current_position, _current_orientation);
}

void ComMotionTask::updateTaskModel(const MatrixXd& N_prec) {
	const int robot_dof = getConstRobotModel()->dof();
	if (N_prec.rows() != N_prec.cols()) {
		throw invalid_argument(
			"N_prec matrix not square in ComMotionTask::updateTaskModel\n");
	}
	if (N_prec.rows() != robot_dof) {
		throw invalid_argument(
			"N_prec matrix size not consistent with robot dof in "
			"ComMotionTask::updateTaskModel\n");
	}

	_N_prec = N_prec;

    MatrixXd J = getConstRobotModel()->JWorldFrame(
                    _link_name, _compliant_frame.translation());
    J.topRows(3) = getConstRobotModel()->comJacobian();
	_jacobian = _partial_task_projection * J;
	_projected_jacobian = _jacobian * _N_prec;

	MatrixXd range_pos =
		Sai2Model::matrixRangeBasis(_projected_jacobian.topRows(3));
	MatrixXd range_ori =
		Sai2Model::matrixRangeBasis(_projected_jacobian.bottomRows(3));

	_pos_range = range_pos.norm() == 0 ? 0 : range_pos.cols();
	_ori_range = range_ori.norm() == 0 ? 0 : range_ori.cols();

	if (_pos_range + _ori_range == 0) {
		// there is no controllable degree of freedom for the task, just return
		// should maybe print a warning here
		_N.setIdentity(robot_dof, robot_dof);
		return;
	}

	_current_task_range.setZero(6, _pos_range + _ori_range);
	if (_pos_range > 0) {
		_current_task_range.block(0, 0, 3, _pos_range) = range_pos;
	}
	if (_ori_range > 0) {
		_current_task_range.block(3, _pos_range, 3, _ori_range) = range_ori;
	}

	Sai2Model::OpSpaceMatrices op_space_matrices =
		getConstRobotModel()->operationalSpaceMatrices(
			_current_task_range.transpose() * _projected_jacobian);
	_Lambda = op_space_matrices.Lambda;
	_Jbar = op_space_matrices.Jbar;
	_N = op_space_matrices.N;

	switch (_dynamic_decoupling_type) {
		case FULL_DYNAMIC_DECOUPLING: {
			_Lambda_modified = _Lambda;
			break;
		}

		case PARTIAL_DYNAMIC_DECOUPLING: {
			_Lambda_modified = _Lambda;
			if (_ori_range > 0) {
				_Lambda_modified.block(_pos_range, _pos_range, _ori_range,
									   _ori_range) =
					MatrixXd::Identity(_ori_range, _ori_range);
				if (_pos_range > 0) {
					_Lambda_modified.block(0, _pos_range, _pos_range,
										   _ori_range) =
						MatrixXd::Zero(_pos_range, _ori_range);
					_Lambda_modified.block(_pos_range, 0, _ori_range,
										   _pos_range) =
						MatrixXd::Zero(_ori_range, _pos_range);
				}
			}

			break;
		}

		case IMPEDANCE: {
			_Lambda_modified = MatrixXd::Identity(_pos_range + _ori_range,
												  _pos_range + _ori_range);
			break;
		}

		case BOUNDED_INERTIA_ESTIMATES: {
			MatrixXd M_BIE = getConstRobotModel()->M();
			for (int i = 0; i < getConstRobotModel()->dof(); i++) {
				if (M_BIE(i, i) < 0.1) {
					M_BIE(i, i) = 0.1;
				}
			}
			MatrixXd M_inv_BIE = M_BIE.inverse();
			MatrixXd Lambda_inv_BIE =
				_current_task_range.transpose() * _projected_jacobian *
				(M_inv_BIE * _projected_jacobian.transpose()) *
				_current_task_range;
			_Lambda_modified = Lambda_inv_BIE.inverse();
			break;
		}

		default: {
			_Lambda_modified = _Lambda;
			break;
		}
	}
}

VectorXd ComMotionTask::computeTorques() {
	VectorXd task_joint_torques = VectorXd::Zero(getConstRobotModel()->dof());
    MatrixXd J = getConstRobotModel()->JWorldFrame(
                    _link_name, _compliant_frame.translation());
    J.topRows(3) = getConstRobotModel()->comJacobian();
	_jacobian = _partial_task_projection * J;
	_projected_jacobian = _jacobian * _N_prec;

	// update controller state
	_current_position = getConstRobotModel()->comPosition();
	_current_orientation = getConstRobotModel()->rotationInWorld(
		_link_name, _compliant_frame.rotation());

	_orientation_error =
		Sai2Model::orientationError(_desired_orientation, _current_orientation);
	_current_velocity =
		_projected_jacobian.block(0, 0, 3, getConstRobotModel()->dof()) *
		getConstRobotModel()->dq();
	_current_angular_velocity =
		_projected_jacobian.block(3, 0, 3, getConstRobotModel()->dof()) *
		getConstRobotModel()->dq();

	if (_pos_range + _ori_range == 0) {
		// there is no controllable degree of freedom for the task, just return
		// zero torques. should maybe print a warning here
		return task_joint_torques;
	}

	Matrix3d sigma_position = Matrix3d::Identity();
	Matrix3d sigma_orientation = Matrix3d::Identity();

	Vector3d position_related_force = Vector3d::Zero();
	Vector3d orientation_related_force = Vector3d::Zero();

	Matrix3d kp_pos =
		_current_orientation * _kp_pos * _current_orientation.transpose();
	Matrix3d kv_pos =
		_current_orientation * _kv_pos * _current_orientation.transpose();
	Matrix3d ki_pos =
		_current_orientation * _ki_pos * _current_orientation.transpose();

	// motion related terms
	// compute next state from trajectory generation
	Vector3d tmp_desired_position = _desired_position;
	Matrix3d tmp_desired_orientation = _desired_orientation;
	Vector3d tmp_desired_velocity = _desired_velocity;
	Vector3d tmp_desired_angular_velocity = _desired_angular_velocity;
	Vector3d tmp_desired_acceleration = _desired_acceleration;
	Vector3d tmp_desired_angular_acceleration = _desired_angular_acceleration;

	if (_use_internal_otg_flag) {
		_otg->setGoalPositionAndLinearVelocity(_desired_position,
											   _desired_velocity);
		_otg->setGoalOrientationAndAngularVelocity(_desired_orientation,
												   _desired_angular_velocity);
		_otg->update();

		tmp_desired_position = _otg->getNextPosition();
		tmp_desired_velocity = _otg->getNextLinearVelocity();
		tmp_desired_acceleration = _otg->getNextLinearAcceleration();
		tmp_desired_orientation = _otg->getNextOrientation();
		tmp_desired_angular_velocity = _otg->getNextAngularVelocity();
		tmp_desired_angular_acceleration = _otg->getNextAngularAcceleration();
	}

	// linear motion
	// update integrated error for I term
	_integrated_position_error += sigma_position *
								  (_current_position - tmp_desired_position) *
								  getLoopTimestep();

	// final contribution
	if (_use_velocity_saturation_flag) {
		tmp_desired_velocity =
			-_kp_pos * _kv_pos.inverse() * sigma_position *
				(_current_position - tmp_desired_position) -
			_ki_pos * _kv_pos.inverse() * _integrated_position_error;
		if (tmp_desired_velocity.norm() > _linear_saturation_velocity) {
			tmp_desired_velocity *=
				_linear_saturation_velocity / tmp_desired_velocity.norm();
		}
		position_related_force =
			sigma_position *
			(tmp_desired_acceleration -
			 _kv_pos * (_current_velocity - tmp_desired_velocity));
	} else {
		position_related_force =
			sigma_position *
			(tmp_desired_acceleration -
			 _kp_pos * (_current_position - tmp_desired_position) -
			 _kv_pos * (_current_velocity - tmp_desired_velocity) -
			 _ki_pos * _integrated_position_error);
	}

	// angular motion
	// orientation error
	Vector3d step_orientation_error =
		sigma_orientation * Sai2Model::orientationError(tmp_desired_orientation,
														_current_orientation);

	// update integrated error for I term
	_integrated_orientation_error += step_orientation_error * getLoopTimestep();

	// final contribution
	if (_use_velocity_saturation_flag) {
		tmp_desired_angular_velocity =
			-_kp_ori * _kv_ori.inverse() * step_orientation_error -
			_ki_ori * _kv_ori.inverse() * _integrated_orientation_error;
		if (tmp_desired_angular_velocity.norm() >
			_angular_saturation_velocity) {
			tmp_desired_angular_velocity *= _angular_saturation_velocity /
											tmp_desired_angular_velocity.norm();
		}
		orientation_related_force =
			sigma_orientation * (tmp_desired_angular_acceleration -
								 _kv_ori * (_current_angular_velocity -
											tmp_desired_angular_velocity));
	} else {
		orientation_related_force =
			sigma_orientation * (tmp_desired_angular_acceleration -
								 _kp_ori * step_orientation_error -
								 _kv_ori * (_current_angular_velocity -
											tmp_desired_angular_velocity) -
								 _ki_ori * _integrated_orientation_error);
	}

	// compute task force
	VectorXd position_orientation_contribution(6);
	position_orientation_contribution.head(3) = position_related_force;
	position_orientation_contribution.tail(3) = orientation_related_force;

	_unit_mass_force = position_orientation_contribution;
	_linear_motion_control = position_related_force;

	_task_force = _Lambda_modified * _current_task_range.transpose() *
					  (position_orientation_contribution);

	// compute task torques
	task_joint_torques =
		_projected_jacobian.transpose() * _current_task_range * _task_force;

	return task_joint_torques;
}

void ComMotionTask::enableInternalOtgAccelerationLimited(
	const double max_linear_velelocity, const double max_linear_acceleration,
	const double max_angular_velocity, const double max_angular_acceleration) {
	_otg->setMaxLinearVelocity(max_linear_velelocity);
	_otg->setMaxLinearAcceleration(max_linear_acceleration);
	_otg->setMaxAngularVelocity(max_angular_velocity);
	_otg->setMaxAngularAcceleration(max_angular_acceleration);
	_otg->disableJerkLimits();
	_use_internal_otg_flag = true;
}

void ComMotionTask::enableInternalOtgJerkLimited(
	const double max_linear_velelocity, const double max_linear_acceleration,
	const double max_linear_jerk, const double max_angular_velocity,
	const double max_angular_acceleration, const double max_angular_jerk) {
	_otg->setMaxLinearVelocity(max_linear_velelocity);
	_otg->setMaxLinearAcceleration(max_linear_acceleration);
	_otg->setMaxAngularVelocity(max_angular_velocity);
	_otg->setMaxAngularAcceleration(max_angular_acceleration);
	_otg->setMaxJerk(max_linear_jerk, max_angular_jerk);
	_use_internal_otg_flag = true;
}

Vector3d ComMotionTask::getPositionError() const {
	return sigmaPosition() * (_desired_position - _current_position);
}

Vector3d ComMotionTask::getOrientationError() const {
	return sigmaOrientation() * _orientation_error;
}

bool ComMotionTask::goalPositionReached(const double tolerance,
										  const bool verbose) {
	double position_error =
		(_desired_position - _current_position).transpose() * sigmaPosition() *
		(_desired_position - _current_position);
	position_error = sqrt(position_error);
	bool goal_reached = position_error < tolerance;
	if (verbose) {
		cout << "position error in ComMotionTask : " << position_error
			 << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
}

bool ComMotionTask::goalOrientationReached(const double tolerance,
											 const bool verbose) {
	double orientation_error = _orientation_error.transpose() *
							   sigmaOrientation() * _orientation_error;
	orientation_error = sqrt(orientation_error);
	bool goal_reached = orientation_error < tolerance;
	if (verbose) {
		cout << "orientation error in ComMotionTask : " << orientation_error
			 << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
}

void ComMotionTask::setPosControlGains(double kp_pos, double kv_pos,
										 double ki_pos) {
	if (kp_pos < 0 || kv_pos < 0 || ki_pos < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"ComMotionTask::setPosControlGains\n");
	}
	if (kv_pos < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_pos = 0 if using velocity saturation in "
			"ComMotionTask::setPosControlGains\n");
	}
	_are_pos_gains_isotropic = true;
	_kp_pos = kp_pos * Matrix3d::Identity();
	_kv_pos = kv_pos * Matrix3d::Identity();
	_ki_pos = ki_pos * Matrix3d::Identity();
}

void ComMotionTask::setPosControlGains(const Vector3d& kp_pos,
										 const Vector3d& kv_pos,
										 const Vector3d& ki_pos) {
	if (kp_pos.minCoeff() < 0 || kv_pos.minCoeff() < 0 ||
		ki_pos.minCoeff() < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"ComMotionTask::setPosControlGains\n");
	}
	if (kv_pos.minCoeff() < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_pos = 0 if using velocity saturation in "
			"ComMotionTask::setPosControlGains\n");
	}
	_are_pos_gains_isotropic = false;
	_kp_pos = kp_pos.asDiagonal();
	_kv_pos = kv_pos.asDiagonal();
	_ki_pos = ki_pos.asDiagonal();
}

vector<PIDGains> ComMotionTask::getPosControlGains() const {
	if (_are_pos_gains_isotropic) {
		return vector<PIDGains>(
			1, PIDGains(_kp_pos(0, 0), _kv_pos(0, 0), _ki_pos(0, 0)));
	}
	Vector3d aniso_kp_robot_base = _kp_pos.diagonal();
	Vector3d aniso_kv_robot_base = _kv_pos.diagonal();
	Vector3d aniso_ki_robot_base = _ki_pos.diagonal();
	return vector<PIDGains>{
		PIDGains(aniso_kp_robot_base(0), aniso_kv_robot_base(0),
				 aniso_ki_robot_base(0)),
		PIDGains(aniso_kp_robot_base(1), aniso_kv_robot_base(1),
				 aniso_ki_robot_base(1)),
		PIDGains(aniso_kp_robot_base(2), aniso_kv_robot_base(2),
				 aniso_ki_robot_base(2))};
}

void ComMotionTask::setOriControlGains(double kp_ori, double kv_ori,
										 double ki_ori) {
	if (kp_ori < 0 || kv_ori < 0 || ki_ori < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"ComMotionTask::setOriControlGains\n");
	}
	if (kv_ori < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_ori = 0 if using velocity saturation in "
			"ComMotionTask::setOriControlGains\n");
	}
	_are_ori_gains_isotropic = true;
	_kp_ori = kp_ori * Matrix3d::Identity();
	_kv_ori = kv_ori * Matrix3d::Identity();
	_ki_ori = ki_ori * Matrix3d::Identity();
}

void ComMotionTask::setOriControlGains(const Vector3d& kp_ori,
										 const Vector3d& kv_ori,
										 const Vector3d& ki_ori) {
	if (kp_ori.minCoeff() < 0 || kv_ori.minCoeff() < 0 ||
		ki_ori.minCoeff() < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"ComMotionTask::setOriControlGains\n");
	}
	if (kv_ori.minCoeff() < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_ori = 0 if using velocity saturation in "
			"ComMotionTask::setOriControlGains\n");
	}
	_are_ori_gains_isotropic = false;
	_kp_ori = kp_ori.asDiagonal();
	_kv_ori = kv_ori.asDiagonal();
	_ki_ori = ki_ori.asDiagonal();
}

vector<PIDGains> ComMotionTask::getOriControlGains() const {
	if (_are_ori_gains_isotropic) {
		return vector<PIDGains>(
			1, PIDGains(_kp_ori(0, 0), _kv_ori(0, 0), _ki_ori(0, 0)));
	}
	Vector3d aniso_kp_robot_base = _kp_ori.diagonal();
	Vector3d aniso_kv_robot_base = _kv_ori.diagonal();
	Vector3d aniso_ki_robot_base = _ki_ori.diagonal();
	return vector<PIDGains>{
		PIDGains(aniso_kp_robot_base(0), aniso_kv_robot_base(0),
				 aniso_ki_robot_base(0)),
		PIDGains(aniso_kp_robot_base(1), aniso_kv_robot_base(1),
				 aniso_ki_robot_base(1)),
		PIDGains(aniso_kp_robot_base(2), aniso_kv_robot_base(2),
				 aniso_ki_robot_base(2))};
}

// Vector3d ComMotionTask::getDesiredForce() const {
// 	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
// 							? getConstRobotModel()->rotationInWorld(
// 								  _link_name, _compliant_frame.rotation())
// 							: Matrix3d::Identity();
// 	return rotation * _desired_force;
// }

// Vector3d ComMotionTask::getDesiredMoment() const {
// 	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
// 							? getConstRobotModel()->rotationInWorld(
// 								  _link_name, _compliant_frame.rotation())
// 							: Matrix3d::Identity();
// 	return rotation * _desired_moment;
// }

void ComMotionTask::enableVelocitySaturation(const double linear_vel_sat,
											   const double angular_vel_sat) {
	if (linear_vel_sat <= 0 || angular_vel_sat <= 0) {
		throw invalid_argument(
			"Velocity saturation values should be strictly positive or zero in "
			"ComMotionTask::enableVelocitySaturation\n");
	}
	if (_kv_pos.determinant() < 1e-3) {
		throw invalid_argument(
			"Cannot enable velocity saturation if kv_pos is singular in "
			"ComMotionTask::enableVelocitySaturation\n");
	}
	if (_kv_ori.determinant() < 1e-3) {
		throw invalid_argument(
			"Cannot enable velocity saturation if kv_ori is singular in "
			"ComMotionTask::enableVelocitySaturation\n");
	}
	_use_velocity_saturation_flag = true;
	_linear_saturation_velocity = linear_vel_sat;
	_angular_saturation_velocity = angular_vel_sat;
}

// void ComMotionTask::setForceSensorFrame(
// 	const string link_name, const Affine3d transformation_in_link) {
// 	if (link_name != _link_name) {
// 		throw invalid_argument(
// 			"The link to which is attached the sensor should be the same as "
// 			"the link to which is attached the control frame in "
// 			"ComMotionTask::setForceSensorFrame\n");
// 	}
// 	_T_control_to_sensor = _compliant_frame.inverse() * transformation_in_link;
// 	_force_sensor->setForceSensorFrame(link_name, transformation_in_link);
// }

// void ComMotionTask::updateSensedForceAndMoment(
// 	const Vector3d sensed_force_sensor_frame,
// 	const Vector3d sensed_moment_sensor_frame) {
	
// 	// get calibrated force sensor data 
// 	ForceMeasurement force_moment = \
// 		_force_sensor->getCalibratedForceMoment(sensed_force_sensor_frame, sensed_moment_sensor_frame);
// 	Vector3d calibrated_sensed_force_sensor_frame = force_moment.force;
// 	Vector3d calibrated_sensed_moment_sensor_frame = force_moment.moment;

// 	// find the transform from base frame to control frame
// 	Affine3d T_world_link = getConstRobotModel()->transformInWorld(_link_name);
// 	Affine3d T_world_compliant_frame = T_world_link * _compliant_frame;

// 	// find the resolved sensed force and moment in control frame
// 	_sensed_force = _T_control_to_sensor.rotation() * calibrated_sensed_force_sensor_frame;
// 	_sensed_moment =
// 		_T_control_to_sensor.translation().cross(_sensed_force) +
// 		_T_control_to_sensor.rotation() * calibrated_sensed_moment_sensor_frame;

// 	// rotate the quantities in base frame
// 	_sensed_force = T_world_compliant_frame.rotation() * _sensed_force;
// 	_sensed_moment = T_world_compliant_frame.rotation() * _sensed_moment;
// }

// void ComMotionTask::parametrizeForceMotionSpaces(
// 	const int force_space_dimension,
// 	const Vector3d& force_or_motion_single_axis) {
// 	if (force_space_dimension < 0 || force_space_dimension > 3) {
// 		throw invalid_argument(
// 			"Force space dimension should be between 0 and 3 in "
// 			"ComMotionTask::parametrizeForceMotionSpaces\n");
// 	}
// 	_force_space_dimension = force_space_dimension;
// 	if (force_space_dimension == 1 || force_space_dimension == 2) {
// 		if (force_or_motion_single_axis.norm() < 1e-2) {
// 			throw invalid_argument(
// 				"Force or motion axis should be a non singular vector in "
// 				"ComMotionTask::parametrizeForceMotionSpaces\n");
// 		}
// 		_force_or_motion_axis = force_or_motion_single_axis.normalized();
// 	}
// 	resetIntegratorsLinear();
// }

// void ComMotionTask::parametrizeMomentRotMotionSpaces(
// 	const int moment_space_dimension,
// 	const Vector3d& moment_or_rot_motion_single_axis) {
// 	if (moment_space_dimension < 0 || moment_space_dimension > 3) {
// 		throw invalid_argument(
// 			"Moment space dimension should be between 0 and 3 in "
// 			"ComMotionTask::parametrizeMomentRotMotionSpaces\n");
// 	}
// 	_moment_space_dimension = moment_space_dimension;
// 	if (moment_space_dimension == 1 || moment_space_dimension == 2) {
// 		if (moment_or_rot_motion_single_axis.norm() < 1e-2) {
// 			throw invalid_argument(
// 				"Moment or rot motion axis should be a non singular vector in "
// 				"ComMotionTask::parametrizeMomentRotMotionSpaces\n");
// 		}
// 		_moment_or_rotmotion_axis =
// 			moment_or_rot_motion_single_axis.normalized();
// 	}
// 	resetIntegratorsAngular();
// }

// Matrix3d ComMotionTask::sigmaForce() const {
// 	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
// 							? getConstRobotModel()->rotationInWorld(
// 								  _link_name, _compliant_frame.rotation())
// 							: Matrix3d::Identity();
// 	switch (_force_space_dimension) {
// 		case 0:
// 			return Matrix3d::Zero();
// 			break;
// 		case 1:
// 			return posSelectionProjector() * rotation * _force_or_motion_axis *
// 				   _force_or_motion_axis.transpose() * rotation.transpose() *
// 				   posSelectionProjector().transpose();
// 			break;
// 		case 2:
// 			return posSelectionProjector() *
// 				   (Matrix3d::Identity() -
// 					rotation * _force_or_motion_axis *
// 						_force_or_motion_axis.transpose() *
// 						rotation.transpose()) *
// 				   posSelectionProjector().transpose();
// 			break;
// 		case 3:
// 			return posSelectionProjector();
// 			break;

// 		default:
// 			// should never happen
// 			throw invalid_argument(
// 				"Force space dimension should be between 0 and 3 in "
// 				"ComMotionTask::sigmaForce\n");
// 			break;
// 	}
// }

Matrix3d ComMotionTask::sigmaPosition() const {
    return Matrix3d::Identity();
	// return posSelectionProjector() * (Matrix3d::Identity() - sigmaForce()) *
	// 	   posSelectionProjector().transpose();
}

// Matrix3d ComMotionTask::sigmaMoment() const {
// 	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
// 							? getConstRobotModel()->rotationInWorld(
// 								  _link_name, _compliant_frame.rotation())
// 							: Matrix3d::Identity();
// 	switch (_moment_space_dimension) {
// 		case 0:
// 			return Matrix3d::Zero();
// 			break;
// 		case 1:
// 			return oriSelectionProjector() * rotation *
// 				   _moment_or_rotmotion_axis *
// 				   _moment_or_rotmotion_axis.transpose() *
// 				   rotation.transpose() * oriSelectionProjector().transpose();
// 			break;
// 		case 2:
// 			return oriSelectionProjector() *
// 				   (Matrix3d::Identity() -
// 					rotation * _moment_or_rotmotion_axis *
// 						_moment_or_rotmotion_axis.transpose() *
// 						rotation.transpose()) *
// 				   oriSelectionProjector().transpose();
// 			break;
// 		case 3:
// 			return oriSelectionProjector();
// 			break;

// 		default:
// 			// should never happen
// 			throw invalid_argument(
// 				"Moment space dimension should be between 0 and 3 in "
// 				"ComMotionTask::sigmaMoment\n");
// 			break;
// 	}
// }

Matrix3d ComMotionTask::sigmaOrientation() const {
    return Matrix3d::Identity();
	// return oriSelectionProjector() * (Matrix3d::Identity() - sigmaMoment()) *
	// 	   oriSelectionProjector().transpose();
}

void ComMotionTask::resetIntegrators() {
	resetIntegratorsLinear();
	resetIntegratorsAngular();
}

void ComMotionTask::resetIntegratorsLinear() {
	_integrated_position_error.setZero();
	_integrated_force_error.setZero();
}

void ComMotionTask::resetIntegratorsAngular() {
	_integrated_orientation_error.setZero();
	_integrated_moment_error.setZero();
}

// void ComMotionTask::addLoad(const std::string link_name,
// 							  const double mass,
// 							  const Vector3d& com,
// 							  const Matrix3d& inertia,
// 							  const std::string body_name) {
// 	_force_sensor->setToolInertia(mass, com, inertia);
// 	getConstRobotModel()->addLoad(link_name, mass, com, inertia, body_name);
// }

// void ComMotionTask::removeLoad(const std::string link_name,
// 								 const double mass,
// 								 const Vector3d& com,
// 								 const Matrix3d& inertia,
// 								 const std::string body_name) {
// 	_force_sensor->clearToolInertia();
// 	getConstRobotModel()->removeLoad(link_name, mass, com, inertia, body_name);
// }

} /* namespace Sai2Primitives */