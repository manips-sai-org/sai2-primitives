/*
 * MotionForceTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "MotionForceTask.h"

#include <stdexcept>

using namespace std;
using namespace Eigen;

namespace Sai2Primitives {

namespace {
const double MAX_FEEDBACK_FORCE_FORCE_CONTROLLER = 20.0;
const double MAX_FEEDBACK_MOMENT_FORCE_CONTROLLER = 10.0;

Affine3d transformFromVecAndMat(const Vector3d& vec, const Matrix3d& mat) {
	Affine3d transform = Affine3d::Identity();
	transform.translation() = vec;
	transform.linear() = mat;
	return transform;
}

}  // namespace

MotionForceTask::MotionForceTask(
	std::shared_ptr<Sai2Model::Sai2Model> robot, const string& link_name,
	const Affine3d& compliant_frame_in_link,
	const bool is_force_parametrization_in_compliant_frame,
	const Vector3d& wrist_point_in_link,
	const double loop_timestep) {
	_loop_timestep = loop_timestep;

	_robot = robot;
	_link_name = link_name;
	_is_force_parametrization_in_compliant_frame =
		is_force_parametrization_in_compliant_frame;

	setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

	_wrist_frame_in_link = Affine3d(Translation3d(wrist_point_in_link));
	_wrist_frame_in_link.linear() = compliant_frame_in_link.rotation();

	_wrist_to_compliant_frame_origin =
		compliant_frame_in_link.rotation().transpose() *
		(compliant_frame_in_link.translation() - wrist_point_in_link);

	int dof = _robot->dof();

	_force_sensor_frame_in_link = Affine3d::Identity();

	// POPC force
	_POPC_force.reset(new POPCExplicitForceControl(_loop_timestep));

	// motion
	_current_wrist_position =
		_robot->position(_link_name, _wrist_frame_in_link.translation());
	_current_wrist_orientation = _robot->rotation(_link_name, _wrist_frame_in_link.rotation());

	// default values for gains and velocity saturation
	setPosControlGains(50.0, 14.0, 0.0);
	setOriControlGains(50.0, 14.0, 0.0);
	setForceControlGains(0.7, 10.0, 1.3);
	setMomentControlGains(0.7, 10.0, 1.3);

	disableVelocitySaturation();
	_linear_saturation_velocity = 0;
	_angular_saturation_velocity = 0;

	_k_ff = 1.0;

	parametrizeForceMotionSpaces(0);

	// initialize matrices sizes
	_jacobian.setZero(6, dof);
	_projected_jacobian.setZero(6, dof);
	_Lambda.setZero(6, 6);
	_Lambda_modified.setZero(6, 6);
	_Jbar.setZero(dof, 6);
	_N.setZero(dof, dof);
	_N_prec = MatrixXd::Identity(dof, dof);

	_URange_pos = MatrixXd::Identity(3, 3);
	_URange_ori = MatrixXd::Identity(3, 3);
	_URange = MatrixXd::Identity(6, 6);

	_pos_dof = 3;
	_ori_dof = 3;

	// trajectory generation
	_otg.reset(new OTG_6dof_cartesian(_current_wrist_position, _current_wrist_orientation,
									  _loop_timestep));
	enableInternalOtgAccelerationLimited(0.3, 1.0, M_PI / 3, M_PI);

	reInitializeTask();
}

void MotionForceTask::reInitializeTask() {
	int dof = _robot->dof();

	// initialize state
	_current_wrist_position =
		_robot->position(_link_name, _wrist_frame_in_link.translation());
	_current_wrist_orientation = _robot->rotation(_link_name, _wrist_frame_in_link.rotation());

	_current_wrist_velocity.setZero();
	_current_wrist_angular_velocity.setZero();

	_integrated_position_error.setZero();
	_integrated_orientation_error.setZero();

	_sensed_force_at_wrist.setZero();
	_sensed_moment_at_wrist.setZero();

	// initialize desired state
	_desired_compliant_frame_position = _current_wrist_position + _current_wrist_orientation * _wrist_to_compliant_frame_origin;
	_desired_compliant_frame_orientation = _current_wrist_orientation;

	_desired_compliant_frame_velocity.setZero();
	_desired_compliant_frame_angular_velocity.setZero();
	_desired_compliant_frame_acceleration.setZero();
	_desired_compliant_frame_angular_acceleration.setZero();

	_desired_compliant_frame_force.setZero();
	_desired_compliant_frame_moment.setZero();

	resetIntegrators();

	_task_force.setZero(6);

	// initialize otg
	_otg->reInitialize(_current_wrist_position, _current_wrist_orientation);
}

void MotionForceTask::updateTaskModel(const MatrixXd N_prec) {
	if (N_prec.rows() != N_prec.cols()) {
		throw invalid_argument(
			"N_prec matrix not square in MotionForceTask::updateTaskModel\n");
	}
	if (N_prec.rows() != _robot->dof()) {
		throw invalid_argument(
			"N_prec matrix size not consistent with robot dof in "
			"MotionForceTask::updateTaskModel\n");
	}

	_N_prec = N_prec;

	_jacobian = _robot->J(_link_name, _wrist_frame_in_link.translation());
	_projected_jacobian = _jacobian * _N_prec;

	_URange_pos = Sai2Model::matrixRangeBasis(_projected_jacobian.topRows(3));
	_URange_ori =
		Sai2Model::matrixRangeBasis(_projected_jacobian.bottomRows(3));

	_pos_dof = _URange_pos.cols();
	_ori_dof = _URange_ori.cols();

	_URange.setZero(6, _pos_dof + _ori_dof);
	_URange.block(0, 0, 3, _pos_dof) = _URange_pos;
	_URange.block(3, _pos_dof, 3, _ori_dof) = _URange_ori;

	Sai2Model::OpSpaceMatrices op_space_matrices =
		_robot->operationalSpaceMatrices(
			_URange.transpose() * _projected_jacobian, _N_prec);
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
			_Lambda_modified.block(_pos_dof, _pos_dof, _ori_dof, _ori_dof) =
				MatrixXd::Identity(_ori_dof, _ori_dof);
			_Lambda_modified.block(0, _pos_dof, _pos_dof, _ori_dof) =
				MatrixXd::Zero(_pos_dof, _ori_dof);
			_Lambda_modified.block(_pos_dof, 0, _ori_dof, _pos_dof) =
				MatrixXd::Zero(_ori_dof, _pos_dof);
			break;
		}

		case IMPEDANCE: {
			_Lambda_modified =
				MatrixXd::Identity(_pos_dof + _ori_dof, _pos_dof + _ori_dof);
			break;
		}

		case BOUNDED_INERTIA_ESTIMATES: {
			MatrixXd M_BIE = _robot->M();
			for (int i = 0; i < _robot->dof(); i++) {
				if (M_BIE(i, i) < 0.1) {
					M_BIE(i, i) = 0.1;
				}
			}
			MatrixXd M_inv_BIE = M_BIE.inverse();
			MatrixXd Lambda_inv_BIE =
				_URange.transpose() * _projected_jacobian *
				(M_inv_BIE * _projected_jacobian.transpose()) * _URange;
			_Lambda_modified = Lambda_inv_BIE.inverse();
			break;
		}

		default: {
			_Lambda_modified = _Lambda;
			break;
		}
	}
}

VectorXd MotionForceTask::computeTorques() {
	VectorXd task_joint_torques = VectorXd::Zero(_robot->dof());
	_jacobian = _robot->J(_link_name, _wrist_frame_in_link.translation());
	_projected_jacobian = _jacobian * _N_prec;

	Matrix3d sigma_force = sigmaForce();
	Matrix3d sigma_moment = sigmaMoment();
	Matrix3d sigma_position = Matrix3d::Identity() - sigma_force;
	Matrix3d sigma_orientation = Matrix3d::Identity() - sigma_moment;

	Vector3d force_feedback_related_force = Vector3d::Zero();
	Vector3d position_related_force = Vector3d::Zero();
	Vector3d moment_feedback_related_force = Vector3d::Zero();
	Vector3d orientation_related_force = Vector3d::Zero();

	// update controller state
	_current_wrist_position =
		_robot->position(_link_name, _wrist_frame_in_link.translation());
	_current_wrist_orientation = _robot->rotation(_link_name, _wrist_frame_in_link.rotation());
	_current_wrist_velocity =
		_projected_jacobian.block(0, 0, 3, _robot->dof()) * _robot->dq();
	_current_wrist_angular_velocity =
		_projected_jacobian.block(3, 0, 3, _robot->dof()) * _robot->dq();

	// compute desired state at wrist point
	Vector3d compliant_frame_to_wrist_in_base_coordinates = - _current_wrist_orientation * _wrist_to_compliant_frame_origin;
	Vector3d desired_wrist_position = _desired_compliant_frame_position + compliant_frame_to_wrist_in_base_coordinates;
	Matrix3d desired_wrist_orientation = _desired_compliant_frame_orientation;

	Vector3d desired_wrist_linear_velocity =
		_desired_compliant_frame_velocity +
		_desired_compliant_frame_angular_velocity.cross(
			compliant_frame_to_wrist_in_base_coordinates);
	Vector3d desired_wrist_angular_velocity =
		_desired_compliant_frame_angular_velocity;

	Vector3d desired_wrist_linear_acceleration =
		_desired_compliant_frame_acceleration +
		_desired_compliant_frame_angular_acceleration.cross(
			compliant_frame_to_wrist_in_base_coordinates) +
		_desired_compliant_frame_angular_velocity.cross(
			_desired_compliant_frame_angular_velocity.cross(
				compliant_frame_to_wrist_in_base_coordinates));
	Vector3d desired_wrist_angular_acceleration =
		_desired_compliant_frame_angular_acceleration;

	Vector3d desired_force_wrist = getDesiredForce();
	Vector3d desired_moment_wrist = getDesiredMoment() - compliant_frame_to_wrist_in_base_coordinates.cross(getDesiredForce());
	
	// force related terms
	if (_closed_loop_force_control) {
		// update the integrated error
		_integrated_force_error +=
			(_sensed_force_at_wrist - desired_force_wrist) * _loop_timestep;

		// compute the feedback term and saturate it
		Vector3d force_feedback_term =
			-_kp_force * (_sensed_force_at_wrist - desired_force_wrist) -
			_ki_force * _integrated_force_error;
		if (force_feedback_term.norm() > MAX_FEEDBACK_FORCE_FORCE_CONTROLLER) {
			force_feedback_term *= MAX_FEEDBACK_FORCE_FORCE_CONTROLLER /
								   force_feedback_term.norm();
		}

		// compute the final contribution
		force_feedback_related_force =
			_POPC_force->computePassivitySaturatedForce(
				sigma_force * desired_force_wrist, sigma_force * _sensed_force_at_wrist,
				sigma_force * force_feedback_term,
				sigma_force * _current_wrist_velocity, _kv_force, _k_ff);
	} else	// open loop force control
	{
		force_feedback_related_force =
			sigma_force * (-_kv_force * _current_wrist_velocity);
	}

	// moment related terms
	if (_closed_loop_moment_control) {
		// update the integrated error
		_integrated_moment_error +=
			(_sensed_moment_at_wrist - desired_moment_wrist) * _loop_timestep;

		// compute the feedback term
		Vector3d moment_feedback_term =
			-_kp_moment * (_sensed_moment_at_wrist - desired_moment_wrist) -
			_ki_moment * _integrated_moment_error;

		// saturate the feedback term
		if (moment_feedback_term.norm() >
			MAX_FEEDBACK_MOMENT_FORCE_CONTROLLER) {
			moment_feedback_term *= MAX_FEEDBACK_MOMENT_FORCE_CONTROLLER /
									moment_feedback_term.norm();
		}

		// compute the final contribution
		moment_feedback_related_force =
			sigma_moment *
			(moment_feedback_term - _kv_moment * _current_wrist_angular_velocity);
	} else	// open loop moment control
	{
		moment_feedback_related_force =
			sigma_moment * (-_kv_moment * _current_wrist_angular_velocity);
	}

	// motion related terms
	// compute next state from trajectory generation
	if(_use_internal_otg_flag) {
		_otg->setGoalPositionAndLinearVelocity(desired_wrist_position,
											   desired_wrist_linear_velocity);
		_otg->setGoalOrientationAndAngularVelocity(desired_wrist_orientation,
												   desired_wrist_angular_velocity);
		_otg->update();
		
		desired_wrist_position = _otg->getNextPosition();
		desired_wrist_linear_velocity = _otg->getNextLinearVelocity();
		desired_wrist_linear_acceleration = _otg->getNextLinearAcceleration();
		desired_wrist_orientation = _otg->getNextOrientation();
		desired_wrist_angular_velocity = _otg->getNextAngularVelocity();
		desired_wrist_angular_acceleration = _otg->getNextAngularAcceleration();
	}

	// linear motion
	// update integrated error for I term
	_integrated_position_error +=
		(_current_wrist_position - desired_wrist_position) * _loop_timestep;

	// final contribution
	if (_use_velocity_saturation_flag) {
		desired_wrist_linear_velocity =
			-_kp_pos * _kv_pos.inverse() *
				(_current_wrist_position - desired_wrist_position) -
			_ki_pos * _kv_pos.inverse() * _integrated_position_error;
		if (desired_wrist_linear_velocity.norm() > _linear_saturation_velocity) {
			desired_wrist_linear_velocity *=
				_linear_saturation_velocity / desired_wrist_linear_velocity.norm();
		}
		position_related_force =
			sigma_position *
			(desired_wrist_linear_acceleration -
			 _kv_pos * (_current_wrist_velocity - desired_wrist_linear_velocity));
	} else {
		position_related_force =
			sigma_position *
			(desired_wrist_linear_acceleration -
			 _kp_pos * (_current_wrist_position - desired_wrist_position) -
			 _kv_pos * (_current_wrist_velocity - desired_wrist_linear_velocity) -
			 _ki_pos * _integrated_position_error);
	}

	// angular motion
	// orientation error
	Vector3d orientation_error = Sai2Model::orientationError(
		desired_wrist_orientation, _current_wrist_orientation);

	// update integrated error for I term
	_integrated_orientation_error += orientation_error * _loop_timestep;

	// final contribution
	if (_use_velocity_saturation_flag) {
		desired_wrist_angular_velocity =
			-_kp_ori * _kv_ori.inverse() * orientation_error -
			_ki_ori * _kv_ori.inverse() * _integrated_orientation_error;
		if (desired_wrist_angular_velocity.norm() >
			_angular_saturation_velocity) {
			desired_wrist_angular_velocity *=
				_angular_saturation_velocity /
				desired_wrist_angular_velocity.norm();
		}
		orientation_related_force =
			sigma_orientation * (desired_wrist_angular_acceleration -
								 _kv_ori * (_current_wrist_angular_velocity -
											desired_wrist_angular_velocity));
	} else {
		orientation_related_force =
			sigma_orientation * (desired_wrist_angular_acceleration -
								 _kp_ori * orientation_error -
								 _kv_ori * (_current_wrist_angular_velocity -
											desired_wrist_angular_velocity) -
								 _ki_ori * _integrated_orientation_error);
	}

	// compute task force
	VectorXd force_moment_contribution(6), position_orientation_contribution(6);
	force_moment_contribution.head(3) = force_feedback_related_force;
	force_moment_contribution.tail(3) = moment_feedback_related_force;

	position_orientation_contribution.head(3) = position_related_force;
	position_orientation_contribution.tail(3) = orientation_related_force;

	VectorXd feedforward_force_moment = VectorXd::Zero(6);
	feedforward_force_moment.head(3) = sigma_force * desired_force_wrist;
	feedforward_force_moment.tail(3) = sigma_moment * desired_moment_wrist;

	if (_closed_loop_force_control) {
		feedforward_force_moment *= _k_ff;
	}

	_linear_force_control =
		force_feedback_related_force + feedforward_force_moment.head(3);
	_linear_motion_control = position_related_force;

	_task_force = _Lambda_modified * _URange.transpose() *
					  (position_orientation_contribution) +
				  _URange.transpose() *
					  (force_moment_contribution + feedforward_force_moment);

	// compute task torques
	task_joint_torques =
		_projected_jacobian.transpose() * _URange * _task_force;

	return task_joint_torques;
}

void MotionForceTask::enableInternalOtgAccelerationLimited(
	const double max_linear_velelocity, const double max_linear_acceleration,
	const double max_angular_velocity, const double max_angular_acceleration) {
	_otg->setMaxLinearVelocity(max_linear_velelocity);
	_otg->setMaxLinearAcceleration(max_linear_acceleration);
	_otg->setMaxAngularVelocity(max_angular_velocity);
	_otg->setMaxAngularAcceleration(max_angular_acceleration);
	_otg->disableJerkLimits();
	_use_internal_otg_flag = true;
}

void MotionForceTask::enableInternalOtgJerkLimited(
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

Vector3d MotionForceTask::getDesiredForce() const {
	return _is_force_parametrization_in_compliant_frame
			   ? getCompliantFrameInRobotBase().linear() * _desired_compliant_frame_force
			   : _desired_compliant_frame_force;
}

Vector3d MotionForceTask::getDesiredMoment() const {
	return _is_force_parametrization_in_compliant_frame
			   ? getCompliantFrameInRobotBase().linear() * _desired_compliant_frame_moment
			   : _desired_compliant_frame_moment;
}

void MotionForceTask::setPosControlGains(double kp_pos, double kv_pos,
										 double ki_pos) {
	if (kp_pos < 0 || kv_pos < 0 || ki_pos < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"MotionForceTask::setPosControlGains\n");
	}
	if (kv_pos < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_pos = 0 if using velocity saturation in "
			"MotionForceTask::setPosControlGains\n");
	}
	_are_pos_gains_isotropic = true;
	_kp_pos = kp_pos * Matrix3d::Identity();
	_kv_pos = kv_pos * Matrix3d::Identity();
	_ki_pos = ki_pos * Matrix3d::Identity();
}

void MotionForceTask::setPosControlGains(const Vector3d& kp_pos,
										 const Vector3d& kv_pos,
										 const Vector3d& ki_pos) {
	if (kp_pos.minCoeff() < 0 || kv_pos.minCoeff() < 0 ||
		ki_pos.minCoeff() < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"MotionForceTask::setPosControlGains\n");
	}
	if (kv_pos.minCoeff() < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_pos = 0 if using velocity saturation in "
			"MotionForceTask::setPosControlGains\n");
	}
	_are_pos_gains_isotropic = false;
	Matrix3d rotation = _is_force_parametrization_in_compliant_frame
							? _wrist_frame_in_link.rotation()
							: Matrix3d::Identity();
	_kp_pos = rotation * kp_pos.asDiagonal() * rotation.transpose();
	_kv_pos = rotation * kv_pos.asDiagonal() * rotation.transpose();
	_ki_pos = rotation * ki_pos.asDiagonal() * rotation.transpose();
}

vector<PIDGains> MotionForceTask::getPosControlGains() const {
	if (_are_pos_gains_isotropic) {
		return vector<PIDGains>(
			1, PIDGains(_kp_pos(0, 0), _kv_pos(0, 0), _ki_pos(0, 0)));
	}
	Matrix3d rotation = _is_force_parametrization_in_compliant_frame
							? _wrist_frame_in_link.rotation()
							: Matrix3d::Identity();
	Vector3d aniso_kp_robot_base =
		(rotation.transpose() * _kp_pos * rotation).diagonal();
	Vector3d aniso_kv_robot_base =
		(rotation.transpose() * _kv_pos * rotation).diagonal();
	Vector3d aniso_ki_robot_base =
		(rotation.transpose() * _ki_pos * rotation).diagonal();
	return vector<PIDGains>{
		PIDGains(aniso_kp_robot_base(0), aniso_kv_robot_base(0),
				 aniso_ki_robot_base(0)),
		PIDGains(aniso_kp_robot_base(1), aniso_kv_robot_base(1),
				 aniso_ki_robot_base(1)),
		PIDGains(aniso_kp_robot_base(2), aniso_kv_robot_base(2),
				 aniso_ki_robot_base(2))};
}

void MotionForceTask::setOriControlGains(double kp_ori, double kv_ori,
										 double ki_ori) {
	if (kp_ori < 0 || kv_ori < 0 || ki_ori < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"MotionForceTask::setOriControlGains\n");
	}
	if (kv_ori < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_ori = 0 if using velocity saturation in "
			"MotionForceTask::setOriControlGains\n");
	}
	_are_ori_gains_isotropic = true;
	_kp_ori = kp_ori * Matrix3d::Identity();
	_kv_ori = kv_ori * Matrix3d::Identity();
	_ki_ori = ki_ori * Matrix3d::Identity();
}

void MotionForceTask::setOriControlGains(const Vector3d& kp_ori,
										 const Vector3d& kv_ori,
										 const Vector3d& ki_ori) {
	if (kp_ori.minCoeff() < 0 || kv_ori.minCoeff() < 0 ||
		ki_ori.minCoeff() < 0) {
		throw invalid_argument(
			"all gains should be positive or zero in "
			"MotionForceTask::setOriControlGains\n");
	}
	if (kv_ori.minCoeff() < 1e-2 && _use_velocity_saturation_flag) {
		throw invalid_argument(
			"cannot have kv_ori = 0 if using velocity saturation in "
			"MotionForceTask::setOriControlGains\n");
	}
	_are_ori_gains_isotropic = false;
	Matrix3d rotation = _is_force_parametrization_in_compliant_frame
							? _wrist_frame_in_link.rotation()
							: Matrix3d::Identity();
	_kp_ori = rotation * kp_ori.asDiagonal() * rotation.transpose();
	_kv_ori = rotation * kv_ori.asDiagonal() * rotation.transpose();
	_ki_ori = rotation * ki_ori.asDiagonal() * rotation.transpose();
}

vector<PIDGains> MotionForceTask::getOriControlGains() const {
	if (_are_ori_gains_isotropic) {
		return vector<PIDGains>(
			1, PIDGains(_kp_ori(0, 0), _kv_ori(0, 0), _ki_ori(0, 0)));
	}
	Matrix3d rotation = _is_force_parametrization_in_compliant_frame
							? _wrist_frame_in_link.rotation()
							: Matrix3d::Identity();
	Vector3d aniso_kp_robot_base =
		(rotation.transpose() * _kp_ori * rotation).diagonal();
	Vector3d aniso_kv_robot_base =
		(rotation.transpose() * _kv_ori * rotation).diagonal();
	Vector3d aniso_ki_robot_base =
		(rotation.transpose() * _ki_ori * rotation).diagonal();
	return vector<PIDGains>{
		PIDGains(aniso_kp_robot_base(0), aniso_kv_robot_base(0),
				 aniso_ki_robot_base(0)),
		PIDGains(aniso_kp_robot_base(1), aniso_kv_robot_base(1),
				 aniso_ki_robot_base(1)),
		PIDGains(aniso_kp_robot_base(2), aniso_kv_robot_base(2),
				 aniso_ki_robot_base(2))};
}

void MotionForceTask::enableVelocitySaturation(const double linear_vel_sat,
											   const double angular_vel_sat) {
	if (linear_vel_sat <= 0 || angular_vel_sat <= 0) {
		throw invalid_argument(
			"Velocity saturation values should be strictly positive or zero in "
			"MotionForceTask::enableVelocitySaturation\n");
	}
	if (_kv_pos.determinant() < 1e-3) {
		throw invalid_argument(
			"Cannot enable velocity saturation if kv_pos is singular in "
			"MotionForceTask::enableVelocitySaturation\n");
	}
	if (_kv_ori.determinant() < 1e-3) {
		throw invalid_argument(
			"Cannot enable velocity saturation if kv_ori is singular in "
			"MotionForceTask::enableVelocitySaturation\n");
	}
	_use_velocity_saturation_flag = true;
	_linear_saturation_velocity = linear_vel_sat;
	_angular_saturation_velocity = angular_vel_sat;
}

void MotionForceTask::updateSensedForceAndMoment(
	const Vector3d sensed_force_sensor_frame,
	const Vector3d sensed_moment_sensor_frame) {
	// find the transform from base frame to control frame
	Affine3d T_base_wrist =
		_robot->transform(_link_name, _wrist_frame_in_link.translation(),
						  _wrist_frame_in_link.rotation());
	Affine3d T_wrist_sensor = _wrist_frame_in_link.inverse() * _force_sensor_frame_in_link;

	// find the resolved sensed force and moment in wrist point frame
	_sensed_force_at_wrist = T_wrist_sensor.rotation() * sensed_force_sensor_frame;
	_sensed_moment_at_wrist =
		T_wrist_sensor.translation().cross(_sensed_force_at_wrist) +
		T_wrist_sensor.rotation() * sensed_moment_sensor_frame;

	// rotate the quantities in base frame
	_sensed_force_at_wrist = T_base_wrist.rotation() * _sensed_force_at_wrist;
	_sensed_moment_at_wrist = T_base_wrist.rotation() * _sensed_moment_at_wrist;
}

void MotionForceTask::parametrizeForceMotionSpaces(
	const int force_space_dimension,
	const Vector3d& force_or_motion_single_axis) {
	if (force_space_dimension < 0 || force_space_dimension > 3) {
		throw invalid_argument(
			"Force space dimension should be between 0 and 3 in "
			"MotionForceTask::parametrizeForceMotionSpaces\n");
	}
	_force_space_dimension = force_space_dimension;
	if (force_space_dimension == 1 || force_space_dimension == 2) {
		if (force_or_motion_single_axis.norm() < 1e-2) {
			throw invalid_argument(
				"Force or motion axis should be a non singular vector in "
				"MotionForceTask::parametrizeForceMotionSpaces\n");
		}
		_force_or_motion_axis = force_or_motion_single_axis.normalized();
	}
	resetIntegratorsLinear();
}

void MotionForceTask::parametrizeMomentRotMotionSpaces(
	const int moment_space_dimension,
	const Vector3d& moment_or_rot_motion_single_axis) {
	if (moment_space_dimension < 0 || moment_space_dimension > 3) {
		throw invalid_argument(
			"Moment space dimension should be between 0 and 3 in "
			"MotionForceTask::parametrizeMomentRotMotionSpaces\n");
	}
	_moment_space_dimension = moment_space_dimension;
	if (moment_space_dimension == 1 || moment_space_dimension == 2) {
		if (moment_or_rot_motion_single_axis.norm() < 1e-2) {
			throw invalid_argument(
				"Moment or rot motion axis should be a non singular vector in "
				"MotionForceTask::parametrizeMomentRotMotionSpaces\n");
		}
		_moment_or_rotmotion_axis =
			moment_or_rot_motion_single_axis.normalized();
	}
	resetIntegratorsAngular();
}

Matrix3d MotionForceTask::sigmaForce() const {
	Matrix3d rotation = _is_force_parametrization_in_compliant_frame
							? _wrist_frame_in_link.rotation()
							: Matrix3d::Identity();
	switch (_force_space_dimension) {
		case 0:
			return Matrix3d::Zero();
			break;
		case 1:
			return rotation * _force_or_motion_axis *
				   _force_or_motion_axis.transpose() * rotation.transpose();
			break;
		case 2:
			return Matrix3d::Identity() -
				   rotation * _force_or_motion_axis *
					   _force_or_motion_axis.transpose() * rotation.transpose();
			break;
		case 3:
			return Matrix3d::Identity();
			break;

		default:
			// should never happen
			throw invalid_argument(
				"Force space dimension should be between 0 and 3 in "
				"MotionForceTask::sigmaForce\n");
			break;
	}
}

Matrix3d MotionForceTask::sigmaMoment() const {
	Matrix3d rotation = _is_force_parametrization_in_compliant_frame
							? _wrist_frame_in_link.rotation()
							: Matrix3d::Identity();
	switch (_moment_space_dimension) {
		case 0:
			return Matrix3d::Zero();
			break;
		case 1:
			return rotation * _moment_or_rotmotion_axis *
				   _moment_or_rotmotion_axis.transpose() * rotation.transpose();
			break;
		case 2:
			return Matrix3d::Identity() -
				   rotation * _moment_or_rotmotion_axis *
					   _moment_or_rotmotion_axis.transpose() *
					   rotation.transpose();
			break;
		case 3:
			return Matrix3d::Identity();
			break;

		default:
			// should never happen
			throw invalid_argument(
				"Moment space dimension should be between 0 and 3 in "
				"MotionForceTask::sigmaMoment\n");
			break;
	}
}

void MotionForceTask::resetIntegrators() {
	resetIntegratorsLinear();
	resetIntegratorsAngular();
}

void MotionForceTask::resetIntegratorsLinear() {
	_integrated_position_error.setZero();
	_integrated_force_error.setZero();
}

void MotionForceTask::resetIntegratorsAngular() {
	_integrated_orientation_error.setZero();
	_integrated_moment_error.setZero();
}

} /* namespace Sai2Primitives */
