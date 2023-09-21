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
}  // namespace

MotionForceTask::MotionForceTask(
	std::shared_ptr<Sai2Model::Sai2Model>& robot, const string& link_name,
	const Affine3d& compliant_frame,
	const bool is_force_motion_parametrization_in_compliant_frame,
	const double loop_timestep)
	: TemplateTask(robot, loop_timestep) {
	_link_name = link_name;
	_compliant_frame = compliant_frame;
	_is_force_motion_parametrization_in_compliant_frame =
		is_force_motion_parametrization_in_compliant_frame;
	setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

	int dof = getConstRobotModel()->dof();

	_T_control_to_sensor = Affine3d::Identity();

	// POPC force
	_POPC_force.reset(new POPCExplicitForceControl(getLoopTimestep()));

	// motion
	_current_position =
		getConstRobotModel()->position(_link_name, _compliant_frame.translation());
	_current_orientation = getConstRobotModel()->rotation(_link_name);

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
	_otg.reset(new OTG_6dof_cartesian(_current_position, _current_orientation,
									  getLoopTimestep()));
	enableInternalOtgAccelerationLimited(0.3, 1.0, M_PI / 3, M_PI);

	reInitializeTask();
}

void MotionForceTask::reInitializeTask() {
	int dof = getConstRobotModel()->dof();

	// motion
	_current_position =
		getConstRobotModel()->position(_link_name, _compliant_frame.translation());
	_desired_position = _current_position;
	_current_orientation = getConstRobotModel()->rotation(_link_name);
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

	_desired_force.setZero();
	_sensed_force.setZero();
	_desired_moment.setZero();
	_sensed_moment.setZero();

	resetIntegrators();

	_task_force.setZero(6);
	_unit_mass_force.setZero(6);

	_otg->reInitialize(_current_position, _current_orientation);
}

void MotionForceTask::updateTaskModel(const MatrixXd& N_prec) {
	if (N_prec.rows() != N_prec.cols()) {
		throw invalid_argument(
			"N_prec matrix not square in MotionForceTask::updateTaskModel\n");
	}
	if (N_prec.rows() != getConstRobotModel()->dof()) {
		throw invalid_argument(
			"N_prec matrix size not consistent with robot dof in "
			"MotionForceTask::updateTaskModel\n");
	}

	_N_prec = N_prec;

	_jacobian = getConstRobotModel()->J(_link_name, _compliant_frame.translation());
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
		getConstRobotModel()->operationalSpaceMatrices(
			_URange.transpose() * _projected_jacobian);
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
			MatrixXd M_BIE = getConstRobotModel()->M();
			for (int i = 0; i < getConstRobotModel()->dof(); i++) {
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
	VectorXd task_joint_torques = VectorXd::Zero(getConstRobotModel()->dof());
	_jacobian = getConstRobotModel()->J(_link_name, _compliant_frame.translation());
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
	_current_position =
		getConstRobotModel()->position(_link_name, _compliant_frame.translation());
	_current_orientation = getConstRobotModel()->rotation(_link_name);
	_current_orientation =
		_current_orientation *
		_compliant_frame
			.linear();	// orientation of compliant frame in robot frame
	_orientation_error =
		Sai2Model::orientationError(_desired_orientation, _current_orientation);
	_current_velocity =
		_projected_jacobian.block(0, 0, 3, getConstRobotModel()->dof()) * getConstRobotModel()->dq();
	_current_angular_velocity =
		_projected_jacobian.block(3, 0, 3, getConstRobotModel()->dof()) * getConstRobotModel()->dq();

	// force related terms
	if (_closed_loop_force_control) {
		// update the integrated error
		_integrated_force_error +=
			(_sensed_force - _desired_force) * getLoopTimestep();

		// compute the feedback term and saturate it
		Vector3d force_feedback_term =
			-_kp_force * (_sensed_force - _desired_force) -
			_ki_force * _integrated_force_error;
		if (force_feedback_term.norm() > MAX_FEEDBACK_FORCE_FORCE_CONTROLLER) {
			force_feedback_term *= MAX_FEEDBACK_FORCE_FORCE_CONTROLLER /
								   force_feedback_term.norm();
		}

		// compute the final contribution
		force_feedback_related_force =
			_POPC_force->computePassivitySaturatedForce(
				sigma_force * _desired_force, sigma_force * _sensed_force,
				sigma_force * force_feedback_term,
				sigma_force * _current_velocity, _kv_force, _k_ff);
	} else	// open loop force control
	{
		force_feedback_related_force =
			sigma_force * (-_kv_force * _current_velocity);
	}

	// moment related terms
	if (_closed_loop_moment_control) {
		// update the integrated error
		_integrated_moment_error +=
			(_sensed_moment - _desired_moment) * getLoopTimestep();

		// compute the feedback term
		Vector3d moment_feedback_term =
			-_kp_moment * (_sensed_moment - _desired_moment) -
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
			(moment_feedback_term - _kv_moment * _current_angular_velocity);
	} else	// open loop moment control
	{
		moment_feedback_related_force =
			sigma_moment * (-_kv_moment * _current_angular_velocity);
	}

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
	_integrated_position_error +=
		(_current_position - tmp_desired_position) * getLoopTimestep();

	// final contribution
	if (_use_velocity_saturation_flag) {
		tmp_desired_velocity =
			-_kp_pos * _kv_pos.inverse() *
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
	Vector3d orientation_error = Sai2Model::orientationError(
		tmp_desired_orientation, _current_orientation);

	// update integrated error for I term
	_integrated_orientation_error += orientation_error * getLoopTimestep();

	// final contribution
	if (_use_velocity_saturation_flag) {
		tmp_desired_angular_velocity =
			-_kp_ori * _kv_ori.inverse() * orientation_error -
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
			sigma_orientation *
			(tmp_desired_angular_acceleration - _kp_ori * orientation_error -
			 _kv_ori *
				 (_current_angular_velocity - tmp_desired_angular_velocity) -
			 _ki_ori * _integrated_orientation_error);
	}

	// compute task force
	VectorXd force_moment_contribution(6), position_orientation_contribution(6);
	force_moment_contribution.head(3) = force_feedback_related_force;
	force_moment_contribution.tail(3) = moment_feedback_related_force;

	position_orientation_contribution.head(3) = position_related_force;
	position_orientation_contribution.tail(3) = orientation_related_force;

	_unit_mass_force = position_orientation_contribution;

	VectorXd feedforward_force_moment = VectorXd::Zero(6);
	feedforward_force_moment.head(3) = sigma_force * _desired_force;
	feedforward_force_moment.tail(3) = sigma_moment * _desired_moment;

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

bool MotionForceTask::goalPositionReached(const double tolerance,
										  const bool verbose) {
	Matrix3d sigma_position = Matrix3d::Identity() - sigmaForce();
	double position_error =
		(_desired_position - _current_position).transpose() *
		(_URange_pos * sigma_position * _URange_pos.transpose()) *
		(_desired_position - _current_position);
	position_error = sqrt(position_error);
	bool goal_reached = position_error < tolerance;
	if (verbose) {
		cout << "position error in MotionForceTask : " << position_error
			 << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
}

bool MotionForceTask::goalOrientationReached(const double tolerance,
											 const bool verbose) {
	Matrix3d sigma_orientation = Matrix3d::Identity() - sigmaMoment();
	double orientation_error = _orientation_error.transpose() * _URange_ori *
							   sigma_orientation * _URange_ori.transpose() *
							   _orientation_error;
	orientation_error = sqrt(orientation_error);
	bool goal_reached = orientation_error < tolerance;
	if (verbose) {
		cout << "orientation error in MotionForceTask : " << orientation_error
			 << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
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
	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
							? _compliant_frame.rotation()
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
	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
							? _compliant_frame.rotation()
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
	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
							? _compliant_frame.rotation()
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
	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
							? _compliant_frame.rotation()
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

void MotionForceTask::setForceSensorFrame(
	const string link_name, const Affine3d transformation_in_link) {
	if (link_name != _link_name) {
		throw invalid_argument(
			"The link to which is attached the sensor should be the same as "
			"the link to which is attached the control frame in "
			"MotionForceTask::setForceSensorFrame\n");
	}
	_T_control_to_sensor = _compliant_frame.inverse() * transformation_in_link;
}

void MotionForceTask::updateSensedForceAndMoment(
	const Vector3d sensed_force_sensor_frame,
	const Vector3d sensed_moment_sensor_frame) {
	// find the transform from base frame to control frame
	Affine3d T_base_link = getConstRobotModel()->transform(_link_name);
	Affine3d T_base_control = T_base_link * _compliant_frame;

	// find the resolved sensed force and moment in control frame
	_sensed_force = _T_control_to_sensor.rotation() * sensed_force_sensor_frame;
	_sensed_moment =
		_T_control_to_sensor.translation().cross(_sensed_force) +
		_T_control_to_sensor.rotation() * sensed_moment_sensor_frame;

	// rotate the quantities in base frame
	_sensed_force = T_base_control.rotation() * _sensed_force;
	_sensed_moment = T_base_control.rotation() * _sensed_moment;
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
	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
							? _compliant_frame.rotation()
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
	Matrix3d rotation = _is_force_motion_parametrization_in_compliant_frame
							? _compliant_frame.rotation()
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
