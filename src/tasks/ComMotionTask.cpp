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
	std::shared_ptr<Sai2Model::Sai2Model>& robot, 
	const std::string& task_name,
	const double loop_timestep)
	: TemplateTask(robot, task_name, TaskType::COM_MOTION_TASK, loop_timestep) {
	initialSetup();
}

void ComMotionTask::initialSetup() {
	setDynamicDecouplingType(FULL_DYNAMIC_DECOUPLING);

	int dof = getConstRobotModel()->dof();

	// motion
    _current_position = getConstRobotModel()->comPosition();

	// default values for gains and velocity saturation
	setPosControlGains(50.0, 14.0, 0.0);

	disableVelocitySaturation();
	_linear_saturation_velocity = 0;

	// initialize matrices sizes
	_jacobian.setZero(3, dof);
	_projected_jacobian.setZero(3, dof);
	_Lambda.setZero(3, 3);
	_Lambda_modified.setZero(3, 3);
	_Jbar.setZero(dof, 3);
	_N.setZero(dof, dof);
	_N_prec = MatrixXd::Identity(dof, dof);

	// trajectory generation
	_otg = make_unique<OTG_6dof_cartesian>(
		_current_position, Matrix3d::Identity(), getLoopTimestep());
	enableInternalOtgAccelerationLimited(0.3, 1.0, M_PI / 3, M_PI);

	reInitializeTask();
}

void ComMotionTask::reInitializeTask() {
	int dof = getConstRobotModel()->dof();

	// motion
    _current_position = getConstRobotModel()->comPosition();
	_desired_position = _current_position;
	_current_velocity.setZero();
	_desired_velocity.setZero();
	_desired_acceleration.setZero();
	_integrated_position_error.setZero();

	resetIntegrators();

	_task_force.setZero(3);
	_unit_mass_force.setZero(3);

	_otg->reInitialize(_current_position, Matrix3d::Identity());
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
	_jacobian = getConstRobotModel()->comJacobian();
	_projected_jacobian = _jacobian * _N_prec;

	Sai2Model::OpSpaceMatrices op_space_matrices =
		getConstRobotModel()->operationalSpaceMatrices(_projected_jacobian);
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
			_Lambda_modified = MatrixXd::Identity(3, 3);
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
    _jacobian = getConstRobotModel()->comJacobian();
    _projected_jacobian = _jacobian * _N_prec;

	// update controller state
    _current_position = getConstRobotModel()->comPosition();
	_current_velocity = _projected_jacobian * getConstRobotModel()->dq();

	if (_pos_range + _ori_range == 0) {
		// there is no controllable degree of freedom for the task, just return
		// zero torques. should maybe print a warning here
		return task_joint_torques;
	}

	Vector3d force_feedback_related_force = Vector3d::Zero();
	Vector3d position_related_force = Vector3d::Zero();
	Vector3d moment_feedback_related_force = Vector3d::Zero();
	Vector3d orientation_related_force = Vector3d::Zero();

	Matrix3d kp_pos = _kp_pos * Matrix3d::Identity();
	Matrix3d kv_pos = _kv_pos * Matrix3d::Identity();
	Matrix3d ki_pos = _ki_pos * Matrix3d::Identity();

	// motion related terms
	// compute next state from trajectory generation
	Vector3d tmp_desired_position = _desired_position;
	Matrix3d tmp_desired_orientation = Matrix3d::Identity();
	Vector3d tmp_desired_velocity = _desired_velocity;
	Vector3d tmp_desired_acceleration = _desired_acceleration;

	if (_use_internal_otg_flag) {
		_otg->setGoalPositionAndLinearVelocity(_desired_position,
											   _desired_velocity);
		_otg->setGoalOrientationAndAngularVelocity(Matrix3d::Identity(),
												   Vector3d::Zero());
		_otg->update();

		tmp_desired_position = _otg->getNextPosition();
		tmp_desired_velocity = _otg->getNextLinearVelocity();
		tmp_desired_acceleration = _otg->getNextLinearAcceleration();
	}

	// linear motion
	// update integrated error for I term
	_integrated_position_error += (_current_position - tmp_desired_position) * getLoopTimestep();

	// final contribution
	if (_use_velocity_saturation_flag) {
		tmp_desired_velocity =
			-_kp_pos * _kv_pos.inverse() * (_current_position - tmp_desired_position) -
			_ki_pos * _kv_pos.inverse() * _integrated_position_error;
		if (tmp_desired_velocity.norm() > _linear_saturation_velocity) {
			tmp_desired_velocity *=
				_linear_saturation_velocity / tmp_desired_velocity.norm();
		}
		position_related_force =
			(tmp_desired_acceleration -
			 _kv_pos * (_current_velocity - tmp_desired_velocity));
	} else {
		position_related_force =
			(tmp_desired_acceleration -
			 _kp_pos * (_current_position - tmp_desired_position) -
			 _kv_pos * (_current_velocity - tmp_desired_velocity) -
			 _ki_pos * _integrated_position_error);
	}

    _task_force = _Lambda * position_related_force;

	// compute task torques
	task_joint_torques = _projected_jacobian.transpose() * _task_force;

	_unit_mass_force = position_related_force;

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

bool ComMotionTask::goalPositionReached(const double tolerance,
										  const bool verbose) {
	double position_error =
		(_desired_position - _current_position).transpose() *
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
	Vector3d aniso_kp_robot_base =
		_kp_pos.diagonal();
	Vector3d aniso_kv_robot_base =
		_kv_pos.diagonal();
	Vector3d aniso_ki_robot_base =
		_ki_pos.diagonal();
	return vector<PIDGains>{
		PIDGains(aniso_kp_robot_base(0), aniso_kv_robot_base(0),
				 aniso_ki_robot_base(0)),
		PIDGains(aniso_kp_robot_base(1), aniso_kv_robot_base(1),
				 aniso_ki_robot_base(1)),
		PIDGains(aniso_kp_robot_base(2), aniso_kv_robot_base(2),
				 aniso_ki_robot_base(2))};
}

void ComMotionTask::enableVelocitySaturation(const double linear_vel_sat) {
	if (linear_vel_sat <= 0) {
		throw invalid_argument(
			"Velocity saturation values should be strictly positive or zero in "
			"ComMotionTask::enableVelocitySaturation\n");
	}
	if (_kv_pos.determinant() < 1e-3) {
		throw invalid_argument(
			"Cannot enable velocity saturation if kv_pos is singular in "
			"ComMotionTask::enableVelocitySaturation\n");
	}
	_use_velocity_saturation_flag = true;
	_linear_saturation_velocity = linear_vel_sat;
}

void ComMotionTask::resetIntegrators() {
	resetIntegratorsLinear();
}

void ComMotionTask::resetIntegratorsLinear() {
	_integrated_position_error.setZero();
	_integrated_force_error.setZero();
}

} /* namespace Sai2Primitives */
