/*
 * MotionForceTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "MotionForceTask.h"

#include <stdexcept>

using namespace std;
using namespace Eigen;

namespace Sai2Primitives
{


MotionForceTask::MotionForceTask(Sai2Model::Sai2Model* robot, 
			const string link_name, 
			const Affine3d control_frame,
			const double loop_timestep) :
	MotionForceTask(robot, link_name, control_frame.translation(), control_frame.linear(), loop_timestep) {}

MotionForceTask::MotionForceTask(Sai2Model::Sai2Model* robot, 
			const string link_name, 
			const Vector3d pos_in_link, 
			const Matrix3d rot_in_link,
			const double loop_timestep)
{

	_loop_timestep = loop_timestep;
	Affine3d control_frame = Affine3d::Identity();
	control_frame.linear() = rot_in_link;
	control_frame.translation() = pos_in_link;

	_robot = robot;
	_link_name = link_name;
	_control_frame = control_frame;

	int dof = _robot->dof();

	_T_control_to_sensor = Affine3d::Identity();

	// POPC force
	_POPC_force.reset(new POPCExplicitForceControl(_loop_timestep));

	// motion
	_current_position = _robot->position(_link_name, _control_frame.translation());
	_current_orientation = _robot->rotation(_link_name);

	// default values for gains and velocity saturation
	_kp_pos = 50.0;
	_kv_pos = 14.0;
	_ki_pos = 0.0;
	_kp_ori = 50.0;
	_kv_ori = 14.0;
	_ki_ori = 0.0;

	_use_isotropic_gains_position = true;
	_use_isotropic_gains_orientation = true;
	_kp_pos_mat = Matrix3d::Zero();
	_kv_pos_mat = Matrix3d::Zero();
	_ki_pos_mat = Matrix3d::Zero();
	_kp_ori_mat = Matrix3d::Zero();
	_kv_ori_mat = Matrix3d::Zero();
	_ki_ori_mat = Matrix3d::Zero();

	_kp_force = 0.7;
	_kv_force = 10.0;
	_ki_force = 1.3;
	_kp_moment = 0.7;
	_kv_moment = 10.0;
	_ki_moment = 1.3;

	_use_velocity_saturation_flag = false;
	_linear_saturation_velocity = 0.3;
	_angular_saturation_velocity = M_PI/3;

	_k_ff = 1.0;

	// initialize matrices sizes
	_jacobian.setZero(6,dof);
	_projected_jacobian.setZero(6,dof);
	_prev_projected_jacobian.setZero(6,dof);
	_Lambda.setZero(6,6);
	_Lambda_modified.setZero(6,6);
	_Jbar.setZero(dof,6);
	_N.setZero(dof,dof);
	_N_prec = MatrixXd::Identity(dof,dof);

	_URange_pos = MatrixXd::Identity(3,3);
	_URange_ori = MatrixXd::Identity(3,3);
	_URange = MatrixXd::Identity(6,6);

	_pos_dof = 3;
	_ori_dof = 3;

#ifdef USING_OTG
	_use_interpolation_flag = true; 
	_otg = new OTG_posori(_current_position, _current_orientation, _loop_timestep);

	_otg->setMaxLinearVelocity(0.3);
	_otg->setMaxLinearAcceleration(1.0);
	_otg->setMaxLinearJerk(3.0);

	_otg->setMaxAngularVelocity(M_PI/3);
	_otg->setMaxAngularAcceleration(M_PI);
	_otg->setMaxAngularJerk(3*M_PI);
#endif
	reInitializeTask();
}

void MotionForceTask::reInitializeTask()
{
	int dof = _robot->dof();

	// motion
	_current_position = _robot->position(_link_name, _control_frame.translation());
	_desired_position = _robot->position(_link_name, _control_frame.translation());
	_current_orientation = _robot->rotation(_link_name);
	_desired_orientation = _robot->rotation(_link_name);

	_current_velocity.setZero();
	_desired_velocity.setZero();
	_current_angular_velocity.setZero();
	_desired_angular_velocity.setZero();
	_desired_acceleration.setZero();
	_desired_angular_acceleration.setZero();

	_orientation_error.setZero();
	_integrated_position_error.setZero();
	_integrated_orientation_error.setZero();

	_step_desired_position = _desired_position;
	_step_desired_velocity = _desired_velocity;
	_step_desired_orientation = _desired_orientation;
	_step_orientation_error.setZero();
	_step_desired_angular_velocity.setZero();
	_step_desired_acceleration.setZero();
	_step_desired_angular_acceleration.setZero();

	_sigma_position = Matrix3d::Identity();
	_sigma_orientation = Matrix3d::Identity();

	_desired_force.setZero();
	_sensed_force.setZero();
	_desired_moment.setZero();
	_sensed_moment.setZero();

	_integrated_force_error.setZero();
	_integrated_moment_error.setZero();

	_sigma_force.setZero();
	_sigma_moment.setZero();

	_closed_loop_force_control = false;
	_closed_loop_moment_control = false;

	_task_force.setZero(6);
	_unit_mass_force.setZero(6);

#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void MotionForceTask::updateTaskModel(const MatrixXd N_prec)
{
	if(N_prec.rows() != N_prec.cols())
	{
		throw invalid_argument("N_prec matrix not square in MotionForceTask::updateTaskModel\n");
	}
	if(N_prec.rows() != _robot->dof())
	{
		throw invalid_argument("N_prec matrix size not consistent with robot dof in MotionForceTask::updateTaskModel\n");
	}

	_N_prec = N_prec;

	_jacobian = _robot->J(_link_name, _control_frame.translation());
	_projected_jacobian = _jacobian * _N_prec;

	_URange_pos = Sai2Model::matrixRangeBasis(_projected_jacobian.topRows(3));
	_URange_ori = Sai2Model::matrixRangeBasis(_projected_jacobian.bottomRows(3));

	_pos_dof = _URange_pos.cols();
	_ori_dof = _URange_ori.cols();

	_URange.setZero(6, _pos_dof + _ori_dof);
	_URange.block(0,0,3,_pos_dof) = _URange_pos;
	_URange.block(3,_pos_dof,3,_ori_dof) = _URange_ori;

	Sai2Model::OpSpaceMatrices op_space_matrices =
		_robot->operationalSpaceMatrices(
			_URange.transpose() * _projected_jacobian, _N_prec);
	_Lambda = op_space_matrices.Lambda;
	_Jbar = op_space_matrices.Jbar;
	_N = op_space_matrices.N;

	switch(_dynamic_decoupling_type)
	{
		case FULL_DYNAMIC_DECOUPLING :
		{
			_Lambda_modified = _Lambda;
			break;
		}

		case PARTIAL_DYNAMIC_DECOUPLING :
		{
			_Lambda_modified = _Lambda;
			_Lambda_modified.block(_pos_dof,_pos_dof, _ori_dof, _ori_dof) = MatrixXd::Identity(_ori_dof, _ori_dof);
			_Lambda_modified.block(0,_pos_dof, _pos_dof, _ori_dof) = MatrixXd::Zero(_pos_dof, _ori_dof);
			_Lambda_modified.block(_pos_dof,0, _ori_dof, _pos_dof) = MatrixXd::Zero(_ori_dof, _pos_dof);
			break;
		}

		case IMPEDANCE :
		{
			_Lambda_modified = MatrixXd::Identity(_pos_dof+_ori_dof,_pos_dof+_ori_dof);
			break;
		}

		case BOUNDED_INERTIA_ESTIMATES :
		{
			MatrixXd M_BIE = _robot->M();
			for(int i=0 ; i<_robot->dof() ; i++)
			{
				if(M_BIE(i,i) < 0.1)
				{
					M_BIE(i,i) = 0.1;
				}
			}
			MatrixXd M_inv_BIE = M_BIE.inverse();
			MatrixXd Lambda_inv_BIE = _URange.transpose() * _projected_jacobian * (M_inv_BIE * _projected_jacobian.transpose()) * _URange;
			_Lambda_modified = Lambda_inv_BIE.inverse();
			break;
		}

		default :
		{
			_Lambda_modified = _Lambda;
			break;			
		}
	}

}

VectorXd MotionForceTask::computeTorques()
{
	VectorXd task_joint_torques = VectorXd::Zero(_robot->dof());
	_jacobian = _robot->J(_link_name, _control_frame.translation());
	_projected_jacobian = _jacobian * _N_prec;

	// update matrix gains
	if(_use_isotropic_gains_position)
	{
		_kp_pos_mat = _kp_pos * Matrix3d::Identity();
		_kv_pos_mat = _kv_pos * Matrix3d::Identity();
		_ki_pos_mat = _ki_pos * Matrix3d::Identity();
	}
	if(_use_isotropic_gains_orientation)
	{
		_kp_ori_mat = _kp_ori * Matrix3d::Identity();
		_kv_ori_mat = _kv_ori * Matrix3d::Identity();
		_ki_ori_mat = _ki_ori * Matrix3d::Identity();
	}

	Vector3d force_feedback_related_force = Vector3d::Zero();
	Vector3d position_related_force = Vector3d::Zero();
	Vector3d moment_feedback_related_force = Vector3d::Zero();
	Vector3d orientation_related_force = Vector3d::Zero();

	// update controller state
	_current_position = _robot->position(_link_name, _control_frame.translation());
	_current_orientation = _robot->rotation(_link_name);
	_current_orientation = _current_orientation * _control_frame.linear(); // orientation of compliant frame in robot frame
	_orientation_error = Sai2Model::orientationError(_desired_orientation, _current_orientation);
	_current_velocity = _projected_jacobian.block(0,0,3,_robot->dof()) * _robot->dq();
	_current_angular_velocity = _projected_jacobian.block(3,0,3,_robot->dof()) * _robot->dq();

	_step_desired_position = _desired_position;
	_step_desired_velocity = _desired_velocity;
	_step_desired_acceleration = _desired_acceleration;
	_step_desired_orientation = _desired_orientation;
	_step_orientation_error = _orientation_error;
	_step_desired_angular_velocity = _desired_angular_velocity;
	_step_desired_angular_acceleration = _desired_angular_acceleration;

	// force related terms
	if(_closed_loop_force_control)
	{
		// update the integrated error
		_integrated_force_error += (_sensed_force - _desired_force) * _loop_timestep;

		// compute the feedback term
		Vector3d force_feedback_term = - _kp_force * (_sensed_force - _desired_force) - _ki_force * _integrated_force_error;
		if(force_feedback_term.norm() > 20.0)
		{
			force_feedback_term *= 20.0 / force_feedback_term.norm();
		}

		// compute the final contribution
		force_feedback_related_force =
			_POPC_force->computePassivitySaturatedForce(
				_sigma_force * _desired_force, _sigma_force * _sensed_force,
				_sigma_force * force_feedback_term,
				_sigma_force * _current_velocity, _kv_force, _k_ff);
	}
	else // open loop force control
	{
		force_feedback_related_force = _sigma_force * (- _kv_force * _current_velocity);
	}

	// moment related terms
	if(_closed_loop_moment_control)
	{
		// update the integrated error
		_integrated_moment_error += (_sensed_moment - _desired_moment) * _loop_timestep;

		// compute the feedback term
		Vector3d moment_feedback_term = - _kp_moment * (_sensed_moment - _desired_moment) - _ki_moment * _integrated_moment_error;

		// saturate the feedback term
		if(moment_feedback_term.norm() > 10.0)
		{
			moment_feedback_term *= 10.0 / moment_feedback_term.norm();
		}

		// compute the final contribution
		moment_feedback_related_force = _sigma_moment * (moment_feedback_term - _kv_moment * _current_angular_velocity);
	}
	else  // open loop moment control
	{
		moment_feedback_related_force = _sigma_moment * (- _kv_moment * _current_angular_velocity);
	}

	// motion related terms
	// compute next state from trajectory generation
#ifdef USING_OTG
	if(_use_interpolation_flag)
	{
		_otg->setGoalPositionAndLinearVelocity(_desired_position, _desired_velocity);
		_otg->setGoalOrientationAndAngularVelocity(_desired_orientation, _current_orientation, _desired_angular_velocity);
		_otg->computeNextState(_step_desired_position, _step_desired_velocity, _step_desired_acceleration,
							_step_desired_orientation, _step_desired_angular_velocity, _step_desired_angular_acceleration);
		Sai2Model::orientationError(_step_orientation_error, _step_desired_orientation, _current_orientation);
	}
#endif
	
	// linear motion
	// update integrated error for I term
	_integrated_position_error += (_current_position - _step_desired_position) * _loop_timestep;

	// final contribution
	if(_use_velocity_saturation_flag)
	{
		_step_desired_velocity = -_kp_pos_mat * _kv_pos_mat.inverse() * (_current_position - _step_desired_position) - _ki_pos_mat * _kv_pos_mat.inverse() * _integrated_position_error;
		if(_step_desired_velocity.norm() > _linear_saturation_velocity)
		{
			_step_desired_velocity *= _linear_saturation_velocity/_step_desired_velocity.norm();
		}
		position_related_force = _sigma_position * (_step_desired_acceleration -_kv_pos_mat*(_current_velocity - _step_desired_velocity));
	}
	else
	{
		position_related_force = _sigma_position*( _step_desired_acceleration - _kp_pos_mat*(_current_position - _step_desired_position) - _kv_pos_mat*(_current_velocity - _step_desired_velocity ) - _ki_pos_mat * _integrated_position_error);
	}


	// angular motion
	// update integrated error for I term
	_integrated_orientation_error += _step_orientation_error * _loop_timestep;

	// final contribution
	if(_use_velocity_saturation_flag)
	{
		_step_desired_angular_velocity = -_kp_ori_mat * _kv_ori_mat.inverse() * _step_orientation_error - _ki_ori_mat * _kv_ori_mat.inverse() * _integrated_orientation_error;
		if(_step_desired_angular_velocity.norm() > _angular_saturation_velocity)
		{
			_step_desired_angular_velocity *= _angular_saturation_velocity/_step_desired_angular_velocity.norm();
		}
		orientation_related_force = _sigma_orientation * (_step_desired_angular_acceleration - _kv_ori_mat*(_current_angular_velocity - _step_desired_angular_velocity));
	}
	else
	{
		orientation_related_force = _sigma_orientation * (_step_desired_angular_acceleration - _kp_ori_mat*_step_orientation_error - _kv_ori_mat*(_current_angular_velocity - _step_desired_angular_velocity) - _ki_ori_mat*_integrated_orientation_error);
	}

	// compute task force
	VectorXd force_moment_contribution(6), position_orientation_contribution(6);
	force_moment_contribution.head(3) = force_feedback_related_force;
	force_moment_contribution.tail(3) = moment_feedback_related_force;

	position_orientation_contribution.head(3) = position_related_force;
	position_orientation_contribution.tail(3) = orientation_related_force;

	_unit_mass_force = position_orientation_contribution;

	VectorXd feedforward_force_moment = VectorXd::Zero(6);
	feedforward_force_moment.head(3) = _sigma_force * _desired_force;
	feedforward_force_moment.tail(3) = _sigma_moment * _desired_moment;

	if(_closed_loop_force_control)
	{
		feedforward_force_moment *= _k_ff;
	}

	_linear_force_control = force_feedback_related_force + feedforward_force_moment.head(3);
	_linear_motion_control = position_related_force;

	_task_force = _Lambda_modified * _URange.transpose() * (position_orientation_contribution) + _URange.transpose() * (force_moment_contribution + feedforward_force_moment);

	// compute task torques
	task_joint_torques = _projected_jacobian.transpose() * _URange * _task_force;

	// update previous time
	_prev_projected_jacobian = _projected_jacobian;

	return task_joint_torques;
}

bool MotionForceTask::goalPositionReached(const double tolerance, const bool verbose)
{
	double position_error = (_desired_position - _current_position).transpose() * ( _URange_pos * _sigma_position * _URange_pos.transpose()) * (_desired_position - _current_position);
	position_error = sqrt(position_error);
	bool goal_reached = position_error < tolerance;
	if(verbose)
	{
		cout << "position error in MotionForceTask : " << position_error << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
}

bool MotionForceTask::goalOrientationReached(const double tolerance, const bool verbose)
{
	double orientation_error = _orientation_error.transpose() * _URange_ori * _sigma_orientation * _URange_ori.transpose() * _orientation_error;
	orientation_error = sqrt(orientation_error);
	bool goal_reached = orientation_error < tolerance;
	if(verbose)
	{
		cout << "orientation error in MotionForceTask : " << orientation_error << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
}

void MotionForceTask::setDynamicDecouplingFull()
{
	_dynamic_decoupling_type = FULL_DYNAMIC_DECOUPLING;
}

void MotionForceTask::setDynamicDecouplingPartial()
{
	_dynamic_decoupling_type = PARTIAL_DYNAMIC_DECOUPLING;
}

void MotionForceTask::setDynamicDecouplingNone()
{
	_dynamic_decoupling_type = IMPEDANCE;
}

void MotionForceTask::setDynamicDecouplingBIE()
{
	_dynamic_decoupling_type = BOUNDED_INERTIA_ESTIMATES;
}

void MotionForceTask::setNonIsotropicGainsPosition(const Matrix3d& frame, const Vector3d& kp, 
	const Vector3d& kv, const Vector3d& ki)
{
	if( (Matrix3d::Identity() - frame.transpose()*frame).norm() > 1e-3 || frame.determinant() < 0)
	{
		throw invalid_argument("not a valid right hand frame in MotionForceTask::setNonIsotropicGainsPosition\n");
	}

	_use_isotropic_gains_position = false;

	Matrix3d kp_pos_mat_tmp = Matrix3d::Zero();
	Matrix3d kv_pos_mat_tmp = Matrix3d::Zero();
	Matrix3d ki_pos_mat_tmp = Matrix3d::Zero();

	for(int i=0 ; i<3 ; i++)
	{
		kp_pos_mat_tmp(i,i) = kp(i);
		kv_pos_mat_tmp(i,i) = kv(i);
		ki_pos_mat_tmp(i,i) = ki(i);
	}

	_kp_pos_mat = frame * kp_pos_mat_tmp * frame.transpose();
	_kv_pos_mat = frame * kv_pos_mat_tmp * frame.transpose();
	_ki_pos_mat = frame * ki_pos_mat_tmp * frame.transpose();

}

void MotionForceTask::setNonIsotropicGainsOrientation(const Matrix3d& frame, const Vector3d& kp, 
	const Vector3d& kv, const Vector3d& ki)
{
	if( (Matrix3d::Identity() - frame.transpose()*frame).norm() > 1e-3 || frame.determinant() < 0)
	{
		throw invalid_argument("not a valid right hand frame in MotionForceTask::setNonIsotropicGainsOrientation\n");
	}

	_use_isotropic_gains_orientation = false;

	Matrix3d kp_ori_mat_tmp = Matrix3d::Zero();
	Matrix3d kv_ori_mat_tmp = Matrix3d::Zero();
	Matrix3d ki_ori_mat_tmp = Matrix3d::Zero();

	for(int i=0 ; i<3 ; i++)
	{
		kp_ori_mat_tmp(i,i) = kp(i);
		kv_ori_mat_tmp(i,i) = kv(i);
		ki_ori_mat_tmp(i,i) = ki(i);
	}

	_kp_ori_mat = frame * kp_ori_mat_tmp * frame.transpose();
	_kv_ori_mat = frame * kv_ori_mat_tmp * frame.transpose();
	_ki_ori_mat = frame * ki_ori_mat_tmp * frame.transpose();	
}

void MotionForceTask::setIsotropicGainsPosition(const double kp, const double kv, const double ki)
{
	_use_isotropic_gains_position = true;

	_kp_pos = kp;
	_kv_pos = kv;
	_ki_pos = ki;
}

void MotionForceTask::setIsotropicGainsOrientation(const double kp, const double kv, const double ki)
{
	_use_isotropic_gains_orientation = true;

	_kp_ori = kp;
	_kv_ori = kv;
	_ki_ori = ki;	
}

void MotionForceTask::setForceSensorFrame(const string link_name, const Affine3d transformation_in_link)
{
	if(link_name != _link_name)
	{
		throw invalid_argument("The link to which is attached the sensor should be the same as the link to which is attached the control frame in MotionForceTask::setForceSensorFrame\n");
	}
	_T_control_to_sensor = _control_frame.inverse() * transformation_in_link;
}

void MotionForceTask::updateSensedForceAndMoment(const Vector3d sensed_force_sensor_frame, 
							    		    const Vector3d sensed_moment_sensor_frame)
{
	// find the transform from base frame to control frame
	Affine3d T_base_link = _robot->transform(_link_name);
	Affine3d T_base_control = T_base_link * _control_frame;

	// find the resolved sensed force and moment in control frame
	_sensed_force = _T_control_to_sensor.rotation() * sensed_force_sensor_frame;
	_sensed_moment = _T_control_to_sensor.translation().cross(_sensed_force) + _T_control_to_sensor.rotation() * sensed_moment_sensor_frame;

	// rotate the quantities in base frame
	_sensed_force = T_base_control.rotation() * _sensed_force;
	_sensed_moment = T_base_control.rotation() * _sensed_moment;
}

void MotionForceTask::setForceAxis(const Vector3d force_axis)
{
	Vector3d normalized_axis = force_axis.normalized();

	_sigma_force = normalized_axis*normalized_axis.transpose();
	_sigma_position = Matrix3d::Identity() - _sigma_force;

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void MotionForceTask::updateForceAxis(const Vector3d force_axis)
{
	Vector3d normalized_axis = force_axis.normalized();

	_sigma_force = normalized_axis*normalized_axis.transpose();
	_sigma_position = Matrix3d::Identity() - _sigma_force;
}

void MotionForceTask::setLinearMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_position = normalized_axis*normalized_axis.transpose();
	_sigma_force = Matrix3d::Identity() - _sigma_position;

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void MotionForceTask::updateLinearMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_position = normalized_axis*normalized_axis.transpose();
	_sigma_force = Matrix3d::Identity() - _sigma_position;	
}

void MotionForceTask::setFullForceControl()
{
	_sigma_force = Matrix3d::Identity();
	_sigma_position.setZero();

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void MotionForceTask::setFullLinearMotionControl()
{
	_sigma_position = Matrix3d::Identity();
	_sigma_force.setZero();

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void MotionForceTask::setMomentAxis(const Vector3d moment_axis)
{
	Vector3d normalized_axis = moment_axis.normalized();

	_sigma_moment = normalized_axis*normalized_axis.transpose();
	_sigma_orientation = Matrix3d::Identity() - _sigma_moment;

	resetIntegratorsAngular();
}

void MotionForceTask::updateMomentAxis(const Vector3d moment_axis)
{
	Vector3d normalized_axis = moment_axis.normalized();

	_sigma_moment = normalized_axis*normalized_axis.transpose();
	_sigma_orientation = Matrix3d::Identity() - _sigma_moment;	
}

void MotionForceTask::setAngularMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_orientation = normalized_axis*normalized_axis.transpose();
	_sigma_moment = Matrix3d::Identity() - _sigma_orientation;

	resetIntegratorsAngular();
}

void MotionForceTask::updateAngularMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_orientation = normalized_axis*normalized_axis.transpose();
	_sigma_moment = Matrix3d::Identity() - _sigma_orientation;
}

void MotionForceTask::setFullMomentControl()
{
	_sigma_moment = Matrix3d::Identity();
	_sigma_orientation.setZero();

	resetIntegratorsAngular();
}

void MotionForceTask::setFullAngularMotionControl()
{
	_sigma_orientation = Matrix3d::Identity();
	_sigma_moment.setZero();

	resetIntegratorsAngular();
}

void MotionForceTask::setClosedLoopForceControl()
{
	_closed_loop_force_control = true;
	resetIntegratorsLinear();
}

void MotionForceTask::setOpenLoopForceControl()
{
	_closed_loop_force_control = false;
}

void MotionForceTask::setClosedLoopMomentControl()
{
	_closed_loop_moment_control = true;
	resetIntegratorsAngular();
}

void MotionForceTask::setOpenLoopMomentControl()
{
	_closed_loop_moment_control = false;
}

void MotionForceTask::enablePassivity()
{
	_POPC_force->enable();
}

void MotionForceTask::disablePassivity()
{
	_POPC_force->disable();
}

void MotionForceTask::resetIntegrators()
{
	_integrated_orientation_error.setZero();
	_integrated_position_error.setZero();
	_integrated_force_error.setZero();
	_integrated_moment_error.setZero();
}

void MotionForceTask::resetIntegratorsLinear()
{
	_integrated_position_error.setZero();
	_integrated_force_error.setZero();
}

void MotionForceTask::resetIntegratorsAngular()
{
	_integrated_orientation_error.setZero();
	_integrated_moment_error.setZero();
}


} /* namespace Sai2Primitives */

