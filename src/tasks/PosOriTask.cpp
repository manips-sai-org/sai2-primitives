/*
 * PosOriTask.cpp
 *
 *      Author: Mikael Jorda
 */

#include "PosOriTask.h"

#include <stdexcept>

using namespace std;
using namespace Eigen;

namespace Sai2Primitives
{


PosOriTask::PosOriTask(Sai2Model::Sai2Model* robot, 
			const string link_name, 
			const Affine3d control_frame,
			const double loop_time) :
	PosOriTask(robot, link_name, control_frame.translation(), control_frame.linear(), loop_time) {}

PosOriTask::PosOriTask(Sai2Model::Sai2Model* robot, 
			const string link_name, 
			const Vector3d pos_in_link, 
			const Matrix3d rot_in_link,
			const double loop_time)
{

	Affine3d control_frame = Affine3d::Identity();
	control_frame.linear() = rot_in_link;
	control_frame.translation() = pos_in_link;

	_robot = robot;
	_link_name = link_name;
	_control_frame = control_frame;
	_selection = MatrixXd::Identity(6, 6);  // default full control 

	int dof = _robot->_dof;

	_T_control_to_sensor = Affine3d::Identity();  

	// motion
	_robot->position(_current_position, _link_name, _control_frame.translation());
	_robot->rotation(_current_orientation, _link_name);

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
	// _Lambda.setZero(6,6);
	// _Lambda_modified.setZero(6,6);
	_Lambda.setIdentity(6,6);
	_Lambda_modified.setIdentity(6,6);
	_Jbar.setZero(dof,6);
	_N.setZero(dof,dof);
	_N_prec = MatrixXd::Identity(dof,dof);

	_URange_pos = MatrixXd::Identity(3,3);
	_URange_ori = MatrixXd::Identity(3,3);
	_URange = MatrixXd::Identity(6,6);

	_pos_dof = 3;
	_ori_dof = 3;

	_first_iteration = true;

	// obtain q dependencies 
	std::vector<int> q_dependency = robot->getLinkDependency(link_name);
	// std::unique_ptr<Sai2Primitives::PartialJointTask> _sing_joint_task(new Sai2Primitives::PartialJointTask(robot, q_dependency, loop_time));
	_sing_joint_task = new Sai2Primitives::PartialJointTask(robot, q_dependency, loop_time); 
	_sing_joint_task->reInitializeTask();
	_sing_joint_task->_use_velocity_saturation_flag = true;

	_prev_torques = VectorXd::Zero(dof);	

#ifdef USING_OTG
	_use_interpolation_flag = true; 
	_loop_time = loop_time;
	_otg = new OTG_posori(_current_position, _current_orientation, _loop_time);

	_otg->setMaxLinearVelocity(0.3);
	_otg->setMaxLinearAcceleration(1.0);
	_otg->setMaxLinearJerk(3.0);

	_otg->setMaxAngularVelocity(M_PI/3);
	_otg->setMaxAngularAcceleration(M_PI);
	_otg->setMaxAngularJerk(3*M_PI);
#endif
	reInitializeTask();
}

PosOriTask::PosOriTask(Sai2Model::Sai2Model* robot, 
	            const Matrix<double, 6, 6> selection,
	            const string link_name, 
	            const Vector3d pos_in_link, 
	            const Matrix3d rot_in_link,
	            const double loop_time) 
{
	Affine3d control_frame = Affine3d::Identity();
	control_frame.linear() = rot_in_link;
	control_frame.translation() = pos_in_link;

	_robot = robot;
	_link_name = link_name;
	_control_frame = control_frame;
	_selection = selection;

	int dof = _robot->_dof;

	_T_control_to_sensor = Affine3d::Identity();  

	// motion
	_robot->position(_current_position, _link_name, _control_frame.translation());
	_robot->rotation(_current_orientation, _link_name);

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

	_first_iteration = true;

#ifdef USING_OTG
	_use_interpolation_flag = true; 
	_loop_time = loop_time;
	_otg = new OTG_posori(_current_position, _current_orientation, _loop_time);

	_otg->setMaxLinearVelocity(0.3);
	_otg->setMaxLinearAcceleration(1.0);
	_otg->setMaxLinearJerk(3.0);

	_otg->setMaxAngularVelocity(M_PI/3);
	_otg->setMaxAngularAcceleration(M_PI);
	_otg->setMaxAngularJerk(3*M_PI);
#endif
	reInitializeTask();
}

void PosOriTask::reInitializeTask()
{
	int dof = _robot->_dof;

	// motion
	_robot->position(_current_position, _link_name, _control_frame.translation());
	_robot->position(_desired_position, _link_name, _control_frame.translation());
	_robot->rotation(_current_orientation, _link_name);
	_robot->rotation(_desired_orientation, _link_name);

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
	_open_loop_force_control_with_lambda = false;

	_passivity_enabled = true;
	_passivity_observer = 0;
	_E_correction = 0;
	_stored_energy_PO = 0;
	_PO_buffer_window = queue<double>();
	_PO_counter = _PO_max_counter;
	_vc_squared_sum = 0;
	_vc.setZero();
	_Rc = 1.0;

	_task_force.setZero(6);
	_unit_mass_force.setZero(6);
	_first_iteration = true;	

#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void PosOriTask::updateTaskModel(const MatrixXd N_prec)
{
	if(N_prec.rows() != N_prec.cols())
	{
		throw invalid_argument("N_prec matrix not square in PosOriTask::updateTaskModel\n");
	}
	if(N_prec.rows() != _robot->_dof)
	{
		throw invalid_argument("N_prec matrix size not consistent with robot dof in PosOriTask::updateTaskModel\n");
	}

	_N_prec = N_prec;

	_robot->J_0(_jacobian, _link_name, _control_frame.translation());
	_projected_jacobian = _jacobian * _N_prec;

	if (_use_lambda_truncation_flag) {
		_robot->URangeJacobian(_URange_pos, _projected_jacobian.topRows(3), _N_prec, 1e-1);
		_robot->URangeJacobian(_URange_ori, _projected_jacobian.bottomRows(3), _N_prec, 1e-1);

		_pos_dof = _URange_pos.cols();
		_ori_dof = _URange_ori.cols();

		_truncated_jacobian = false;
		if (_pos_dof != 3 || _ori_dof != 3) {
			std::cout << "(pos_dof, ori_dof): " << "(" << _pos_dof << ", " << _ori_dof << ")\n";
			_truncated_jacobian = true;
		}

		_URange.setZero(6, _pos_dof + _ori_dof);
		_URange.block(0,0,3,_pos_dof) = _URange_pos;
		_URange.block(3,_pos_dof,3,_ori_dof) = _URange_ori;

		_robot->operationalSpaceMatrices(_Lambda, _Jbar, _N, _URange.transpose() * _projected_jacobian, _N_prec);

	} else if (_use_lambda_smoothing_flag) {
		// Lambda smoothing method (Park 2012)
		_Lambda_inv = _projected_jacobian * _robot->_M_inv * _projected_jacobian.transpose();

		// eigendecomposition 
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(_Lambda_inv);  // sorted increasing e-values 
		int n_cols = 0;
		for (int i = 5; i >= 0; --i) {
			if (abs(eigensolver.eigenvalues()(i)) < _e_max) {
				n_cols = i + 1;
				_sing_flag = 1;
				break;
			}
			_sing_flag = 0;
		}
		if (_sing_flag) {
			if (n_cols == 6) {
				MatrixXd U_s = eigensolver.eigenvectors();
				MatrixXd D_s = MatrixXd::Zero(6, 6);
				VectorXd e_s = eigensolver.eigenvalues();
				for (int i = 0; i < 6; ++i) {
					if (abs(e_s(i)) < _e_min) {
						D_s(i, i) = 0;
					} else if (abs(e_s(i)) > _e_max) {
						D_s(i, i) = abs(e_s(i));
					} else {
						D_s(i, i) = abs(e_s(i)) * (0.5 + 0.5 * sin( (M_PI / (_e_max - _e_min)) * (abs(e_s(i)) - _e_min) - (M_PI / 2)));
					}
				}
				_Lambda = U_s * D_s.inverse() * U_s.transpose();
				_Jbar = _robot->_M_inv * _projected_jacobian.transpose() * _Lambda;
				_N = MatrixXd::Identity(_robot->_dof, _robot->_dof) - _Jbar * _projected_jacobian;
			} else {
				// non-singular components
				MatrixXd U_ns = eigensolver.eigenvectors().rightCols(6 - n_cols);
				VectorXd e_ns = eigensolver.eigenvalues().tail(6 - n_cols);
				MatrixXd D_ns = e_ns.asDiagonal();

				// singular components 
				MatrixXd U_s = eigensolver.eigenvectors().leftCols(n_cols);
				MatrixXd H_s = MatrixXd::Zero(n_cols, n_cols);
				VectorXd e_s = eigensolver.eigenvalues().head(n_cols);
				MatrixXd D_s = e_s.asDiagonal();
				for (int i = 0; i < n_cols; ++i) {
					if (abs(e_s(i)) < _e_min) {
						H_s(i, i) = 0;
					} else if (abs(e_s(i)) > _e_max) {
						H_s(i, i) = 1;
					} else {
						H_s(i, i) = 0.5 + 0.5 * sin( (M_PI / (_e_max - _e_min)) * (abs(e_s(i)) - _e_min) - (M_PI / 2));
					}
				}
				_Lambda_ns = U_ns * D_ns.inverse() * U_ns.transpose();
				_Lambda_s = U_s * H_s * D_s.inverse() * U_s.transpose();
				// std::cout << "Lambda_ns: \n" << _Lambda_ns << "\n";
				// std::cout << "Lambda_s: \n" << _Lambda_s << "\n";
				// std::cout << "U_s: \n" << U_s << "\n";
				// std:cout << "D_s: \n" << D_s << "\n";
				// std::cout << "H_s: \n" << H_s << "\n";
				// std::cout << "Lambda smoothing: \n" << _Lambda << "\n";
				_Lambda = _Lambda_ns + _Lambda_s;	
				_Jbar = _robot->_M_inv * _projected_jacobian.transpose() * _Lambda;
				_N = MatrixXd::Identity(_robot->_dof, _robot->_dof) - _Jbar * _projected_jacobian;			
			}
		} else {
			_Lambda = _Lambda_inv.inverse();
			_Jbar = _robot->_M_inv * _projected_jacobian.transpose() * _Lambda;
			_N = MatrixXd::Identity(_robot->_dof, _robot->_dof) - _Jbar * _projected_jacobian;
		}
	} else if (_use_lambda_interpolation_flag) {
		// Lambda interpolation method (custom)
		_Lambda_inv = _projected_jacobian * _robot->_M_inv * _projected_jacobian.transpose();

		_robot->URangeJacobian(_URange_pos, _projected_jacobian.topRows(3), _N_prec, 1e-1);
		_robot->URangeJacobian(_URange_ori, _projected_jacobian.bottomRows(3), _N_prec, 1e-1);

		_pos_dof = _URange_pos.cols();
		_ori_dof = _URange_ori.cols();

		if (_pos_dof != 3 || _ori_dof != 3) {
			_truncated_jacobian = true;
			if (_sing_transition == false) {
				_sing_transition = true;  // initiate transition here
				_sing_first_transition = true;
				_transition_cnt = 0;
				_Lambda_start = _Lambda;  
				_Lambda_end = invLambdaReduction(_Lambda_inv).inverse();
				_Lambda = _Lambda_start;  // using previous lambda first 
			} else {
				_transition_cnt++;
				_Lambda = lambdaInterpolation(_Lambda_start, _Lambda_end, _transition_cnt * 1.0 / _sing_samples);
				// _Jbar = _robot->_M_inv * (_URange.transpose() * _projected_jacobian).transpose() * _Lambda;
				// _N = MatrixXd::Identity(_robot->dof(), _robot->dof()) - _Jbar * _URange.transpose() * _projected_jacobian;
			}
		} else {
			_truncated_jacobian = false;
			// _sing_transition flag only reversed after N samples of transition
		}

		if (_transition_cnt > _sing_samples) {
			_sing_transition = false;
		}

	}

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
			MatrixXd M_BIE = _robot->_M;
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

void PosOriTask::computeTorques(VectorXd& task_joint_torques)
{
	_robot->J_0(_jacobian, _link_name, _control_frame.translation());
	_projected_jacobian = _jacobian * _N_prec;

	// get time since last call for the I term
	_t_curr = chrono::high_resolution_clock::now();
	if(_first_iteration)
	{
		_t_prev = _t_curr;
		_first_iteration = false;
	}
	_t_diff = _t_curr - _t_prev;
	
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
	_robot->position(_current_position, _link_name, _control_frame.translation());
	_robot->rotation(_current_orientation, _link_name);
	_current_orientation = _current_orientation * _control_frame.linear(); // orientation of compliant frame in robot frame
	Sai2Model::orientationError(_orientation_error, _desired_orientation, _current_orientation);
	_current_velocity = _projected_jacobian.block(0,0,3,_robot->_dof) * _robot->_dq;
	_current_angular_velocity = _projected_jacobian.block(3,0,3,_robot->_dof) * _robot->_dq;

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
		_integrated_force_error += (_sensed_force - _desired_force) * _t_diff.count();

		// compute the feedback term
		Vector3d force_feedback_term = - _kp_force * (_sensed_force - _desired_force) - _ki_force * _integrated_force_error;
		_vc = force_feedback_term;

		// saturate the feedback term
		if(_vc.norm() > 20.0)
		{
			_vc *= 20.0 / _vc.norm();
		}

		// implement passivity observer and controller
		if(_passivity_enabled)
		{
			Vector3d vc_force_space = _sigma_force * _vc;
			Vector3d vr_force_space = _sigma_force * _current_velocity;
			Vector3d F_cmd = _k_ff * _desired_force + _Rc * _vc - _kv_force * vr_force_space;
			double vc_squared = _vc.transpose() * _sigma_force * _vc;
			Vector3d f_diff = _sensed_force - _desired_force;

			// compute power input and output
			double power_input_output = (f_diff.dot(vc_force_space) - F_cmd.dot(vr_force_space)) * _t_diff.count();

			// compute stored energy (intentionally diabled)
			// _stored_energy_PO = 0.5 * _ki_force * (double) (_integrated_force_error.transpose() * _sigma_force * _integrated_force_error);

			// windowed PO
			_passivity_observer += power_input_output;
			_PO_buffer_window.push(power_input_output);

			if(_passivity_observer + _stored_energy_PO + _E_correction > 0)
			{
				while(_PO_buffer_window.size() > _PO_window_size)
				{
					if(_passivity_observer + _E_correction + _stored_energy_PO > _PO_buffer_window.front())
					{
						if(_PO_buffer_window.front() > 0)
						{
							_passivity_observer -= _PO_buffer_window.front();
						}
						_PO_buffer_window.pop();
					}
					else
					{
						break;
					}
				}
			}

			// compute PC
			if(_PO_counter <= 0)
			{
				_PO_counter = _PO_max_counter;

				double old_Rc = _Rc;
				if(_passivity_observer + _stored_energy_PO + _E_correction < 0)   // activity detected
				{
					_Rc = 1 + (_passivity_observer + _stored_energy_PO + _E_correction) / (_vc_squared_sum * _t_diff.count());

					if(_Rc > 1){_Rc = 1;}
					if(_Rc < 0){_Rc = 0;}

				}
				else    // no activity detected
				{
					_Rc = (1 + (0.1*_PO_max_counter-1)*_Rc)/(double)(0.1*_PO_max_counter);
				}

				_E_correction += (1 - old_Rc) * _vc_squared_sum * _t_diff.count();
				_vc_squared_sum = 0;
			}

			_PO_counter--;
			_vc_squared_sum += vc_squared;
		}
		else  // no passivity enabled
		{
			_Rc = 1;
		}

		// compute the final contribution
		force_feedback_related_force = _sigma_force * ( _Rc * _vc - _kv_force * _current_velocity);
	}
	else // open loop force control
	{
		force_feedback_related_force = _sigma_force * (- _kv_force * _current_velocity);
	}

	// moment related terms
	if(_closed_loop_moment_control)
	{
		// update the integrated error
		_integrated_moment_error += (_sensed_moment - _desired_moment) * _t_diff.count();

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
	_integrated_position_error += (_current_position - _step_desired_position) * _t_diff.count();

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
	_integrated_orientation_error += _step_orientation_error * _t_diff.count();

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

	// if (_open_loop_force_control_with_lambda && !_closed_loop_force_control) {
	// 	feedforward_force_moment = _Lambda_modified * feedforward_force_moment;
	// }

	if(_closed_loop_force_control)
	{
		feedforward_force_moment *= _k_ff;
	}

	_linear_force_control = force_feedback_related_force + feedforward_force_moment.head(3);
	_linear_motion_control = position_related_force;	

	_task_force = _Lambda_modified * _URange.transpose() * (position_orientation_contribution) + _URange.transpose() * (force_moment_contribution + feedforward_force_moment);

	// open loop control with added lambda (custom option)
	if (_open_loop_force_control_with_lambda && !_closed_loop_force_control) {
		_task_force = _Lambda_modified * _URange.transpose() * (position_orientation_contribution) + _Lambda_modified * _URange.transpose() * (force_moment_contribution + feedforward_force_moment);
	}

	// compute task torques
	task_joint_torques = _projected_jacobian.transpose() * _URange * _task_force;

	// compute task torques with interpolated lambda 
	if (_use_lambda_interpolation_flag) {
		_task_force = _Lambda_modified * (position_orientation_contribution) + (force_moment_contribution + feedforward_force_moment);
		task_joint_torques = _projected_jacobian.transpose() * _task_force;
	}

	// add partial joint task torques if jacobian is truncated
	if (_use_lambda_truncation_flag && _truncated_jacobian) {
		// for (int i = 0; i < _sing_joint_task->_desired_position.size(); ++i) {
			// _sing_joint_task->_desired_position(i) += 0.1 * sgn(_robot->_dq(_sing_joint_task->_active_joints[i])) * M_PI / 180;  // naive update of 0.1 degrees per control cycle in direction of last velocity
			// _sing_joint_task->_desired_position.array() += 0.1 * M_PI / 180;
		// }
		_sing_joint_task->updateTaskModel(_N);
		_sing_joint_task->computeTorques(_sing_joint_torques);
		task_joint_torques += _sing_joint_torques;
		_N = _sing_joint_task->_N;
	}

	// // compute saturated task torques for torque spike
	// for (int i = 0; i < task_joint_torques.size(); ++i) {
	// 	if (abs(_prev_torques(i) - task_joint_torques(i)) > _max_torque_diff) {
	// 		std::cout << "Torque spike saturation \n";
	// 		task_joint_torques(i) = _prev_torques(i) + _max_torque_diff * sgn(task_joint_torques(i) - _prev_torques(i));
	// 	}
	// }

	// add partial joint task torques if jacobian is truncated
	// _sing_joint_task->_desired_position = 
	// _sing_joint_task->updateTaskModel(_N);
	// _sing_joint_task->computeTorques(_sing_joint_torques);
	// _N = _sing_joint_task->_N;
	// task_joint_torques += _sing_joint_torques;

	// // overwrite task torques if in a singularity transition 
	// if (_sing_transition && _use_blending_flag) {
	// 	// std::cout << "Singularity transition: \n";
	// 	if (_sing_first_transition) {
	// 		_sing_start_torques = _prev_torques;
	// 		_sing_end_torques = task_joint_torques;
	// 		_sing_first_transition = false; 
	// 	}
	// 	double alpha = 1. / exp(_sing_blending_rate * _sing_cnt); // 1 -> 0
	// 	std::cout << "alpha: " << alpha << "\n";

	// 	// _manip_matrix = _projected_jacobian.topRows(_pos_dof) * _projected_jacobian.topRows(_pos_dof).transpose();
	// 	// _sing_joint_task->_desired_position = - _manip_matrix.determinant() * (_manip_matrix.inverse() * )
	// 	_sing_joint_task->_desired_position.array() += 1e-1;  // random exit strategy; if dJdq is available, then minimize manipulability metric 
	// 	_sing_joint_task->updateTaskModel(_N);  // uses the nullspace of the truncated jacobian task 
	// 	_sing_joint_task->computeTorques(_sing_joint_torques);
	// 	_N = _sing_joint_task->_N;

	// 	task_joint_torques = alpha * _sing_start_torques + (1 - alpha) * _sing_end_torques + _sing_joint_torques;

	// 	_sing_cnt++;
	// 	if (_sing_cnt > _sing_samples) {
	// 		_sing_transition = false;
	// 		_sing_cnt = 0;
	// 		// std::cout << "Leaving singularity transition: \n";
	// 	}
	// }

	// update previous time
	// partial joint task for singularity is updated only if the jacobian isn't truncated
	if (_use_lambda_truncation_flag == true && _truncated_jacobian == false) {
		for (int i = 0; i < _sing_joint_task->_desired_position.size(); ++i) {
			_sing_joint_task->_desired_position(i) = _robot->_q(_sing_joint_task->_active_joints[i]);  
			_sing_joint_task->_desired_velocity(i) = _robot->_dq(_sing_joint_task->_active_joints[i]);
		}
	}
	_prev_projected_jacobian = _projected_jacobian;
	_prev_torques = task_joint_torques;
	_t_prev = _t_curr;
}

bool PosOriTask::goalPositionReached(const double tolerance, const bool verbose)
{
	double position_error = (_desired_position - _current_position).transpose() * ( _URange_pos * _sigma_position * _URange_pos.transpose()) * (_desired_position - _current_position);
	position_error = sqrt(position_error);
	bool goal_reached = position_error < tolerance;
	if(verbose)
	{
		cout << "position error in PosOriTask : " << position_error << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
}

bool PosOriTask::goalOrientationReached(const double tolerance, const bool verbose)
{
	double orientation_error = _orientation_error.transpose() * _URange_ori * _sigma_orientation * _URange_ori.transpose() * _orientation_error;
	orientation_error = sqrt(orientation_error);
	bool goal_reached = orientation_error < tolerance;
	if(verbose)
	{
		cout << "orientation error in PosOriTask : " << orientation_error << endl;
		cout << "Tolerance : " << tolerance << endl;
		cout << "Goal reached : " << goal_reached << endl << endl;
	}

	return goal_reached;
}

void PosOriTask::setDynamicDecouplingFull()
{
	_dynamic_decoupling_type = FULL_DYNAMIC_DECOUPLING;
}

void PosOriTask::setDynamicDecouplingPartial()
{
	_dynamic_decoupling_type = PARTIAL_DYNAMIC_DECOUPLING;
}

void PosOriTask::setDynamicDecouplingNone()
{
	_dynamic_decoupling_type = IMPEDANCE;
}

void PosOriTask::setDynamicDecouplingBIE()
{
	_dynamic_decoupling_type = BOUNDED_INERTIA_ESTIMATES;
}

void PosOriTask::setNonIsotropicGainsPosition(const Matrix3d& frame, const Vector3d& kp, 
	const Vector3d& kv, const Vector3d& ki)
{
	if( (Matrix3d::Identity() - frame.transpose()*frame).norm() > 1e-3 || frame.determinant() < 0)
	{
		throw invalid_argument("not a valid right hand frame in PosOriTask::setNonIsotropicGainsPosition\n");
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

void PosOriTask::setNonIsotropicGainsOrientation(const Matrix3d& frame, const Vector3d& kp, 
	const Vector3d& kv, const Vector3d& ki)
{
	if( (Matrix3d::Identity() - frame.transpose()*frame).norm() > 1e-3 || frame.determinant() < 0)
	{
		throw invalid_argument("not a valid right hand frame in PosOriTask::setNonIsotropicGainsOrientation\n");
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

void PosOriTask::setIsotropicGainsPosition(const double kp, const double kv, const double ki)
{
	_use_isotropic_gains_position = true;

	_kp_pos = kp;
	_kv_pos = kv;
	_ki_pos = ki;
}

void PosOriTask::setIsotropicGainsOrientation(const double kp, const double kv, const double ki)
{
	_use_isotropic_gains_orientation = true;

	_kp_ori = kp;
	_kv_ori = kv;
	_ki_ori = ki;	
}

void PosOriTask::setForceSensorFrame(const string link_name, const Affine3d transformation_in_link)
{
	if(link_name != _link_name)
	{
		throw invalid_argument("The link to which is attached the sensor should be the same as the link to which is attached the control frame in PosOriTask::setForceSensorFrame\n");
	}
	_T_control_to_sensor = _control_frame.inverse() * transformation_in_link;
}

void PosOriTask::updateSensedForceAndMoment(const Vector3d sensed_force_sensor_frame, 
							    		    const Vector3d sensed_moment_sensor_frame)
{
	// find the transform from base frame to control frame
	Affine3d T_base_link;
	_robot->transform(T_base_link, _link_name);
	Affine3d T_base_control = T_base_link * _control_frame;

	// find the resolved sensed force and moment in control frame
	_sensed_force = _T_control_to_sensor.rotation() * sensed_force_sensor_frame;
	_sensed_moment = _T_control_to_sensor.translation().cross(_sensed_force) + _T_control_to_sensor.rotation() * sensed_moment_sensor_frame;

	// rotate the quantities in base frame
	_sensed_force = T_base_control.rotation() * _sensed_force;
	_sensed_moment = T_base_control.rotation() * _sensed_moment;
}

void PosOriTask::setForceAxis(const Vector3d force_axis)
{
	Vector3d normalized_axis = force_axis.normalized();

	_sigma_force = normalized_axis*normalized_axis.transpose();
	_sigma_position = Matrix3d::Identity() - _sigma_force;

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void PosOriTask::updateForceAxis(const Vector3d force_axis)
{
	Vector3d normalized_axis = force_axis.normalized();

	_sigma_force = normalized_axis*normalized_axis.transpose();
	_sigma_position = Matrix3d::Identity() - _sigma_force;
}

void PosOriTask::setLinearMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_position = normalized_axis*normalized_axis.transpose();
	_sigma_force = Matrix3d::Identity() - _sigma_position;

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void PosOriTask::updateLinearMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_position = normalized_axis*normalized_axis.transpose();
	_sigma_force = Matrix3d::Identity() - _sigma_position;	
}

void PosOriTask::setFullForceControl()
{
	_sigma_force = Matrix3d::Identity();
	_sigma_position.setZero();

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void PosOriTask::setFullLinearMotionControl()
{
	_sigma_position = Matrix3d::Identity();
	_sigma_force.setZero();

	resetIntegratorsLinear();
#ifdef USING_OTG 
	_otg->reInitialize(_current_position, _current_orientation);
#endif
}

void PosOriTask::setMomentAxis(const Vector3d moment_axis)
{
	Vector3d normalized_axis = moment_axis.normalized();

	_sigma_moment = normalized_axis*normalized_axis.transpose();
	_sigma_orientation = Matrix3d::Identity() - _sigma_moment;

	resetIntegratorsAngular();
}

void PosOriTask::updateMomentAxis(const Vector3d moment_axis)
{
	Vector3d normalized_axis = moment_axis.normalized();

	_sigma_moment = normalized_axis*normalized_axis.transpose();
	_sigma_orientation = Matrix3d::Identity() - _sigma_moment;	
}

void PosOriTask::setAngularMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_orientation = normalized_axis*normalized_axis.transpose();
	_sigma_moment = Matrix3d::Identity() - _sigma_orientation;

	resetIntegratorsAngular();
}

void PosOriTask::updateAngularMotionAxis(const Vector3d motion_axis)
{
	Vector3d normalized_axis = motion_axis.normalized();

	_sigma_orientation = normalized_axis*normalized_axis.transpose();
	_sigma_moment = Matrix3d::Identity() - _sigma_orientation;
}

void PosOriTask::setFullMomentControl()
{
	_sigma_moment = Matrix3d::Identity();
	_sigma_orientation.setZero();

	resetIntegratorsAngular();
}

void PosOriTask::setFullAngularMotionControl()
{
	_sigma_orientation = Matrix3d::Identity();
	_sigma_moment.setZero();

	resetIntegratorsAngular();
}

void PosOriTask::setClosedLoopForceControl()
{
	_closed_loop_force_control = true;
	resetIntegratorsLinear();
}

void PosOriTask::setOpenLoopForceControl()
{
	_closed_loop_force_control = false;
}

void PosOriTask::setClosedLoopMomentControl()
{
	_closed_loop_moment_control = true;
	resetIntegratorsAngular();
}

void PosOriTask::setOpenLoopMomentControl()
{
	_closed_loop_moment_control = false;
}

void PosOriTask::enablePassivity()
{
	_passivity_enabled = true;
}

void PosOriTask::disablePassivity()
{
	_passivity_enabled = false;
}

void PosOriTask::resetIntegrators()
{
	_integrated_orientation_error.setZero();
	_integrated_position_error.setZero();
	_integrated_force_error.setZero();
	_integrated_moment_error.setZero();
	_first_iteration = true;	
}

void PosOriTask::resetIntegratorsLinear()
{
	_integrated_position_error.setZero();
	_integrated_force_error.setZero();
}

void PosOriTask::resetIntegratorsAngular()
{
	_integrated_orientation_error.setZero();
	_integrated_moment_error.setZero();
}

MatrixXd PosOriTask::lambdaInterpolation(const MatrixXd& lambda_0, const MatrixXd& lambda_1, const double& theta) 
{
	// Eigen::Matrix<double, 6, 6> lambda_0_log = lambda_0.log();
	// Eigen::Matrix<double, 6, 6> lambda_1_log = lambda_1.log();
	// Eigen::Matrix<double, 6, 6> lambda_linear_interp = lambda_0_log + theta * lambda_1_log;
	// return lambda_linear_interp.exp();
	// return (lambda_0.log() + theta * lambda_1.log()).exp();
	// return lambda_0.pow(1 - theta) * lambda_1.pow(theta);
	return lambda_0 * (1 - theta) + lambda_1 * theta;
}

MatrixXd PosOriTask::lambdaReduction(const MatrixXd& A, const double& tol) 
{
    int k = A.rows();
    if (A.rows() > A.cols()) {
        k = A.cols();
    }
    MatrixXd Sigma = MatrixXd::Zero(k, k);
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Sigma(0, 0) = svd.singularValues()(0);
    double ratio = 0;

    for (int i = 1; i < svd.singularValues().size(); ++i) {
        // ratio = svd.singularValues()(i) / svd.singularValues()(0);
		ratio = svd.singularValues()(i);
        if (ratio > tol) {
            Sigma(i, i) = svd.singularValues()(i);
        }
    }
    return svd.matrixU() * Sigma * svd.matrixV();
}

MatrixXd PosOriTask::invLambdaReduction(const MatrixXd& A, const double& tol) 
{
    int k = A.rows();
    if (A.rows() > A.cols()) {
        k = A.cols();
    }
    MatrixXd Sigma = MatrixXd::Zero(k, k);
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
	double ratio = 0;

    for (int i = svd.singularValues().size() - 1; i > 0; --i) {
        // ratio = svd.singularValues()(i) / svd.singularValues()(0);
		ratio = svd.singularValues()(i);
        if (ratio < tol) {
            Sigma(i, i) = svd.singularValues()(i);
        }
    }
    return svd.matrixU() * Sigma * svd.matrixV();
}

} /* namespace Sai2Primitives */
