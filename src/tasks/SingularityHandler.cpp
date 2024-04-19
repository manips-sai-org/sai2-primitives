/**
 * @file SingularityHandler.cpp
 * @author William Chong (wmchong@stnaford.edu)
 * @brief 
 * @version 0.1
 * @date 2024-01-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "SingularityHandler.h"

namespace Sai2Primitives {

SingularityHandler::SingularityHandler(std::shared_ptr<Sai2Model::Sai2Model> robot,
                                       const std::string& link_name,
                                       const Affine3d& compliant_frame,
                                       const int& task_rank,
                                       const double& s_abs_tol,
                                       const double& type_1_tol,
                                       const double& type_2_torque_ratio,
                                       const double& perturb_step_size,
                                       const int& buffer_size,
                                       const bool& verbose) : 
                                       _robot(robot),
                                       _link_name(link_name),
                                       _compliant_frame(compliant_frame),
                                       _task_rank(task_rank),
                                       _s_abs_tol(s_abs_tol), 
                                       _type_1_tol(type_1_tol), 
                                       _type_2_torque_ratio(type_2_torque_ratio), 
                                       _perturb_step_size(perturb_step_size),
                                       _verbose(verbose)
{
    // initialize robot specific variables 
    _q_upper = VectorXd::Zero(_robot->dof());
    _q_lower = VectorXd::Zero(_robot->dof());
    _tau_upper = VectorXd::Zero(_robot->dof());
    _tau_lower = VectorXd::Zero(_robot->dof());
    _joint_midrange = VectorXd::Zero(_robot->dof());
    _type_2_torque_vector = VectorXd::Zero(_robot->dof());
    auto joint_limits = _robot->jointLimits();
    for (int i = 0; i < joint_limits.size(); ++i) {
        _q_upper(i) = joint_limits[i].position_upper;
        _q_lower(i) = joint_limits[i].position_lower;
        _joint_midrange(i) = 0.5 * (joint_limits[i].position_lower + joint_limits[i].position_upper);
        _type_2_torque_vector(i) = _type_2_torque_ratio * joint_limits[i].effort;
        _tau_upper(i) = joint_limits[i].effort;
        _tau_lower(i) = - joint_limits[i].effort;
    }

    // initialize singularity handling variables 
    _singularity_types.resize(0);
    _q_prior = _joint_midrange;
    _dq_prior = VectorXd::Zero(_robot->dof());
    setGains(50, 14, 5);
    _type_1_counter = 0;
    _type_2_counter = 0;
    _singular_task_force = VectorXd::Zero(6);
    _singular_task_torques = VectorXd::Zero(_robot->dof());
    _joint_strategy_torques = VectorXd::Zero(_robot->dof());
    _type_2_direction = VectorXd::Ones(_robot->dof());
}

void SingularityHandler::updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec) {
    
    // task range decomposition
    int dof = _robot->dof();

    JacobiSVD<MatrixXd> J_svd(projected_jacobian, ComputeThinU | ComputeThinV);
    _svd_U = J_svd.matrixU();
    _svd_s = J_svd.singularValues();
    _svd_V = J_svd.matrixV();

    if (_svd_s(0) < _s_abs_tol) {
        // fully singular task
        _alpha = 0;

        // non-singular task
        _task_range_ns = MatrixXd::Zero(_task_rank, 1);

        // singular task 
        _task_range_s = _svd_U.leftCols(_task_rank);
        _joint_task_range_s = _svd_V.leftCols(_task_rank);
        _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;
        _Lambda_s = (_projected_jacobian_s *
					_robot->MInv() * 
					_projected_jacobian_s.transpose()).completeOrthogonalDecomposition().pseudoInverse();
    } else {
        for (int i = 1; i < _task_rank; ++i) {
            double condition_number = _svd_s(i) / _svd_s(0);

            if (condition_number < _s_max) {
                // task enters singularity blending region
                _alpha = std::clamp((condition_number - _s_min) / (_s_max - _s_min), 0., 1.);

                // non-singular task
                _task_range_ns = _svd_U.leftCols(i);
                _projected_jacobian_ns = _task_range_ns.transpose() * projected_jacobian;
                Sai2Model::OpSpaceMatrices ns_matrices =
                    _robot->operationalSpaceMatrices(_projected_jacobian_ns);
                _Lambda_ns = ns_matrices.Lambda;
                _Jbar_ns = ns_matrices.Jbar;
                _N_ns = ns_matrices.N;

                // singular task 
                // task range only collects columns of U up to size task_rank - non-singular task rank
                _task_range_s = _svd_U.block(0, i, _svd_U.rows(), _task_rank - i);  
                _joint_task_range_s = _svd_V.block(0, i, _svd_V.rows(), _task_rank - i);
                _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;  // in N_ns?
                Sai2Model::OpSpaceMatrices s_matrices =
                    _robot->operationalSpaceMatrices(_projected_jacobian_s);
                _Lambda_s = s_matrices.Lambda;
                break;

            } else if (i == _task_rank - 1) {
                // fully non-singular task  
                _alpha = 1;
                
                // non-singular task
                _task_range_ns = _svd_U.leftCols(_task_rank); 
                _projected_jacobian_ns = _task_range_ns.transpose() * projected_jacobian;
                Sai2Model::OpSpaceMatrices ns_matrices =
                    _robot->operationalSpaceMatrices(_projected_jacobian_ns);
                _Lambda_ns = ns_matrices.Lambda;
                _Jbar_ns = ns_matrices.Jbar;
                _N_ns = ns_matrices.N;

                // singular task 
                _task_range_s = MatrixXd::Zero(_task_rank, 1);
                _joint_task_range_s = MatrixXd::Zero(dof, 1);
            }
        }
    }

    // task update
    if (_task_range_s.norm() == 0) {
        _N = _N_ns;  // _N = N_ns (rank(N_ns) = task_rank)
    } else {
        // update posture task (occupies all singular joint directions, but joint torque handles one at a time)
        _posture_projected_jacobian = _joint_task_range_s.transpose() * _N_ns * N_prec;
        Sai2Model::OpSpaceMatrices op_space_matrices =
            _robot->operationalSpaceMatrices(_posture_projected_jacobian);
        _Lambda_joint_s = op_space_matrices.Lambda;
        _N = op_space_matrices.N * _N_ns;  // _N = N_partial_joint * N_Ns (rank(_N = N_partial_joint * N_ns) = task_rank)
    }

    // apply dynamic decoupling modifications
    switch (_dynamic_decoupling_type) {
		case FULL_DYNAMIC_DECOUPLING: {
			_Lambda_ns_modified = _Lambda_ns;
			_Lambda_s_modified = _Lambda_s;
            _Lambda_joint_s_modified = _Lambda_joint_s;
			break;
		}

		case IMPEDANCE: {
			_Lambda_ns_modified.setIdentity();
			_Lambda_s_modified.setIdentity();
            _Lambda_joint_s_modified.setIdentity();
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

			// nonsingular lambda
			MatrixXd Lambda_inv_BIE =
				_projected_jacobian_ns *
				M_inv_BIE * 
				_projected_jacobian_ns.transpose();
			_Lambda_ns_modified = Lambda_inv_BIE.inverse();

			// singular lambda
			if (_task_range_s.norm() != 0) {
				Lambda_inv_BIE =
					_projected_jacobian_s *
					M_inv_BIE * 
					_projected_jacobian_s.transpose();
                _Lambda_s_modified = Lambda_inv_BIE.inverse();
			} else {
				_Lambda_s_modified = _Lambda_s;
			}

            // joint strategy lambda 
            if (_task_range_s.norm() != 0) {
                Lambda_inv_BIE = _posture_projected_jacobian * 
                                    M_inv_BIE * 
                                    _posture_projected_jacobian.transpose();
                _Lambda_joint_s_modified = Lambda_inv_BIE.inverse();
            } 
			break;
		}

		default: {
			_Lambda_s_modified = _Lambda_s;
			_Lambda_ns_modified = _Lambda_ns;
            _Lambda_joint_s_modified = _Lambda_joint_s;
			break;
		}
	}

    // classify singularity
    classifySingularity(_task_range_s, _joint_task_range_s);

}

void SingularityHandler::classifySingularity(const MatrixXd& singular_task_range,
                                             const MatrixXd& singular_joint_task_range) {
    // memory of entering conditions 
    if (_singularity_types.size() == 0 || (_type_2_counter > _type_1_counter)) {
        _q_prior = _robot->q();
        _dq_prior = _robot->dq();
    } 

    // if singular task range is empty, return no singularities 
    if (singular_task_range.norm() == 0) {
        _singularity_types.resize(0);
        _singularity_history.clear();
        _type_1_counter = 0;
        _type_2_counter = 0;
        _singularity_task_range_history.clear();
        return;
    }

    // classify each column in the singular task range
    _singularity_types.resize(singular_task_range.cols());
    VectorXd curr_q = _robot->q();
    Vector3d curr_pos = _robot->position(_link_name, _compliant_frame.translation());
    Matrix3d curr_ori = _robot->rotation(_link_name, _compliant_frame.linear());

    for (int i = 0; i < singular_task_range.cols(); ++i) {
        VectorXd delta_q = _perturb_step_size * singular_joint_task_range.col(i);
        _robot->setQ(curr_q + delta_q);
        _robot->updateKinematics();

        // compute classification based on motion along singular direction from perturbation 
        Vector3d pos_delta = _robot->position(_link_name, _compliant_frame.translation()) - curr_pos;
        Vector3d ori_delta = Sai2Model::orientationError(_robot->rotation(_link_name, _compliant_frame.linear()), curr_ori);
        VectorXd delta_vector(6);
        delta_vector.head(3) = pos_delta;
        delta_vector.tail(3) = ori_delta;
        double motion_along_singular_direction = std::abs(delta_vector.dot(singular_task_range.col(i)));
        if (motion_along_singular_direction > _type_1_tol) {
            _singularity_types[i] = TYPE_1_SINGULARITY;
        } else {
            _singularity_types[i] = TYPE_2_SINGULARITY;
        }
            
        _robot->setQ(curr_q);
        _robot->updateKinematics();
    }

    // add to buffer and counters (preference for handling type 1 over type 2 for multiple singularities)
    auto it = std::find(_singularity_types.begin(), _singularity_types.end(), TYPE_1_SINGULARITY);
    if (it != _singularity_types.end()) {
        int index = std::distance(_singularity_types.begin(), it);
        _singularity_history.push_back(TYPE_1_SINGULARITY);
        _singularity_task_range_history.push_back(singular_task_range.col(index));
        _type_1_counter++;
    } else {
        _singularity_history.push_back(TYPE_2_SINGULARITY);
        _singularity_task_range_history.push_back(singular_task_range.col(0));
        _type_2_counter++;
    }

    // pop oldest if greater than buffer size
    if (_singularity_history.size() > _buffer_size) {
        if (_singularity_history.front() == TYPE_1_SINGULARITY) {
            _type_1_counter--;
        } else if (_singularity_history.front() == TYPE_2_SINGULARITY) {
            _type_2_counter--;
        }
        _singularity_history.pop_front();
        _singularity_task_range_history.pop_front();
    }

}

VectorXd SingularityHandler::computeTorques(const VectorXd& unit_mass_force, const VectorXd& force_related_terms)
{
    if (_verbose) {
        if (_singularity_types.size() != 0) {
            for (auto type : _singularity_types) {
                std::cout << "Singularity: " << singularity_labels[type] << " | ";
            }
            std::cout << "\n---\n";
        }
    }

    if (_singularity_types.size() == 0) {
        VectorXd tau = _projected_jacobian_ns.transpose() * (_Lambda_ns_modified * _task_range_ns.transpose() * unit_mass_force + \
                            _task_range_ns.transpose() * force_related_terms);
        // logging 
        _nonsingular_task_force = _Lambda_ns_modified * _task_range_ns.transpose() * unit_mass_force;
        _singular_task_force = VectorXd::Zero(1);
        _singular_task_torques = VectorXd::Zero(_robot->dof());
        _joint_strategy_torques = VectorXd::Zero(_robot->dof());
        _original_torques = tau;
        _blended_torques = tau;
        _singular_torques = VectorXd::Zero(7);
        _nonsingular_torques = tau;
        return tau;
    } else {
        VectorXd unit_torques = VectorXd::Zero(_robot->dof());
        VectorXd tau_ns = VectorXd::Zero(_robot->dof());

        // compute non-singular torques 
        if (_task_range_ns.norm() != 0) {
            tau_ns = _projected_jacobian_ns.transpose() * (_Lambda_ns_modified * _task_range_ns.transpose() * unit_mass_force + \
                        _task_range_ns.transpose() * force_related_terms);
            _nonsingular_task_force = _Lambda_ns_modified * _task_range_ns.transpose() * unit_mass_force;
        }

        // handle singularity type based on which one has more coutns 
        if (_type_1_counter > _type_2_counter) {
            // joint holding to entering joint conditions  
            unit_torques = - _kp * (_robot->q() - _q_prior) - _kv * _robot->dq();  
            _joint_strategy_torques = _posture_projected_jacobian.transpose() * _Lambda_joint_s_modified * _joint_task_range_s.transpose() * unit_torques;
        } else {
            // apply open-loop torque proportional to dot(unit mass force, singular direction)
            // zero torque achieved when singular direction is orthgonal to the desired unit mass force direction
            // the direction is chosen away from joint limit if joint is in limit 
            _fTd = ((unit_mass_force.normalized() + force_related_terms.normalized()).normalized()).dot(_task_range_s.col(0));
            VectorXd magnitude_unit_torques = std::abs(_fTd) * _type_2_torque_vector;
            for (int i = 0; i < _joint_task_range_s.rows(); ++i) {
                if (_joint_task_range_s(i, 0) != 0) {
                    if (std::abs(_robot->q()(i) - _q_upper(i)) < 5 * M_PI / 180) {
                        _type_2_direction(i) = -1;
                    } else if (std::abs(_robot->q()(i) - _q_lower(i)) < 5 * M_PI / 180) {
                        _type_2_direction(i) = 1;
                    } 
                }
            }
            unit_torques = _type_2_direction.array() * magnitude_unit_torques.array(); 
            _joint_strategy_torques = _posture_projected_jacobian.transpose() * _joint_task_range_s.transpose() * unit_torques + \
                                        _posture_projected_jacobian.transpose() * _Lambda_joint_s_modified * _joint_task_range_s.transpose() * (- _kv_type_2 * _robot->dq());
        }
        
        // combine non-singular torques and blended singular torques with joint strategy torques (add joint space damping in singular task force)
        _singular_task_force = _Lambda_s_modified * _task_range_s.transpose() * unit_mass_force + \
                                            _task_range_s.transpose() * force_related_terms;
        _singular_task_torques = _projected_jacobian_s.transpose() * _singular_task_force;

        if (!_handling_flag) {
            return tau_ns;
        }

        for (int i = 0; i < _robot->dof(); ++i) {
            if (isnan(_singular_task_torques(i))) {
                _singular_task_torques(i) = _singular_task_torques(i) > 0 ? _tau_upper(i) : _tau_lower(i);
            } else if (_singular_task_torques(i) > _tau_upper(i)) {
                _singular_task_torques(i) = _tau_upper(i);
            } else if (_singular_task_torques(i) < _tau_lower(i)) {
                _singular_task_torques(i) = _tau_lower(i);
            }
        }
        _tau_s = pow(_alpha, 1) * _singular_task_torques + (1 - pow(_alpha, 1)) * _joint_strategy_torques;
        _original_torques = tau_ns + _singular_task_torques;
        _blended_torques = tau_ns + _tau_s;
        _nonsingular_torques = tau_ns;
        _singular_torques = _tau_s;

        return tau_ns + _tau_s;
    }
}

}  // namespace