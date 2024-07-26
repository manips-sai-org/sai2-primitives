/*
 * SingularityHandler.cpp
 *
 *      Author: William Chong 
 */

#include "SingularityHandler.h"

// Default parameters 
namespace {
    double S_ABS_TOL = 1e-3;
    double TYPE_1_TOL = 0.5;
    double TYPE_2_TORQUE_RATIO = 1e-2;
    double TYPE_2_ANGLE_THRESHOLD = 5 * M_PI / 180;
    double PERTURB_STEP_SIZE = 5;
    double BUFFER_SIZE = 200;
    double KP_TYPE_1 = 50;
    double KV_TYPE_1 = 14;
    double KV_TYPE_2 = 5;
}

namespace Sai2Primitives {

SingularityHandler::SingularityHandler(std::shared_ptr<Sai2Model::Sai2Model> robot,
                                       const std::string& link_name,
                                       const Affine3d& compliant_frame,
                                       const int& task_rank,
                                       const bool& verbose) : 
                                       _robot(robot),
                                       _link_name(link_name),
                                       _compliant_frame(compliant_frame),
                                       _task_rank(task_rank),
                                       _verbose(verbose)
{
    // initialize limits 
    _dof = _robot->dof();
    _q_upper = VectorXd::Zero(_dof);
    _q_lower = VectorXd::Zero(_dof);
    _tau_upper = VectorXd::Zero(_dof);
    _tau_lower = VectorXd::Zero(_dof);
    _joint_midrange = VectorXd::Zero(_dof);
    _type_2_torque_vector = VectorXd::Zero(_dof);
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
    _dq_prior = VectorXd::Zero(_dof);
    setSingularityHandlingGains(KP_TYPE_1, KV_TYPE_1, KV_TYPE_2);
    setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);
	setBoundedInertiaEstimateThreshold(0.1);
    _type_1_counter = 0;
    _type_2_counter = 0;
    _type_2_direction = VectorXd::Ones(_dof);
    _enforce_type_1_strategy = false;
    _enforce_handling_strategy = true;

    // initialize singularity handling classification variables
    _s_abs_tol = S_ABS_TOL;
    _type_1_tol = TYPE_1_TOL; 
    _type_2_torque_ratio = TYPE_2_TORQUE_RATIO;
    _type_2_angle_threshold = TYPE_2_ANGLE_THRESHOLD;
    _perturb_step_size = PERTURB_STEP_SIZE;
    _buffer_size = BUFFER_SIZE;
}

void SingularityHandler::updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec) {
    
    // task range decomposition
    JacobiSVD<MatrixXd> J_svd(projected_jacobian, ComputeThinU | ComputeThinV);
    _svd_U = J_svd.matrixU();
    _svd_s = J_svd.singularValues();
    _svd_V = J_svd.matrixV();

    if (_svd_s(0) < _s_abs_tol) {
        // fully singular task
        _alpha = 0;

        // placeholder non-singular terms 
        _task_range_ns = MatrixXd::Zero(_task_rank, 1);
        _projected_jacobian_ns = MatrixXd::Zero(_task_rank, _dof);
        _Lambda_ns = MatrixXd::Zero(_task_rank, _task_rank);

        // singular task 
        _task_range_s = _svd_U.leftCols(_task_rank);
        _joint_task_range_s = _svd_V.leftCols(_task_rank);
        _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;
        _Lambda_s = (_projected_jacobian_s *
					_robot->MInv() * 
					_projected_jacobian_s.transpose()).completeOrthogonalDecomposition().pseudoInverse();
    } else {
        for (int i = 1; i < _task_rank; ++i) {
            double inv_condition_number = _svd_s(i) / _svd_s(0);

            if (inv_condition_number < _s_max) {
                // task enters singularity blending region
                _alpha = std::clamp((inv_condition_number - _s_min) / (_s_max - _s_min), 0., 1.);

                // non-singular task
                _task_range_ns = _svd_U.leftCols(i);
                _projected_jacobian_ns = _task_range_ns.transpose() * projected_jacobian;
                Sai2Model::OpSpaceMatrices ns_matrices =
                    _robot->operationalSpaceMatrices(_projected_jacobian_ns);
                _Lambda_ns = ns_matrices.Lambda;
                _Jbar_ns = ns_matrices.Jbar;
                _N_ns = ns_matrices.N;

                // singular task: task range only collects columns of U up to size task_rank - non-singular task rank
                _task_range_s = _svd_U.block(0, i, _svd_U.rows(), _task_rank - i);  
                _joint_task_range_s = _svd_V.block(0, i, _svd_V.rows(), _task_rank - i);
                _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;  
                _Lambda_s = (_projected_jacobian_s * _robot->MInv() * _projected_jacobian_s.transpose()).inverse();
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

                // placeholder singular task terms 
                _task_range_s = MatrixXd::Zero(_task_rank, 1);
                _joint_task_range_s = MatrixXd::Zero(_dof, 1);
                _projected_jacobian_s = MatrixXd::Zero(_task_rank, _dof);
                _Lambda_s = MatrixXd::Zero(_task_rank, _task_rank);
            }
        }
    }

    // model updates 
    if (_task_range_s.norm() == 0 || !_enforce_handling_strategy) {
        _N = _N_ns;  
        _Lambda_joint_s = MatrixXd::Zero(1, 1);  // placeholder
    } else if (_task_range_ns.norm() == 0) {
        _N = N_prec;  // if task is fully singular, then pass through the task 
        _Lambda_joint_s = MatrixXd::Zero(1, 1);  // placeholder
    } else {
        _posture_projected_jacobian = _joint_task_range_s.transpose() * _N_ns * N_prec;
        Sai2Model::OpSpaceMatrices op_space_matrices =
            _robot->operationalSpaceMatrices(_posture_projected_jacobian);
        _Lambda_joint_s = op_space_matrices.Lambda;
        _N = op_space_matrices.N * _N_ns; 
    }

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
                if (M_BIE(i, i) < _bie_threshold) {
                    M_BIE(i, i) = _bie_threshold;
                }
            }
            MatrixXd M_inv_BIE = M_BIE.inverse();

            // non-singular lambda
            if (_task_range_ns.norm() != 0) {
                MatrixXd Lambda_inv_BIE =
                    _projected_jacobian_ns *
                    M_inv_BIE * 
                    _projected_jacobian_ns.transpose();
                _Lambda_ns_modified = Lambda_inv_BIE.inverse();
            } else {
                _Lambda_ns_modified = _Lambda_ns;
            }

            // singular lambda
            if (_task_range_s.norm() != 0) {
                MatrixXd Lambda_inv_BIE =
                    _projected_jacobian_s *
                    M_inv_BIE * 
                    _projected_jacobian_s.transpose();
                _Lambda_s_modified = Lambda_inv_BIE.inverse();
            } else {
                _Lambda_s_modified = _Lambda_s;
            }

            // joint strategy lambda 
            if (_task_range_s.norm() != 0) {
                MatrixXd Lambda_inv_BIE = 
                    _posture_projected_jacobian * 
                    M_inv_BIE * 
                    _posture_projected_jacobian.transpose();
                _Lambda_joint_s_modified = Lambda_inv_BIE.inverse();
            } else {
                _Lambda_joint_s_modified = _Lambda_joint_s;
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

    // add to buffer and counters (preference for handling type 1 over type 2 for multiple, simultaneous singularities)
    auto it = std::find(_singularity_types.begin(), _singularity_types.end(), TYPE_1_SINGULARITY);
    if (it != _singularity_types.end()) {
        _singularity_history.push_back(TYPE_1_SINGULARITY);
        _type_1_counter++;
    } else {
        _singularity_history.push_back(TYPE_2_SINGULARITY);
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
    }

}

VectorXd SingularityHandler::computeTorques(const VectorXd& unit_mass_force, const VectorXd& force_related_terms) {
    if (_verbose) {
        if (_singularity_types.size() != 0) {
            for (auto type : _singularity_types) {
                std::cout << "Singularity: " << singularity_labels[type] << " | ";
            }
            std::cout << "\n---\n";
        }
    }

    if (_singularity_types.size() == 0) {
        return _projected_jacobian_ns.transpose() * (_Lambda_ns_modified * _task_range_ns.transpose() * unit_mass_force + \
                    _task_range_ns.transpose() * force_related_terms);
    } else {
        VectorXd tau_ns = VectorXd::Zero(_dof);

        // compute non-singular torques 
        if (_task_range_ns.norm() == 0) {
            return tau_ns;  // pass through task if fully singular 
        } else {
            tau_ns = _projected_jacobian_ns.transpose() * (_Lambda_ns_modified * _task_range_ns.transpose() * unit_mass_force + \
                        _task_range_ns.transpose() * force_related_terms);
            if (!_enforce_handling_strategy) {
                return tau_ns;
            }
        } 

        // handle singularity type based on which one has more counts  
        if (_type_1_counter > _type_2_counter || _enforce_type_1_strategy) {
            // joint holding to entering joint conditions  
            VectorXd unit_torques = - _kp_type_1 * (_robot->q() - _q_prior) - _kv_type_1 * _robot->dq();  
            _joint_strategy_torques = _posture_projected_jacobian.transpose() * _Lambda_joint_s_modified * _joint_task_range_s.transpose() * unit_torques;
        } else {
            // apply open-loop torque proportional to dot(unit mass force, singular direction)
            // zero torque achieved when singular direction is orthogonal to the desired unit mass force direction
            // the direction is reversed if the joint is approaching a joint limit 
            for (int i = 0; i < _joint_task_range_s.rows(); ++i) {
                if (_joint_task_range_s(i, 0) != 0) {
                    if (std::abs(_robot->q()(i) - _q_upper(i)) < _type_2_angle_threshold) {
                        _type_2_direction(i) = - 1;
                    } else if (std::abs(_robot->q()(i) - _q_lower(i)) < _type_2_angle_threshold) {
                        _type_2_direction(i) = 1;
                    } 
                }
            }
            double fTd = ((unit_mass_force + force_related_terms).normalized()).dot(_task_range_s.col(0));
            VectorXd magnitude_unit_torques = std::abs(fTd) * _type_2_torque_vector;
            VectorXd unit_torques = _type_2_direction.array() * magnitude_unit_torques.array(); 
            _joint_strategy_torques = _posture_projected_jacobian.transpose() * _joint_task_range_s.transpose() * unit_torques + \
                                        _posture_projected_jacobian.transpose() * _Lambda_joint_s_modified * \
                                        _joint_task_range_s.transpose() * (- _kv_type_2 * _robot->dq());
        }

        // combine non-singular torques and blended singular torques with joint strategy torques
        _singular_task_torques = _projected_jacobian_s.transpose() * (_Lambda_s_modified * _task_range_s.transpose() * unit_mass_force + \
                                            _task_range_s.transpose() * force_related_terms);

        for (int i = 0; i < _dof; ++i) {
            if (isnan(_singular_task_torques(i))) {
                _singular_task_torques(i) = 0;  
            } else if (_singular_task_torques(i) > _tau_upper(i)) {
                _singular_task_torques(i) = _tau_upper(i);
            } else if (_singular_task_torques(i) < _tau_lower(i)) {
                _singular_task_torques(i) = _tau_lower(i);
            }
        }
        return tau_ns + _alpha * _singular_task_torques + (1 - _alpha) * _joint_strategy_torques;
    }
}

}  // namespace