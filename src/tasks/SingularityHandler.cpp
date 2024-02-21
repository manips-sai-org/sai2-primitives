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
                                       const int& task_rank,
                                       const MatrixXd& J_posture,
                                       const double& s_abs_tol,
                                       const double& type_1_tol,
                                       const double& type_2_torque_ratio,
                                       const int& queue_size,
                                       const bool& verbose) : _robot(robot), _task_rank(task_rank),
                                       _J_posture(J_posture), _s_abs_tol(s_abs_tol), _type_1_tol(type_1_tol), 
                                       _type_2_torque_ratio(type_2_torque_ratio), _queue_size(queue_size),
                                       _verbose(verbose)
{
    // initialize robot specific variables 
    _joint_midrange = VectorXd::Zero(_robot->dof());
    _type_2_torque_vector = VectorXd::Zero(_robot->dof());
    auto joint_limits = _robot->jointLimits();
    for (int i = 0; i < joint_limits.size(); ++i) {
        _type_2_torque_vector(i) = _type_2_torque_ratio * joint_limits[i].effort;
        _joint_midrange(i) = 0.5 * (joint_limits[i].position_lower + joint_limits[i].position_upper);
    }

    // initialize singularity variables 
    _sing_history = std::queue<SingularityType>();
    _sing_direction_queue = std::queue<Eigen::Matrix<double, 6, 1>>();
    _q_prior = robot->q();
    _dq_prior = robot->dq();
    if (task_rank == 6) {
        double kp = 20;
        setGains(kp, 2 * std::sqrt(kp));
    } else {
        double kp = 100;
        setGains(kp, 2 * std::sqrt(kp));
    }
    setSingularity(NO_SINGULARITY);
    setDynamicDecouplingType(BOUNDED_INERTIA_ESTIMATES);

    // initialize singularity history
    // the entire buffer history must be classified as type 2 to classify type 2 singularity, 
    // otherwise, it's type 1 for stability since type 2 singularity commands open-loop torque 
    _sing_history = std::queue<SingularityType>();
    for (int i = 0; i < _queue_size; ++i) {
        _sing_history.push(NO_SINGULARITY);
    }
}

void SingularityHandler::updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec) {
    
    // task range decomposition
    int dof = _robot->dof();
    Sai2Model::SvdData J_svd = Sai2Model::matrixSvd(projected_jacobian);
    _s_values = J_svd.s;
    
    if (J_svd.s(0) < _s_abs_tol) {
        // fully singular task
        _alpha = 0;

        // non-singular task
        _task_range_ns = MatrixXd::Zero(6, _task_rank);
        _projected_jacobian_ns = MatrixXd::Zero(_task_rank, dof);
        _Lambda_ns = MatrixXd::Zero(_task_rank, _task_rank);
        _Jbar_ns = MatrixXd::Zero(dof, _task_rank);
        _N_ns = MatrixXd::Zero(dof, dof);

        // singular task 
        _task_range_s = J_svd.U.leftCols(_task_rank);
        _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;
        Sai2Model::OpSpaceMatrices s_matrices =
            _robot->operationalSpaceMatrices(_projected_jacobian_s);
        _Lambda_s = s_matrices.Lambda;
        _Jbar_s = s_matrices.Jbar;
        _N_s = s_matrices.N;
    } else {
        for (int i = 1; i < _task_rank; ++i) {
            double condition_number = J_svd.s(i) / J_svd.s(0);

            if (condition_number < _s_max) {
                // task enters singularity blending region
                _alpha = std::clamp((condition_number - _s_min) / (_s_max - _s_min), 0., 1.);

                // non-singular task
                _task_range_ns = J_svd.U.leftCols(i);
                _projected_jacobian_ns = _task_range_ns.transpose() * projected_jacobian;
                Sai2Model::OpSpaceMatrices ns_matrices =
                    _robot->operationalSpaceMatrices(_projected_jacobian_ns);
                _Lambda_ns = ns_matrices.Lambda;
                _Jbar_ns = ns_matrices.Jbar;
                _N_ns = ns_matrices.N;

                // singular task 
                // task range only collects columns of U up to size _task_rank - non-singular task rank
                _task_range_s = J_svd.U.block(0, i, J_svd.U.rows(), _task_rank - i);
                _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;
                Sai2Model::OpSpaceMatrices s_matrices =
                    _robot->operationalSpaceMatrices(_projected_jacobian_s);
                _Lambda_s = s_matrices.Lambda;
                _Jbar_s = s_matrices.Jbar;
                _N_s = s_matrices.N;
                break;

            } else if (i == _task_rank - 1) {
                // fully non-singular task 
                _alpha = 1;
                
                // non-singular task
                _task_range_ns = J_svd.U.leftCols(_task_rank); 
                _projected_jacobian_ns = _task_range_ns.transpose() * projected_jacobian;
                Sai2Model::OpSpaceMatrices ns_matrices =
                    _robot->operationalSpaceMatrices(_projected_jacobian_ns);
                _Lambda_ns = ns_matrices.Lambda;
                _Jbar_ns = ns_matrices.Jbar;
                _N_ns = ns_matrices.N;

                // singular task 
                _task_range_s = MatrixXd::Zero(6, _task_rank);
                _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;
                _Lambda_s = MatrixXd::Zero(_task_rank, _task_rank);
                _Jbar_s = MatrixXd::Zero(dof, _task_rank);
                _N_s = MatrixXd::Zero(dof, dof);
            }
        }
    }

    // apply dynamic decoupling modifications
    switch (_dynamic_decoupling_type) {
		case FULL_DYNAMIC_DECOUPLING: {
			_Lambda_ns_modified = _Lambda_ns;
			_Lambda_s_modified = _Lambda_s;
			break;
		}

		case IMPEDANCE: {
			_Lambda_ns_modified.setIdentity();
			_Lambda_s_modified.setIdentity();
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
				_Lambda_s_modified = Lambda_inv_BIE.completeOrthogonalDecomposition().pseudoInverse();
			} else {
				_Lambda_s_modified = _Lambda_s;
			}
			break;
		}

		default: {
			_Lambda_s_modified = _Lambda_s;
			_Lambda_ns_modified = _Lambda_ns;
			break;
		}
	}

    // classify singularity
    classifySingularity(_task_range_s);

    // task update
    if (_sing_type == NO_SINGULARITY) {
        _N = _N_ns;  // _N = N_ns (rank(N_ns) = task_rank)
    } else {
        // update posture task 
        _posture_projected_jacobian = _J_posture * _N_ns * N_prec;
        _current_task_range = Sai2Model::matrixRangeBasis(_posture_projected_jacobian); 
        Sai2Model::OpSpaceMatrices op_space_matrices =
            _robot->operationalSpaceMatrices(_current_task_range.transpose() * _posture_projected_jacobian);
        _M_partial = op_space_matrices.Lambda;
        _N = op_space_matrices.N * _N_ns;  // _N = N_partial_joint * N_ns is rank of task_rank
    }
}

void SingularityHandler::classifySingularity(const MatrixXd& singular_task_range) {
    // memory of entering conditions 
    if (_sing_type == NO_SINGULARITY) {
        _q_prior = _robot->q();
        _dq_prior = _robot->dq();
    }

    // if singular task range is empty 
    if (singular_task_range.norm() == 0) {
        _sing_type = NO_SINGULARITY;
        // clear singularity direction queue (robot exited singularity)
        while (!_sing_direction_queue.empty()) {
            _sing_direction_queue.pop();
        }
        return;
    }

    // fill buffer with the most singular task range (already disregarded zero columns)
    _sing_direction_queue.push(singular_task_range.rightCols(1));
    VectorXd last_direction = _sing_direction_queue.back();
    VectorXd current_direction = _sing_direction_queue.front();

    // check singular direction alignment for type 1 singularity
    // type 1 singularity is preferred due to holding behavior
    // type 1 is classified if the first and last singular directions in queue are aligned within tolerance
    // type 2 is classified only if the entire past queue_size timesteps is type 2 
    if (std::abs(current_direction.dot(last_direction)) > _type_1_tol) {
        _sing_history.push(TYPE_1_SINGULARITY);
        _sing_type = TYPE_1_SINGULARITY;
    } else {
        _sing_history.push(TYPE_2_SINGULARITY);
        if (allElementsSame(_sing_history)) {
            _sing_type = TYPE_2_SINGULARITY;
        } else {
            _sing_type = TYPE_1_SINGULARITY;
        }
    }
}

VectorXd SingularityHandler::computeTorques(const VectorXd& unit_mass_force, const VectorXd& force_related_terms)
{
    VectorXd joint_strategy_torques = VectorXd::Zero(_robot->dof());
    VectorXd unit_torques = VectorXd::Zero(_robot->dof());

    if (_verbose) {
        if (_sing_type != NO_SINGULARITY) {
            std::cout << "Singularity: " << singularity_labels[_sing_type] << "\n---\n";
        }
    }

    if (_sing_type == TYPE_1_SINGULARITY) {
        // joint holding to entering joint position 
        unit_torques = - _kp * (_robot->q() - _q_prior) - _kv * _robot->dq();  
        joint_strategy_torques = (_current_task_range.transpose() * _posture_projected_jacobian).transpose() * \
                                            _M_partial * _current_task_range.transpose() * unit_torques;
    } else if (_sing_type == TYPE_2_SINGULARITY) {
        // apply open-loop torque proportional to dot(unit mass force, singular direction)
        // zero torque achieved when singular direction is orthgonal to the desired unit mass force direction
        // the direction is chosen to approach to the joint mid-range 
        _M_partial.setIdentity();  
        double fTd = ((unit_mass_force.normalized()).transpose() * _task_range_s.rightCols(1))(0);
        VectorXd midrange_distance = _joint_midrange - _robot->q();
        VectorXd torque_sign = (midrange_distance.array() > 0).cast<double>() - (midrange_distance.array() < 0).cast<double>();
        VectorXd magnitude_unit_torques = std::abs(fTd) * _type_2_torque_vector;
        unit_torques = torque_sign.array() * magnitude_unit_torques.array();
        joint_strategy_torques = (_current_task_range.transpose() * _posture_projected_jacobian).transpose() * \
                                            _M_partial * _current_task_range.transpose() * unit_torques;
    }

    // Combine non-singular torques and blended singular torques with joint strategy torques 
    VectorXd tau_ns = _projected_jacobian_ns.transpose() * (_Lambda_ns_modified * _task_range_ns.transpose() * unit_mass_force + \
                            _task_range_ns.transpose() * force_related_terms);
    if (_sing_type == NO_SINGULARITY) {
        return tau_ns;
    } else {
        VectorXd singular_task_force = _Lambda_s_modified * _task_range_s.transpose() * unit_mass_force + \
                                            _task_range_s.transpose() * force_related_terms;
        VectorXd tau_s = _alpha * _projected_jacobian_s.transpose() * singular_task_force + \
                            (1 - _alpha) * joint_strategy_torques;
        _tau_s = tau_s;
        return tau_ns + tau_s;
    }
                
}

}  // namespace