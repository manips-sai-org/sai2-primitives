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
                                       const MatrixXd& J_posture,
                                       const double type_1_tol,
                                       const int buffer_size) : _robot(robot), 
                                       _J_posture(J_posture), _type_1_tol(type_1_tol)
{
    // set type 2 torque value
    _type_2_torque_vector = VectorXd::Zero(_robot->dof());
    setTorqueRatio(0.02);
    auto joint_limits = _robot->jointLimits();
    for (int i = 0; i < joint_limits.size(); ++i) {
        _type_2_torque_vector(i) = _type_2_torque_ratio * joint_limits[i].effort;
    }

    // initialize singularity variables 
    _sing_direction_buffer = std::make_pair(std::queue<Vector6d>(), std::queue<Vector6d>());
    _q_prior = std::make_pair(robot->q(), robot->q());
    _dq_prior = std::make_pair(robot->dq(), robot->dq());
    double kp = 10;
    double kv = 2 * std::sqrt(kp);
    setGains(kp, kv);
    setSingularity(NO_SINGULARITY, NO_SINGULARITY);
}

/*
    Performs the steps:
    - Compute the eigen-decomposition of lambda_inv
    - Compute Lambda_ns and Lambda_s with eigenvalue ratio cut-off
    - Get the singular and non-singular task ranges 
    - Get projected jacobians in the singular and non-singular task ranges (NOT the same: JL = sqrt(lambda(Lambda)), M^{-1} = LL^{T})
    - Form nullspace from non-singular task range
    - Compute blended force and joint strategy:
        - Type 1 singularity takes priority over type 2 singularity (type 1 breaks, and often not simultaneous)
        - Consequently, only 1 alpha value is used 
*/
SingularityOpSpaceMatrices SingularityHandler::updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec) {
    // compute Lambda inv
    MatrixXd Lambda_inv = projected_jacobian * _robot->MInv() * projected_jacobian.transpose();

    // compute eigen-decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(Lambda_inv);

    // compute singular and non-singular task ranges based on ratios
    const VectorXd& e = eigensolver.eigenvalues();  
	const MatrixXd& U = eigensolver.eigenvectors();
    // debug
    _e_values = e;
	int n_cols = 0;
	for (int i = 5; i >= 0; --i) {
		double e_ratio = e(i);
		// double e_ratio = e(i) / e.maxCoeff();
        _alpha = std::clamp((e_ratio - _e_min) / (_e_max - _e_min), 0., 1.);
		if (e_ratio < _e_max) {
			n_cols = i + 1;
			break;
		}
	}
	if (n_cols != 0) {
		MatrixXd U_s = U.leftCols(n_cols);
		MatrixXd U_ns = U.rightCols(6 - n_cols);
		VectorXd D_s(n_cols);
		VectorXd D_ns(6 - n_cols);
		for (int i = 0; i < n_cols; ++i) {
			D_s(i) = 1 / e(i);
		}
		for (int i = 0; i < 6 - n_cols; ++i) {
			D_ns(i) = 1 / e(i + n_cols);
		}

        // compute op-space values for singular and non-singular tasks 
        _task_range_ns = U_ns;
        _task_range_s = U_s;
        _projected_jacobian_ns = _task_range_ns.transpose() * projected_jacobian;
        _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;

        Sai2Model::OpSpaceMatrices ns_matrices = 
            _robot->operationalSpaceMatrices(_projected_jacobian_ns);
        _Lambda_ns = ns_matrices.Lambda;
        _Jbar_ns = ns_matrices.Jbar;
        _N_ns = ns_matrices.N;

        Sai2Model::OpSpaceMatrices s_matrices = 
            _robot->operationalSpaceMatrices(_projected_jacobian_s);
        _Lambda_s = s_matrices.Lambda;
        _Jbar_s = s_matrices.Jbar;
        _N_s = s_matrices.N;

	} else {
		MatrixXd U_s = MatrixXd::Zero(6, 6);
		MatrixXd U_ns = U;
        _projected_jacobian_ns = projected_jacobian;
		Sai2Model::OpSpaceMatrices op_space_matrices =
			_robot->operationalSpaceMatrices(
				projected_jacobian);
		_Lambda_s = MatrixXd::Zero(6, 6);
		_Lambda_ns = op_space_matrices.Lambda;
        // _task_range_s = U_s;
        // _task_range_ns = U_ns;
        _task_range_ns = MatrixXd::Identity(6, 6);
        _task_range_s = MatrixXd::Zero(6, 6);
		_Jbar_ns = op_space_matrices.Jbar;
		_N = op_space_matrices.N;
	}

    // classify singularities 
    classifySingularity(_task_range_s);

    // if task_range_s is empty, don't compute singularity-related terms 
    if (_task_range_s.norm() != 0) {
        // update posture task 
        _posture_projected_jacobian = _J_posture * _N_ns * N_prec;
        _current_task_range = Sai2Model::matrixRangeBasis(_posture_projected_jacobian); 
        Sai2Model::OpSpaceMatrices op_space_matrices =
            _robot->operationalSpaceMatrices(_current_task_range.transpose() * _posture_projected_jacobian);
        _M_partial = op_space_matrices.Lambda;
        _N = op_space_matrices.N * _N_ns;  // _N = N_partial_joint * N_ns 
    }

    return SingularityOpSpaceMatrices{_projected_jacobian_ns,
                                      _Lambda_ns,
                                      _N_ns,
                                      _task_range_ns,
                                      _projected_jacobian_s,
                                      _Lambda_s,
                                      _N_s,
                                      _task_range_s};
}

// SingularityOpSpaceMatrices SingularityHandler::updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec) {
//     // linear jacobian task range 
// 	Sai2Model::SvdData linear_svd = Sai2Model::matrixSvd(projected_jacobian.topRows(3));

//     _linear_singular_values = linear_svd.s;

//     // debug
//     // std::cout << "linear svd: " << linear_svd.s.transpose() << "\n";
//     // std::cout << "linear svd ratios: " << linear_svd.s.transpose() / linear_svd.s(0) << "\n";

// 	double sigma_0 = linear_svd.s(0);
// 	if (sigma_0 < _linear_sing_tol_max) {
// 		_linear_task_range_ns = MatrixXd::Zero(3, 3);
// 		// _linear_task_range_s = MatrixXd::Identity(3, 3);
// 		_linear_task_range_s = linear_svd.U;
//         _alpha_linear = 0;
// 	} else {
// 		for (int i = 1; i < 3; ++i) {
// 			double sigma_ratio = linear_svd.s(i) / sigma_0;
//             // double sigma_ratio = linear_svd.s(i);
// 			if (sigma_ratio < _linear_sing_tol_max) {
// 				_linear_task_range_ns = linear_svd.U.leftCols(i);
// 				_linear_task_range_s = linear_svd.U.rightCols(3 - i);
// 				_alpha_linear = std::clamp((sigma_ratio - _linear_sing_tol_min) / (_linear_sing_tol_max - _linear_sing_tol_min), 0., 1.);
// 				// _alpha_linear *= _alpha_linear;
//                 break;
// 			}
// 			if (i == 2) {
// 				// _linear_task_range_ns = MatrixXd::Identity(3, 3);
// 				_linear_task_range_ns = linear_svd.U;
// 				_linear_task_range_s = MatrixXd::Zero(3, 3);
// 				_alpha_linear = 1;
// 			}
// 		}
// 	}

// 	// angular jacobian task range
// 	Sai2Model::SvdData angular_svd = Sai2Model::matrixSvd(projected_jacobian.bottomRows(3));

//     // debug
//     // std::cout << "angular svd: " << angular_svd.s.transpose() / angular_svd.s(0) << "\n";

// 	sigma_0 = angular_svd.s(0);
// 	if (sigma_0 < _angular_sing_tol_max) {
// 		_angular_task_range_ns = MatrixXd::Zero(3, 3);
// 		// _angular_task_range_s = MatrixXd::Identity(3, 3);
// 		_angular_task_range_s = angular_svd.U;
//         _alpha_angular = 0;
// 	} else {
// 		for (int i = 1; i < 3; ++i) {
// 			double sigma_ratio = angular_svd.s(i) / sigma_0;
// 			// double sigma_ratio = angular_svd.s(i);
// 			if (sigma_ratio < _angular_sing_tol_max) {
// 				_angular_task_range_ns = angular_svd.U.leftCols(i);
// 				_angular_task_range_s = angular_svd.U.rightCols(3 - i);
// 				_alpha_angular = std::clamp((sigma_ratio - _angular_sing_tol_min) / (_angular_sing_tol_max - _angular_sing_tol_min), 0., 1.);
// 				// _alpha_angular *= _alpha_angular;
//                 break;
// 			}
// 			if (i == 2) {
// 				// _angular_task_range_ns = MatrixXd::Identity(3, 3);
// 				_angular_task_range_ns = angular_svd.U;                
// 				_angular_task_range_s = MatrixXd::Zero(3, 3); 
// 				_alpha_angular = 1;
// 			}
// 		}
// 	}

//     // classify linear and angular singularities 
//     classifyLinearSingularity(_linear_task_range_s);
//     classifyAngularSingularity(_angular_task_range_s);

//     // stack task ranges 
//     _task_range_ns = getBlockMatrix(_linear_task_range_ns, _angular_task_range_ns);
//     _task_range_s = getBlockMatrix(_linear_task_range_s, _angular_task_range_s);

//     // obtain non-singular related terms 
//     _projected_jacobian_ns = _task_range_ns.transpose() * projected_jacobian;
//     Sai2Model::OpSpaceMatrices ns_matrices = _robot->operationalSpaceMatrices(_projected_jacobian_ns);
//     _Lambda_ns = ns_matrices.Lambda;
//     _N_ns = ns_matrices.N;
//     _N = _N_ns;

//     // if task_range_s is empty, don't compute singularity-related terms 
//     if (_task_range_s.norm() != 0) {
//         // get op-space matrices for entire singular task range 
//         _projected_jacobian_s = _task_range_s.transpose() * projected_jacobian;
//         Sai2Model::OpSpaceMatrices op_space_matrices = 
//             _robot->operationalSpaceMatrices(_projected_jacobian_s);
//         _Lambda_s = op_space_matrices.Lambda;
//         _N_s = op_space_matrices.N;

//         // get Jbar for linear singular task 
//         if (_linear_task_range_s.norm() != 0) {
//             _projected_jacobian_s_linear = _linear_task_range_s.transpose() * projected_jacobian.topRows(3);
//             Sai2Model::OpSpaceMatrices op_space_matrices = 
//                 _robot->operationalSpaceMatrices(_projected_jacobian_s_linear);
//             _Jbar_s_linear = op_space_matrices.Jbar;
//         }

//         // get Jbar for angular singular task 
//         if (_angular_task_range_ns.norm() != 0) {
//             _projected_jacobian_s_angular = _angular_task_range_s.transpose() * projected_jacobian.bottomRows(3);
//             Sai2Model::OpSpaceMatrices op_space_matrices = 
//                 _robot->operationalSpaceMatrices(_projected_jacobian_s_angular);
//             _Jbar_s_angular = op_space_matrices.Jbar; 
//         }

//         // update posture task 
//         _posture_projected_jacobian = _J_posture * _N_ns * N_prec;
//         _current_task_range = Sai2Model::matrixRangeBasis(_posture_projected_jacobian); 
//         op_space_matrices =
//             _robot->operationalSpaceMatrices(_current_task_range.transpose() * _posture_projected_jacobian);
//         _M_partial = op_space_matrices.Lambda;
//         _N = op_space_matrices.N * _N_ns;  // _N = N_partial_joint * N_ns 
//     }

//     return SingularityOpSpaceMatrices{_projected_jacobian_ns,
//                                       _Lambda_ns,
//                                       _N_ns,
//                                       _task_range_ns,
//                                       _projected_jacobian_s,
//                                       _Lambda_s,
//                                       _N_s,
//                                       _task_range_s};
// }

void SingularityHandler::classifySingularity(const MatrixXd& singular_task_range) {
    // memory of entering conditions 
    if (_sing_type.first == NO_SINGULARITY) {
        _q_prior.first = _robot->q();
        _dq_prior.first = _robot->dq();
    }

    if (singular_task_range.norm() == 0) {
        _sing_type.first = NO_SINGULARITY;
        // clear singularity direction queue 
        while (!_sing_direction_buffer.first.empty()) {
            _sing_direction_buffer.first.pop();
        }
        return;
    }

    // fill buffer with the most singular task range
    _sing_direction_buffer.first.push(singular_task_range.rightCols(1));
    Vector6d last_direction = _sing_direction_buffer.first.back();
    Vector6d current_direction = _sing_direction_buffer.first.front();
    if (std::abs(current_direction.dot(last_direction)) > _type_1_tol) {
        _sing_type.first = TYPE_1_SINGULARITY;
    } else {
        _sing_type.first = TYPE_1_SINGULARITY;
        // _sing_type.first = TYPE_2_SINGULARITY;
    }
}

void SingularityHandler::classifyLinearSingularity(const MatrixXd& singular_task_range) {
    // memory of entering conditions 
    if (_sing_type.first == NO_SINGULARITY) {
        _q_prior.first = _robot->q();
        _dq_prior.first = _robot->dq();
    }

    if (singular_task_range.norm() == 0) {
        _sing_type.first = NO_SINGULARITY;
        // clear singularity direction queue 
        while (!_sing_direction_buffer.first.empty()) {
            _sing_direction_buffer.first.pop();
        }
        return;
    }

    // fill buffer with the most singular task range
    _sing_direction_buffer.first.push(singular_task_range.rightCols(1));
    Vector6d last_direction = _sing_direction_buffer.first.back();
    Vector6d current_direction = _sing_direction_buffer.first.front();
    if (std::abs(current_direction.dot(last_direction)) > _type_1_tol) {
        _sing_type.first = TYPE_1_SINGULARITY;
    } else {
        _sing_type.first = TYPE_1_SINGULARITY;
        // _sing_type.first = TYPE_2_SINGULARITY;
    }
}

void SingularityHandler::classifyAngularSingularity(const MatrixXd& singular_task_range) {
    if (_sing_type.second == NO_SINGULARITY) {
        _q_prior.second = _robot->q();
        _dq_prior.second = _robot->dq();
    }

    if (singular_task_range.norm() == 0) {
        _sing_type.second = NO_SINGULARITY;
        while (!_sing_direction_buffer.second.empty()) {
            _sing_direction_buffer.second.pop();
        }
        return;
    }

    // fill buffer with the most singular task range
    _sing_direction_buffer.second.push(singular_task_range.rightCols(1));
    Vector6d last_direction = _sing_direction_buffer.second.back();
    Vector6d current_direction = _sing_direction_buffer.second.front();
    if (std::abs(current_direction.dot(last_direction)) > _type_1_tol) {
        _sing_type.second = TYPE_1_SINGULARITY;
    } else {
        _sing_type.second = TYPE_1_SINGULARITY;
        // _sing_type.second = TYPE_2_SINGULARITY;
    }
}

VectorXd SingularityHandler::computeTorques(const VectorXd& unit_mass_force, const VectorXd& force_related_terms)
{
    VectorXd joint_strategy_torques = VectorXd::Zero(_robot->dof());
    VectorXd unit_torques = VectorXd::Zero(_robot->dof());

    // debug
    if (_sing_type.first != NO_SINGULARITY) {
        std::cout << "Singularity: " << singularity_labels[_sing_type.first] << "\n";
    }

    /*
        Type 1 strategy: joint damping towards limit; joint holding + damping away from limit
        Type 2 strategy: open-loop torque towards q_prior initially, proportional to dot(unit mass force, singular direction)
    */
    if (_sing_type.first == TYPE_1_SINGULARITY) {
        if ((_dq_prior.first).dot(_robot->dq()) > 0) {
            unit_torques = - _kp * (_robot->q() - _q_prior.first) - _kv * _robot->dq();  
            // unit_torques = - _kv * _robot->dq();
        } else {
            unit_torques = - _kp * (_robot->q() - _q_prior.first) - _kv * _robot->dq();  
        }
        joint_strategy_torques = (_current_task_range.transpose() * _posture_projected_jacobian).transpose() * \
                                            _M_partial * _current_task_range.transpose() * unit_torques;

    } else if (_sing_type.first == TYPE_2_SINGULARITY) {
        // apply open-loop torque proportional to dot(unit mass force, singular direction)
        // if multiple columns of _U_s, then use the vector average
        _M_partial.setIdentity();  
        double fTd = unit_mass_force.head(3).normalized().transpose() * _linear_task_range_s.col(0);
        unit_torques = fTd * _type_2_torque_vector;
        joint_strategy_torques = (_current_task_range.transpose() * _posture_projected_jacobian).transpose() * \
                                            _M_partial * _current_task_range.transpose() * unit_torques;
    }

    /*
        Combine non-singular and singular torques:
        - tau = tau_{ns} + \alpha_linear * tau_{s, linear} + (1 - \alpha_linear) * linear_joint_strategy + 
                            \alpha_angular * tau_{s, angular} + (1 - \alpha_angular) * angular_joint_strategy 
        - tau = tau_{ns} + J_{s}^{T} [alpha_linear; alpha_angular; ...] * Lambda_s * F_s  
    */

    // // debug
    // std::cout << "J_ns: \n" << _projected_jacobian_ns << "\n";
    // std::cout << "Lambda_ns: \n" << _Lambda_ns << "\n";
    // std::cout << "Task_range_ns: \n" << _task_range_ns << "\n";

    if (_sing_type.first == NO_SINGULARITY) {
        // std::cout << "No singularity torque computation\n";
        // std::cout << "task range ns: \n" << _task_range_ns << "\n";
        return _projected_jacobian_ns.transpose() * (_Lambda_ns * _task_range_ns.transpose() * unit_mass_force + \
                            _task_range_ns.transpose() * force_related_terms);
    } else {
        // debug
        // std::cout << "Linear task range s: \n" << _linear_task_range_s << "\n";
        // std::cout << "Angular task range s: \n" << _angular_task_range_s << "\n";
        std::cout << "alpha: \n" << _alpha << "\n";
        std::cout << "Lambda_s: \n" << _Lambda_s << "\n";  
        // std::cout << "linear joint torques: " << linear_joint_strategy_torques.transpose() << "\n";
        // std::cout << "angular joint torques: " << angular_joint_strategy_torques.transpose() << "\n";

        VectorXd singular_task_force = _alpha * _Lambda_s * _task_range_s.transpose() * unit_mass_force + \
                                            _task_range_s.transpose() * force_related_terms;
        VectorXd tau_s = _projected_jacobian_s.transpose() * singular_task_force + \
                            (1 - _alpha) * joint_strategy_torques;
        VectorXd tau_ns = _projected_jacobian_ns.transpose() * (_Lambda_ns * _task_range_ns.transpose() * unit_mass_force + \
                            _task_range_ns.transpose() * force_related_terms);
        return tau_ns + tau_s;
    }
                 
}

MatrixXd SingularityHandler::getBlockMatrix(const MatrixXd& A, const MatrixXd& B) {
    MatrixXd diagonal_matrix;
    if (A.norm() == 0 && B.norm() == 0) {
        return MatrixXd::Zero(6, 1);
    } else if (B.norm() == 0) {
        diagonal_matrix = MatrixXd::Zero(6, A.cols());
        diagonal_matrix.block(0, 0, 3, A.cols()) = A;
        return diagonal_matrix;
    } else if (A.norm() == 0) {
        diagonal_matrix = MatrixXd::Zero(6, B.cols());
        diagonal_matrix.block(3, 0, 3, B.cols()) = B;
        return diagonal_matrix;
    }

    int rows = A.rows() + B.rows();
    int cols = A.cols() + B.cols();
    diagonal_matrix = MatrixXd::Zero(rows, cols);

    int current_row = 0;
    int current_col = 0;

    diagonal_matrix.block(current_row, current_col, A.rows(), A.cols()) = A;
    current_row += A.rows();
    current_col += A.cols();

    diagonal_matrix.block(current_row, current_col, B.rows(), B.cols()) = B;
    current_row += B.rows();
    current_col += B.cols();

    return diagonal_matrix;
}

VectorXd SingularityHandler::getSigmaValues() {
    return _e_values;
}

}  // namespace