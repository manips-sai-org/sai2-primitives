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
                                       const double& type_1_tol) : _robot(robot), \
                                       _J_posture(J_posture), _type_1_tol(type_1_tol), _q_prior(robot->q())
{
    _type_2_torque_vector = VectorXd::Zero(_robot->dof());
    auto joint_limits = _robot->jointLimits();
    for (int i = 0; i < joint_limits.size(); ++i) {
        _type_2_torque_vector(i) = _type_2_torque_ratio * joint_limits[i].effort;
    }
}

void SingularityHandler::updateTaskModel(const MatrixXd& full_jacobian,
                                         const MatrixXd& projected_jacobian_ns,
                                         const MatrixXd& projected_jacobian_s,
                                         const MatrixXd& orthogonal_projection_ns, 
                                         const MatrixXd& orthogonal_projection_s, 
                                         const MatrixXd& Lambda_ns, 
                                         const MatrixXd& Lambda_s) 
{
    _full_jacobian = full_jacobian;
    _projected_jacobian_ns = projected_jacobian_ns;
    _projected_jacobian_s = projected_jacobian_s;
    _orthogonal_projection_ns = orthogonal_projection_ns;
    _orthogonal_projection_s = orthogonal_projection_s;
    _Lambda_ns = Lambda_ns;
    _Lambda_s = Lambda_s;                                        
}

MatrixXd SingularityHandler::updateNullspace(const MatrixXd& N_ns, const MatrixXd& U_J_s, const double& alpha) {
    _alpha = alpha;
    _U_J_s = U_J_s;
    _N = N_ns;  // from nonsingular task * N_prec
    _projected_jacobian = _J_posture * _N;
    _current_task_range = Sai2Model::matrixRangeBasis(_projected_jacobian); 
    _Jbar = _robot->MInv() * _full_jacobian.transpose() * (_Lambda_s + _Lambda_ns);

    if (_alpha == 1) {
        _sing_type = NONE;
		return _N;
	}

	Sai2Model::OpSpaceMatrices op_space_matrices =
		_robot->operationalSpaceMatrices(_current_task_range.transpose() * _projected_jacobian);
    _M_partial = op_space_matrices.Lambda;
    // _Jbar = op_space_matrices.Jbar;
	_N = op_space_matrices.N;

    // classify singularity 
    classifySingularity();

    return _N;
}

void SingularityHandler::classifySingularity() 
{
    if (_sing_type == NONE) {
        _q_prior = _robot->q();
        _dq_prior = _robot->dq();
    }

    // compute the singular vectors of the nullspace matrix
    // type 1 if _J_bar * _current_task_range.transpose() * _projected_jacobian has a direction in the singular direction
    // type 2 if not 
    Sai2Model::SvdData JNJb_svd = Sai2Model::matrixSvd(_J_posture * _current_task_range.transpose() * _Jbar);
    MatrixXd dot_product_matrix = JNJb_svd.U.transpose() * _U_J_s;
    std::cout << "JNJb_svd_U:\n" << JNJb_svd.U << "\n\n";
    std::cout << "U_J_s: \n" << _U_J_s << "\n\n";
    std::cout << "dot product matrix max: \n" << dot_product_matrix.maxCoeff() << "\n";
    if ((dot_product_matrix.array().abs() > _type_1_tol).any()) {
        _sing_type = TYPE_1;
    } else {
        // _sing_type = TYPE_2;
        _sing_type = TYPE_1;
    }
}

VectorXd SingularityHandler::computeTorques(const VectorXd& unit_mass_force)
{
    // joint strategy
    VectorXd joint_strategy_torques = VectorXd::Zero(_robot->dof());
    VectorXd unit_torques = VectorXd::Zero(_robot->dof());
    if (_sing_type == TYPE_1) {
        unit_torques = - _kp * (_robot->q() - _q_prior) - _kv * _robot->dq();  // apply desired joint positions
        // if (((_desired_position.transpose() * unit_mass_force).array() < 0).any()) {
            // unit_torques = - _kp * (_robot->q() - _q_prior) - _kv * _robot->dq();  // apply desired joint positions
        // } else {
            // unit_torques = - _kp * (_robot->q() - _q_prior) - _kv * _robot->dq();  // apply desired joint positions
            // unit_torques = - _kv * _robot->dq();  // apply damping 
        // }
        joint_strategy_torques = (_current_task_range.transpose() * _projected_jacobian).transpose() * _M_partial * _current_task_range.transpose() * unit_torques;
    } else if (_sing_type == TYPE_2) {
        double fTd = unit_mass_force.normalized().transpose() * _U_J_s.col(0);
        unit_torques = fTd * _type_2_torque_vector;
        joint_strategy_torques = (_current_task_range.transpose() * _projected_jacobian).transpose() * _M_partial * _current_task_range.transpose() * unit_torques;
    }

    VectorXd nonsingular_task_torques = _projected_jacobian_ns.transpose() * \
                                        _Lambda_ns * \
                                        _orthogonal_projection_ns * unit_mass_force;
    VectorXd singular_task_torques = _projected_jacobian_s.transpose() * \
                                     _Lambda_s * \
                                     _orthogonal_projection_s * unit_mass_force;
 
    return nonsingular_task_torques + _alpha * singular_task_torques + (1 - _alpha) * joint_strategy_torques;
}

}  // namespace