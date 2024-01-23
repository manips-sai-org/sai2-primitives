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
                                       const double& type_1_tol) : _robot(robot), 
                                       _J_posture(J_posture), _type_1_tol(type_1_tol), 
                                       _q_prior(robot->q()), _dq_prior(robot->dq())
{
    _type_2_torque_vector = VectorXd::Zero(_robot->dof());
    auto joint_limits = _robot->jointLimits();
    for (int i = 0; i < joint_limits.size(); ++i) {
        _type_2_torque_vector(i) = _type_2_torque_ratio * joint_limits[i].effort;
    }
    setGains(50, 14);
    setSingularity(NO_SINGULARITY);
    setTorqueRatio(0.02);
}

void SingularityHandler::updateTaskModel(const MatrixXd& full_jacobian,
                                         const MatrixXd& projected_jacobian_ns,
                                         const MatrixXd& projected_jacobian_s,
                                         const MatrixXd& orthogonal_projection_ns, 
                                         const MatrixXd& orthogonal_projection_s, 
                                         const MatrixXd& Lambda_ns, 
                                         const MatrixXd& Lambda_s,
                                         const MatrixXd& N,
                                         const MatrixXd& U_s,
                                         const double& alpha) 
{
    // update member variables 
    _full_jacobian = full_jacobian;
    _projected_jacobian_ns = projected_jacobian_ns;
    _projected_jacobian_s = projected_jacobian_s;
    _orthogonal_projection_ns = orthogonal_projection_ns;
    _orthogonal_projection_s = orthogonal_projection_s;
    _Lambda_ns = Lambda_ns;
    _Lambda_s = Lambda_s;             
    _N = N;
    _U_s = U_s;
    _alpha = alpha;

    // update nullspace
    _projected_jacobian = _J_posture * _N;
    _current_task_range = Sai2Model::matrixRangeBasis(_projected_jacobian);  

    if (_alpha == 1) {
        _sing_type = NO_SINGULARITY;
        return;  // _N = N_ns
	}

	Sai2Model::OpSpaceMatrices op_space_matrices =
		_robot->operationalSpaceMatrices(_current_task_range.transpose() * _projected_jacobian);
    _M_partial = op_space_matrices.Lambda;
	_N = op_space_matrices.N;  // _N = N_partial_joint * N_ns 

    // classify singularity 
    classifySingularity();

    return;
}

void SingularityHandler::classifySingularity() 
{
    if (_sing_type == NO_SINGULARITY) {
        _q_prior = _robot->q();
        _dq_prior = _robot->dq();
    }

    /*
        Type 1: If (J_bar^{T} J^{T}) is aligned with a _U_s column 
        Type 2: If not the above
    */

    _Jbar = _robot->MInv() * _full_jacobian.transpose() * (_Lambda_s + _Lambda_ns);
    Sai2Model::SvdData JbarJT = Sai2Model::matrixSvd(_Jbar.transpose() * (_current_task_range.transpose() * _projected_jacobian).transpose());
    MatrixXd dot_product_matrix = JbarJT.U.transpose() * _U_s;  

    if ((dot_product_matrix.array().abs() > _type_1_tol).any()) {
        std::cout << "Type 1 singularity\n";
        if (_sing_type != TYPE_2_SINGULARITY) {
            _sing_type = TYPE_1_SINGULARITY;
        }
    } else {
        std::cout << "Type 2 singularity\n";
        if (_sing_type != TYPE_1_SINGULARITY) {
            _sing_type = TYPE_2_SINGULARITY;
        }
    }
}

VectorXd SingularityHandler::computeTorques(const VectorXd& unit_mass_force)
{
    VectorXd joint_strategy_torques = VectorXd::Zero(_robot->dof());
    VectorXd unit_torques = VectorXd::Zero(_robot->dof());
    if (_sing_type == TYPE_1_SINGULARITY) {
        // apply joint holding torque to q when entering 
        unit_torques = - _kp * (_robot->q() - _q_prior) - _kv * _robot->dq();  
        joint_strategy_torques = (_current_task_range.transpose() * _projected_jacobian).transpose() * \
                                     _M_partial * _current_task_range.transpose() * unit_torques;
    } else if (_sing_type == TYPE_2_SINGULARITY) {
        // apply open-loop torque proportional to dot(unit mass force, singular direction)
        // if multiple columns of _U_s, then use the vector average
        _M_partial.setIdentity();  
        double fTd = unit_mass_force.normalized().transpose() * (_U_s.rowwise().mean()).normalized();
        unit_torques = fTd * _type_2_torque_vector;
        joint_strategy_torques = (_current_task_range.transpose() * _projected_jacobian).transpose() * \
                                    _M_partial * _current_task_range.transpose() * unit_torques;
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