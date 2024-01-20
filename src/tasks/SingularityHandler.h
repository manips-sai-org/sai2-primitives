/**
 * @file SingularityHandler.h
 * @author William Chong (wmchong@stnaford.edu)
 * @brief 
 * @version 0.1
 * @date 2024-01-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once 

#include "Sai2Model.h"

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <string>
#include <algorithm>

using namespace Eigen;
namespace Sai2Primitives {

enum SingularityType {
    NONE = 0,
    TYPE_1,
    TYPE_2
};

class SingularityHandler {
public:
    SingularityHandler(std::shared_ptr<Sai2Model::Sai2Model> robot,
                       const MatrixXd& J_posture,
                       const double& type_1_tol = 0.95);
    void updateTaskModel(const MatrixXd& full_jacobian,
                         const MatrixXd& projected_jacobian_ns,
                         const MatrixXd& projected_jacobian_s,
                         const MatrixXd& orthogonal_projection_ns, 
                         const MatrixXd& orthogonal_projection_s,
                         const MatrixXd& Lambda_ns, 
                         const MatrixXd& Lambda_s);
    MatrixXd updateNullspace(const MatrixXd& N_ns, const MatrixXd& U_J_s, const double& alpha);
    MatrixXd getNullspace() { return _N; };
    void classifySingularity();
    VectorXd computeTorques(const VectorXd& unit_mass_force);

private:
    std::shared_ptr<Sai2Model::Sai2Model> _robot;
    int _sing_type = NONE;
    MatrixXd _J_posture;

    // type 1 specifications
    VectorXd _q_prior, _dq_prior;
    double _kp = 50;
    double _kv = 14;
    double _type_1_tol = 0.7;

    // type 2 specifications 
    double _type_2_torque_ratio = 0.05;  // use X% of the max joint torque 
    VectorXd _type_2_torque_vector;

    // singularity information
    double _alpha = 1;
    MatrixXd _U_J_s;

    // model
    MatrixXd _full_jacobian;
    MatrixXd _projected_jacobian;
    MatrixXd _current_task_range, _joint_orthogonal_projection;
    MatrixXd _N;
    MatrixXd _Jbar;
    MatrixXd _M_partial;
    MatrixXd _Lambda_ns, _Lambda_s;
	MatrixXd _orthogonal_projection_ns, _orthogonal_projection_s;
	MatrixXd _projected_jacobian_s, _projected_jacobian_ns;

};

}  // namespace
