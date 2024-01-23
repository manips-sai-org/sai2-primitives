/**
 * @file SingularityHandler.h
 * @author William Chong (wmchong@stnaford.edu)
 * @brief This class handles type 1 and type 2 singularity.
 *          For type 1, the joint task is blended with the force task in the singular space.
 *          For type 2, an open-loop torque is applied proportional to the dot product between 
 *          the force task and the singular direction and blended. The torque direction can be either direction
 *          since the singular direction can move pi/2 either direction to be orthogonal to the force vector
 *          since the singular direction only needs to be in-line with the force vector.
 * @version 0.1
 * @date 2024-01-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once 

#include "Sai2Model.h"
#include <Eigen/Dense>

using namespace Eigen;
namespace Sai2Primitives {

enum SingularityType {
    NO_SINGULARITY = 0,
    TYPE_1_SINGULARITY,
    TYPE_2_SINGULARITY
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
                         const MatrixXd& Lambda_s,
                         const MatrixXd& N_ns,
                         const MatrixXd& N_prec,
                         const MatrixXd& U_s,
                         const double& alpha);
    MatrixXd getNullspace() { return _N; };
    void classifySingularity();
    VectorXd computeTorques(const VectorXd& unit_mass_force);

    void setGains(const double& kp, const double& kv) {
        _kp = kp;
        _kv = kv;
    }

    void setSingularity(const SingularityType& type) {
        _sing_type = type;
    }

    void setTorqueRatio(const double& ratio) {
        _type_2_torque_ratio = ratio;
    }

private:
    std::shared_ptr<Sai2Model::Sai2Model> _robot;
    int _sing_type;
    MatrixXd _J_posture;

    // type 1 specifications
    VectorXd _q_prior, _dq_prior;
    double _kp, _kv;
    double _type_1_tol;

    // type 2 specifications 
    double _type_2_torque_ratio;  // use X% of the max joint torque 
    VectorXd _type_2_torque_vector;

    // singularity information
    double _alpha;
    MatrixXd _U_s;

    // model
    MatrixXd _full_jacobian;
    MatrixXd _projected_jacobian;
    MatrixXd _current_task_range, _joint_orthogonal_projection;
    MatrixXd _N_ns, _N_prec, _N;
    MatrixXd _Jbar;
    MatrixXd _M_partial;
    MatrixXd _Lambda_ns, _Lambda_s;
	MatrixXd _orthogonal_projection_ns, _orthogonal_projection_s;
	MatrixXd _projected_jacobian_s, _projected_jacobian_ns;

};

}  // namespace
