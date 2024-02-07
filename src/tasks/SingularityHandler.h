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

#ifndef SAI2_PRIMITIVES_SINGULARITY_HANDLER_
#define SAI2_PRIMITIVES_SINGULARITY_HANDLER_

#include "Sai2Model.h"
#include <Eigen/Dense>
#include <queue>

using namespace Eigen;
namespace Sai2Primitives {

enum SingularityType {
    NO_SINGULARITY = 0,
    TYPE_1_SINGULARITY,
    TYPE_2_SINGULARITY
};

const std::vector<std::string> singularity_labels {"No Singularity", "Type 1 Singularity", "Type 2 Singularity"};

struct SingularityOpSpaceMatrices {
    MatrixXd projected_jacobian_ns;
    MatrixXd Lambda_ns;
    MatrixXd N_ns;
    MatrixXd task_range_ns;
    MatrixXd projected_jacobian_s;
    MatrixXd Lambda_s;
    MatrixXd N_s;
    MatrixXd task_range_s;
};

class SingularityHandler {
public:
    SingularityHandler(std::shared_ptr<Sai2Model::Sai2Model> robot,
                       const MatrixXd& J_posture,
                       const double type_1_tol = 0.8,
                       const int buffer_size = 50);
    SingularityOpSpaceMatrices updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec);
    void classifyLinearSingularity(const MatrixXd& singular_task_range);
    void classifyAngularSingularity(const MatrixXd& singular_task_range);
    VectorXd computeTorques(const VectorXd& unit_mass_force, const VectorXd& force_related_terms);

    // getters and setters 
    void setSingularityBounds(const double linear_sing_tol_min,
                              const double linear_sing_tol_max,
                              const double angular_sing_tol_min,
                              const double angular_sing_tol_max) 
    {
        _linear_sing_tol_min = linear_sing_tol_min;
        _linear_sing_tol_max = linear_sing_tol_max;
        _angular_sing_tol_min = angular_sing_tol_min;
        _angular_sing_tol_max = angular_sing_tol_max;
    }

    void setLambda(const MatrixXd& Lambda_ns, const MatrixXd& Lambda_s) {
        _Lambda_ns = Lambda_ns;
        _Lambda_s = Lambda_s;
    }

    MatrixXd getNullspace() { return _N; };

    void setGains(const double& kp, const double& kv) {
        _kp = kp;
        _kv = kv;
    }

    void setSingularity(const SingularityType& linear, const SingularityType& angular) {
        _sing_type.first = linear;
        _sing_type.second = angular;
    }

    void setTorqueRatio(const double& ratio) {
        _type_2_torque_ratio = ratio;
    }

    MatrixXd getBlockMatrix(const MatrixXd& A, const MatrixXd& B);

    VectorXd getSigmaValues();

private:
    std::shared_ptr<Sai2Model::Sai2Model> _robot;
    std::pair<SingularityType, SingularityType> _sing_type;
    MatrixXd _J_posture;

    // type 1 specifications
    std::pair<Eigen::VectorXd, Eigen::VectorXd> _q_prior;
    std::pair<Eigen::VectorXd, Eigen::VectorXd> _dq_prior; 
    std::pair<std::queue<Eigen::Vector3d>, std::queue<Eigen::Vector3d>> _sing_direction_buffer;
    double _kp, _kv;
    double _type_1_tol;

    // type 2 specifications 
    double _type_2_torque_ratio;  // use X% of the max joint torque 
    VectorXd _type_2_torque_vector;

    // singularity information
    double _alpha_linear, _alpha_angular;
    MatrixXd _N;

    // singularity bounds 
    double _linear_sing_tol_min, _linear_sing_tol_max, _angular_sing_tol_min, _angular_sing_tol_max;
    MatrixXd _linear_task_range_ns, _linear_task_range_s;
    MatrixXd _angular_task_range_ns, _angular_task_range_s;
    MatrixXd _task_range_ns, _task_range_s;
    MatrixXd _projected_jacobian_ns, _projected_jacobian_s;
    MatrixXd _Lambda_ns, _N_ns;
    MatrixXd _Lambda_s, _N_s, _Jbar_s;
    MatrixXd _Jbar_s_linear, _Jbar_s_angular;

    // joint task
    MatrixXd _posture_projected_jacobian, _current_task_range, _M_partial;
    MatrixXd _projected_jacobian_s_linear, _projected_jacobian_s_angular;

    // debug
    VectorXd _linear_singular_values;

};

}  // namespace

#endif