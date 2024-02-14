/**
 * @file SingularityHandler.h
 * @author William Chong (wmchong@stnaford.edu)
 * @brief This class handles near-singularity cases 
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
                       const int& task_rank,
                       const MatrixXd& J_posture,
                       const double& s_tol = 1e-3,
                       const double& type_1_tol = 0.8,
                       const double& type_2_torque_ratio = 0.02,
                       const int& buffer_size = 10);
    SingularityOpSpaceMatrices updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec);
    void classifySingularity(const MatrixXd& singular_task_range);
    VectorXd computeTorques(const VectorXd& unit_mass_force, const VectorXd& force_related_terms);

    // getters and setters 
    MatrixXd getNullspace() { return _N; };

    VectorXd getSigmaValues();

    void setSingularityBounds(const double& s_min, const double& s_max) {
        _s_min = s_min;
        _s_max = s_max;
    }

    void setLambda(const MatrixXd& Lambda_ns, const MatrixXd& Lambda_s) {
        _Lambda_ns = Lambda_ns;
        _Lambda_s = Lambda_s;
    }

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
    // singularity setup
    std::shared_ptr<Sai2Model::Sai2Model> _robot;
    SingularityType _sing_type;
    int _task_rank;
    MatrixXd _J_posture;
    VectorXd _joint_midrange;
    int _buffer_size;
    std::queue<SingularityType> _sing_history;
    std::queue<Eigen::Matrix<double, 6, 1>> _sing_direction_buffer;

    // type 1 specifications
    VectorXd _q_prior;
    VectorXd _dq_prior; 
    double _kp, _kv;
    double _type_1_tol;

    // type 2 specifications 
    double _type_2_torque_ratio;  // use X% of the max joint torque 
    VectorXd _type_2_torque_vector;

    // singularity information
    double _s_abs_tol;  
    double _s_min, _s_max;
    double _alpha;
    MatrixXd _N;
    MatrixXd _task_range_ns, _task_range_s;
    MatrixXd _projected_jacobian_ns, _projected_jacobian_s;
    MatrixXd _Lambda_ns, _N_ns, _Jbar_ns;
    MatrixXd _Lambda_s, _N_s, _Jbar_s;

    // joint task
    MatrixXd _posture_projected_jacobian, _current_task_range, _M_partial;

    // debug
    VectorXd _s_values;

    template<typename T>
    bool allElementsSame(const std::queue<T>& q) {
        if (q.empty()) {
            return true; // An empty queue has all elements the same (technically)
        }

        // Get the first element
        T firstElement = q.front();

        // Iterate through the queue
        std::queue<T> tempQueue = q; // Create a copy of the original queue
        while (!tempQueue.empty()) {
            // If any element is different from the first element, return false
            if (tempQueue.front() != firstElement) {
                return false;
            }
            tempQueue.pop(); // Remove the front element
        }

        return true; // All elements are the same
    }
};

}  // namespace

#endif