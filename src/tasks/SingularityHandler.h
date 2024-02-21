/*
 * SingularityHandling.h
 *
 *      This class creates a singularity classifying and handling class for 
 * type 1 and type 2 singularities. The singularity strategy linearly blends 
 * the torques from the singular task directions and the torques from the 
 * singularity torque strategy, and adds this to the torques from the 
 * non-singular task directions.
 *
 *      Author: William Chong 
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

enum DynamicDecouplingType {
    FULL_DYNAMIC_DECOUPLING = 0,	 // use the real Lambda matrix
    IMPEDANCE,					 // use Identity for the mass matrix
    BOUNDED_INERTIA_ESTIMATES,	 // use a Lambda computed from a saturated
                                 // joint space mass matrix
};            

class SingularityHandler {
public:
    /**
     * @brief Construct a new Singularity Handler task
     * 
     * @param robot robot model from motion force task
     * @param task_rank rank of the motion force task after partial task projection
     * @param J_posture partial joint selection matrix of the controlled point from motion force task
     * @param s_abs_tol tolerance to declare task completely singular if all singular values from the projected
     * jacobian are under this value
     * @param type_1_tol tolerance for the dot product between the first and last singular direction in the 
     * singular direction buffer for which a type 1 singularity is classified
     * @param type_2_torque_ratio percentage of the max joint torque to use for the type 2 singularity strategy
     * @param queue_size size of the queue for storing past singular directions
     * @param verbose set to true to print singularity status every timestep 
     */
    SingularityHandler(std::shared_ptr<Sai2Model::Sai2Model> robot,
                       const int& task_rank,
                       const MatrixXd& J_posture,
                       const double& s_abs_tol = 1e-3,
                       const double& type_1_tol = 0.8,
                       const double& type_2_torque_ratio = 0.01,
                       const int& queue_size = 10,
                       const bool& verbose = false);

    /**
     * @brief Updates the model quantities for the singularity handling task, and performs singularity classification
     * 
     * @param projected_jacobian Projected jacobian from motion force task
     * @param N_prec Nullspace of preceding tasks from motion force task
     */
    void updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec);

    /**
     * @brief Classifies the singularity based on the column of U (singular columns) from the smallest singular value. 
     * U is obtained from the thin SVD decomposition of the projected jacobian. The singular column is placed in a queue,
     * and type 1 is automatically classified. Type 2 is classified if all elements in the queue are Type 2.
     * 
     * @param singular_task_range Singular task range corresponding to the columns of U from SVD
     */
    void classifySingularity(const MatrixXd& singular_task_range);

    /**
     * @brief Computes the torques from the singularity handling. If the projected jacobian doesn't have
     * a condition number below the _s_max value, then the torque is computed as usual with _projected_jacobian_ns.
     * If the projected jacobian has a condition number below the _s_max value, then the torque is computed as
     * \tau = \tau_{ns} + (1 - \_alpha) * \tau_{joint strategy} + \alpha * \tau_{s} where \alpha is the linear blending ratio, 
     * \tau_{ns} is the torque computed from the non-singular terms, \tau_{s} is the torque computed from the singular 
     * terms, and \tau_{joint strategy} is the torque computed from the singularity strategy.
     * 
     * @param unit_mass_force Desired unit mass forces from motion force task
     * @param force_related_terms Desired forces from motion force task
     * @return VectorXd Torque vector 
     */
    VectorXd computeTorques(const VectorXd& unit_mass_force, const VectorXd& force_related_terms);

    /**
     * @brief Set the dynamic decoupling type 
     * 
     * @param type DynamicDecoupling type 
     */
    void setDynamicDecouplingType(const DynamicDecouplingType& type) {
        _dynamic_decoupling_type = type;
    }

    /**
     * @brief Get the nullspace of the singularity task
     * 
     * @return MatrixXd nullspace 
     */
    MatrixXd getNullspace() { return _N; };

    /**
     * @brief Get the vector of singular values for the projected jacobian
     * 
     * @return VectorXd singular values vector
     */
    VectorXd getSigmaValues() {
        return _s_values;
    }

    /**
     * @brief Get the non-singular op-space matrices
     * 
     * @return Sai2Model::OpSpaceMatrices non-singular op-space matrices
     */
    Sai2Model::OpSpaceMatrices getNonSingularOpSpaceMatrices() {
        return Sai2Model::OpSpaceMatrices(_projected_jacobian_ns, _Lambda_ns, _Jbar_ns, _N_ns);
    }

    /**
     * @brief Get the singular op-space matrices
     * 
     * @return Sai2Model::OpSpaceMatrices singular op-space matrices
     */
    Sai2Model::OpSpaceMatrices getSingularOpSpaceMatrices() {
        return Sai2Model::OpSpaceMatrices(_projected_jacobian_s, _Lambda_s, _Jbar_s, _N_s);
    }

    /**
     * @brief Set the singularity bounds for torque blending based on the condition number
     * The linear blending coefficient \alpha is computed as \alpha = (s - _s_min) / (_s_max - _s_min),
     * and is clamped between 0 and 1.
     * 
     * @param s_min condition number value to only use singularity strategy torques
     * @param s_max condition number value to start blending
     */
    void setSingularityBounds(const double& s_min, const double& s_max) {
        _s_min = s_min;
        _s_max = s_max;
    }

    /**
     * @brief Set the gains for the partial joint task for the singularity strategy
     * 
     * @param kp position gain
     * @param kv velocity gain
     */
    void setGains(const double& kp, const double& kv) {
        _kp = kp;
        _kv = kv;
    }

    /**
     * @brief Set the singularity type for the current timestep
     * 
     * @param type Singularity type (none, type 1, or type 2)
     */
    void setSingularity(const SingularityType& type) {
        _sing_type = type;
    }

    /**
     * @brief Set the torque ratio used for type 2 singularity strategy
     * 
     * @param ratio Ratio of the specific joint's max torque to use 
     */
    void setTorqueRatio(const double& ratio) {
        _type_2_torque_ratio = ratio;
    }

private:
    // singularity setup
    std::shared_ptr<Sai2Model::Sai2Model> _robot;
    DynamicDecouplingType _dynamic_decoupling_type;
    int _task_rank;
    MatrixXd _J_posture;
    VectorXd _joint_midrange;
    bool _verbose;

    // singularity history 
    SingularityType _sing_type;
    int _queue_size;
    std::queue<SingularityType> _sing_history;
    std::queue<Eigen::Matrix<double, 6, 1>> _sing_direction_queue;

    // type 1 specifications
    VectorXd _q_prior, _dq_prior;
    double _kp, _kv;
    double _type_1_tol;

    // type 2 specifications 
    double _type_2_torque_ratio;  // use X% of the max joint torque 
    VectorXd _type_2_torque_vector;

    // model quantities 
    double _s_abs_tol;  
    double _s_min, _s_max;
    double _alpha;
    MatrixXd _N;
    MatrixXd _task_range_ns, _task_range_s;
    MatrixXd _projected_jacobian_ns, _projected_jacobian_s;
    MatrixXd _Lambda_ns, _N_ns, _Jbar_ns;
    MatrixXd _Lambda_s, _N_s, _Jbar_s;
    MatrixXd _Lambda_ns_modified, _Lambda_s_modified;

    // joint task quantities 
    MatrixXd _posture_projected_jacobian, _current_task_range, _M_partial;

    // logging 
    VectorXd _s_values;

    /**
     * @brief Checks if all elements in a queue are the same
     */
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