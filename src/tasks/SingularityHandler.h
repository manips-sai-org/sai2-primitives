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
#include <memory>

using namespace Eigen;
namespace Sai2Primitives {

struct SvdData {
    MatrixXd U, V;
    VectorXd s;
    SvdData(const MatrixXd& U,
            const VectorXd& s,
            const MatrixXd& V) : 
            U(U), s(s), V(V) { }
};

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
     * @param s_abs_tol tolerance to declare task completely singular if all singular values from the projected
     * jacobian are under this value
     * @param type_1_tol tolerance for the dot product between the first and last singular direction in the 
     * singular direction buffer for which a type 1 singularity is classified
     * @param type_2_torque_ratio percentage of the max joint torque to use for the type 2 singularity strategy
     * @param perturb_step_size the joint angle increment to classify singularity 
     * @param verbose set to true to print singularity status every timestep 
     */
    SingularityHandler(std::shared_ptr<Sai2Model::Sai2Model> robot,
                       const std::string& link_name,
                       const Affine3d& compliant_frame,
                       const int& task_rank,
                       const double& bie = 0.5,
                       const double& s_abs_tol = 1e-3,
                       const double& type_1_tol = 0.05,
                       const double& type_1_torque_ratio = 1e-1,
                       const double& type_2_torque_ratio = 1e-3,
                       const double& perturb_step_size = 1.5,
                       const int& buffer_size = 100,
                       const bool& verbose = true);

    /**
     * @brief Updates the model quantities for the singularity handling task, and performs singularity classification
     * 
     * @param projected_jacobian Projected jacobian from motion force task
     * @param N_prec Nullspace of preceding tasks from motion force task
     */
    void updateTaskModel(const MatrixXd& projected_jacobian, const MatrixXd& N_prec);

    /**
     * @brief Classifies the singularity based on a joint perturbation in the J_{s} N_{ns} direction since 
     * \delta x = J_{ns} \delta q + J_{s} N_{ns} \delta q_{0}. Type 1 singularity if the norm of 
     * \delta x is greater than the type 1 tolerance, otherwise type 2. Type 2 singularity will have a very
     * small \delta x norm.
     * 
     * @param projected_jacobian Projected jacobian for task 
     * @param singular_task_range Singular task range corresponding to the columns of U from SVD
     * @param singular_joint_task_range Singular task range corresponding to the columns of V from SVD
     */
    void classifySingularity(const MatrixXd& projected_jacobian, 
                             const MatrixXd& singular_task_range, 
                             const MatrixXd& singular_joint_task_range);

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
     * @brief Get the Svd Data object
     * 
     * @return MatrixXd 
     */
    SvdData getSvdData() {
        return SvdData(_svd_U, _svd_s, _svd_V);
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
     * @brief Set the torque ratio used for type 2 singularity strategy
     * 
     * @param ratio Ratio of the specific joint's max torque to use 
     */
    void setTorqueRatio(const double& ratio) {
        _type_2_torque_ratio = ratio;
    }

    void setBIE(const double& bie) {
        _bie = bie;
    }

    void setType1TorqueRatio(const double& ratio) {
        _type_1_torque_ratio = ratio;
    }

private:
    // singularity setup
    std::shared_ptr<Sai2Model::Sai2Model> _robot;
    DynamicDecouplingType _dynamic_decoupling_type;
    std::string _link_name;
    Affine3d _compliant_frame;
    int _task_rank;
    VectorXd _joint_midrange, _q_upper, _q_lower;
    bool _verbose;
    double _bie;

    // singularity information
    std::vector<SingularityType> _singularity_types;
    double _perturb_step_size;
    std::deque<SingularityType> _singularity_history;
    std::deque<MatrixXd> _singularity_task_range_history;
    int _type_1_counter, _type_2_counter;
    int _buffer_size;

    // type 1 specifications
    VectorXd _q_prior, _dq_prior;
    double _kp, _kv;
    double _type_1_tol;
    double _type_1_torque_ratio;
    VectorXd _type_1_torque_vector;

    // type 2 specifications
    double _type_2_torque_ratio;  // use X% of the max joint torque 
    VectorXd _type_2_torque_vector;

    // model quantities 
    double _s_abs_tol;  
    double _s_min, _s_max;
    double _alpha;
    MatrixXd _N;
    MatrixXd _task_range_ns, _task_range_s, _joint_task_range_s;
    MatrixXd _projected_jacobian_ns, _projected_jacobian_s;
    MatrixXd _non_truncated_projected_jacobian_s;
    MatrixXd _Lambda_ns, _Jbar_ns, _N_ns;
    MatrixXd _Lambda_s;
    MatrixXd _Lambda_ns_modified, _Lambda_s_modified;

    // joint task quantities 
    MatrixXd _posture_projected_jacobian, _M_partial;

    // svd logging  
    MatrixXd _svd_U, _svd_V;
    VectorXd _svd_s;

};

}  // namespace

#endif