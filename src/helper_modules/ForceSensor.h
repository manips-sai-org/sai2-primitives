/**
 * @file ForceSensor.h
 * @author William Chong (wmchong@stnaford.edu)
 * @brief 
 * @version 0.1
 * @date 2024-02-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "Sai2Model.h"

namespace Sai2Primitives {

struct ForceMeasurement {
    Vector3d force;
    Vector3d moment;
};

class ForceSensor {
public:
    ForceSensor(std::shared_ptr<Sai2Model::Sai2Model> robot,
                const Vector3d& gravity = Vector3d(0, 0, -9.81)) {
        _robot = robot;
        _gravity = gravity;
    }

    void setForceSensorFrame(const string& link_name, 
                             const Affine3d& transformation_in_link) {
        _link_name = link_name;
        _T_link_to_sensor = transformation_in_link;
    }
    void setToolInertia(const double mass, const Vector3d& com, const Matrix3d& inertia) {
        _mass = mass;
        _com = com;
        _inertia = inertia;
    }
    ForceMeasurement getCalibratedForceMoment(const Vector3d& raw_force, const Vector3d& raw_moment);

private:
    std::shared_ptr<Sai2Model::Sai2Model> _robot;
    Vector3d _gravity;
    // inertia
    double _mass;
    Vector3d _com;
    Matrix3d _inertia;
    // sensor location
    Affine3d _T_link_to_sensor;
    std::string _link_name;
    // bias
    Vector3d _force_bias, _moment_bias;
};

}  // namespace 